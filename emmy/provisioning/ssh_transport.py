"""SSH transport: run commands and write files on remote servers via SSH/SCP."""

import asyncio
import logging
import os
import signal
import tempfile

logger = logging.getLogger(__name__)

REMOTE_DEPLOY_DIR = "~/.local/share/emmy"


def ssh_base_args(server, ssh_key, ssh_port):
    """Build base SSH arguments."""
    args = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=20",
        "-o",
        "TCPKeepAlive=no",
    ]
    if ssh_key:
        args += ["-i", ssh_key]
    if ssh_port and ssh_port != 22:
        args += ["-p", str(ssh_port)]
    args.append(server)
    return args


def make_run_cmd(server, ssh_key, ssh_port, dry_run=False, *, local=False):
    """Create a run_cmd callable for SSH or local command execution."""

    async def run_cmd(command, stream=True, timeout=600, log_output=False):
        # Use sg to run docker commands under the docker group
        if local:
            full_cmd = command
        elif command.strip().startswith("docker"):
            escaped = command.replace('"', '\\"')
            full_cmd = f'sg docker -c "cd {REMOTE_DEPLOY_DIR} && {escaped}"'
        else:
            full_cmd = f"cd {REMOTE_DEPLOY_DIR} && {command}"
        if dry_run:
            logger.info(f"[dry-run] {'local' if local else f'ssh {server}'}: {full_cmd}")
            return 0, "", ""

        argv = ["bash", "-c", full_cmd] if local else [*ssh_base_args(server, ssh_key, ssh_port), full_cmd]

        proc = None
        try:
            use_pipe = not stream or log_output
            spawn = asyncio.ensure_future(
                asyncio.create_subprocess_exec(
                    *argv,
                    start_new_session=local,
                    stdout=asyncio.subprocess.PIPE if use_pipe else None,
                    stderr=asyncio.subprocess.PIPE if use_pipe else None,
                )
            )
            try:
                proc = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                # A cancel landing mid-spawn would make asyncio close the transport, which kills the
                # shell alone; finish the spawn so the handler below reaches the whole group.
                proc = await spawn
                raise

            if log_output:
                stdout_lines, stderr_lines = [], []

                async def _read_stream(pipe, lines, level):
                    async for raw_line in pipe:
                        line = raw_line.decode().rstrip("\n")
                        logger.log(level, line)
                        lines.append(line)

                await asyncio.wait_for(
                    asyncio.gather(
                        _read_stream(proc.stdout, stdout_lines, logging.INFO),
                        _read_stream(proc.stderr, stderr_lines, logging.ERROR),
                        proc.wait(),
                    ),
                    timeout=timeout,
                )
                return proc.returncode, "\n".join(stdout_lines), "\n".join(stderr_lines)
            else:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                stdout = "" if stream else (stdout_bytes.decode() if stdout_bytes else "")
                stderr = "" if stream else (stderr_bytes.decode() if stderr_bytes else "")
                return proc.returncode, stdout, stderr
        except (TimeoutError, asyncio.CancelledError) as exc:
            if not isinstance(exc, asyncio.CancelledError):
                logger.error(f"Command timed out after {timeout}s: {command}")
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL) if local else proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
            if isinstance(exc, asyncio.CancelledError):
                raise
            return 1, "", ""
        except Exception as e:
            logger.error(f"Error running SSH command: {e}")
            return 1, "", ""
        finally:
            if local and proc is not None:
                # Background descendants belong to this finite command, including when the
                # shell exits normally before they do. Never leak them into the next row.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    return run_cmd


async def scp_file(local_path, server, ssh_key, ssh_port, remote_path, timeout=300):
    """Copy a file to the remote server via SCP."""
    scp_args = [
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=20",
        "-o",
        "TCPKeepAlive=no",
    ]
    if ssh_key:
        scp_args += ["-i", ssh_key]
    if ssh_port and ssh_port != 22:
        scp_args += ["-P", str(ssh_port)]
    scp_args += [local_path, f"{server}:{remote_path}"]

    try:
        proc = await asyncio.create_subprocess_exec(
            *scp_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        stderr = stderr_bytes.decode() if stderr_bytes else ""
        return proc.returncode, stderr
    except TimeoutError:
        logger.error(f"SCP timed out after {timeout}s: {local_path} -> {server}:{remote_path}")
        proc.kill()
        await proc.wait()
        return 1, "timeout"


async def scp_from_remote(server, ssh_key, ssh_port, remote_path, local_path, timeout=300):
    """Copy a file (or directory) FROM the remote server via SCP.
    ``-r`` is unconditional: harmless on regular files, required when
    ``remote_path`` resolves to a directory (e.g.
    ``EMMY_DUMP_DIR``'s ``*.kernels/`` subdirs that would
    otherwise silently get skipped). Mirror of scp_file()."""
    scp_args = [
        "scp",
        "-r",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=20",
        "-o",
        "TCPKeepAlive=no",
    ]
    if ssh_key:
        scp_args += ["-i", ssh_key]
    if ssh_port and ssh_port != 22:
        scp_args += ["-P", str(ssh_port)]
    scp_args += [f"{server}:{remote_path}", local_path]

    try:
        proc = await asyncio.create_subprocess_exec(
            *scp_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        stderr = stderr_bytes.decode() if stderr_bytes else ""
        return proc.returncode, stderr
    except TimeoutError:
        logger.error(f"SCP timed out after {timeout}s: {server}:{remote_path} -> {local_path}")
        proc.kill()
        await proc.wait()
        return 1, "timeout"


def make_write_file(server, ssh_key, ssh_port, dry_run=False):
    """Create a write_file callable that SCPs files to the remote server."""

    async def write_file(path, content):
        remote_path = f"{REMOTE_DEPLOY_DIR}/{path}"
        if dry_run:
            logger.info(f"[dry-run] scp {path} -> {server}:{remote_path}")
            return

        # Write to a temp file locally, then SCP
        with tempfile.NamedTemporaryFile(mode="w", suffix=f"_{path}", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        try:
            rc, stderr = await scp_file(tmp_path, server, ssh_key, ssh_port, remote_path)
            if rc != 0:
                logger.error(f"Failed to SCP {path} to {server}:{remote_path}: {stderr}")
        finally:
            os.unlink(tmp_path)

    return write_file
