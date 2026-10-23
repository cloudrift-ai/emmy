"""Synthetic sequential completion sessions, executed inside the serving client container."""

import asyncio
import json
import logging
import random
import sys
import time

import httpx

logger = logging.getLogger(__name__)


def initial_length(mean_input: int, calls: int, output: int, new_input: int) -> int:
    """Center a linearly growing session on the requested mean input length."""
    length = mean_input - (calls - 1) * (output + new_input) // 2
    if calls < 1 or output < 1 or new_input < 0 or length < 1:
        raise ValueError('Session growth exceeds the requested mean input length')
    return length


async def measure(config, client):
    """Emit raw per-request usage and timing, preserving sequential dependencies within each session."""
    bench = config['benchmark']
    calls = bench['session_calls']
    count = bench['num_prompts']
    if count % calls:
        raise ValueError('num_prompts must be a multiple of session_calls')
    first = initial_length(bench['random_input_len'], calls, bench['random_output_len'], bench['session_new_tokens'])
    # Obtain ordinary token IDs from the server tokenizer; no local model download is needed by the client.
    response = await client.post(config['base_url'] + '/tokenize', json={
        'model': config['model'], 'prompt': 'Review code function return value variable tests error input output. ' * 32,
        'add_special_tokens': False,
    })
    response.raise_for_status()
    vocabulary = sorted(set(response.json()['tokens']))
    if bench['max_concurrency'] < 1 or count < 1:
        raise ValueError('Concurrency and prompt count must be positive')
    common = random.Random(bench.get('seed') or 0).choices(vocabulary, k=bench['random_prefix_len'])
    if len(common) >= first:
        raise ValueError('Shared prefix must be shorter than the first prompt')
    semaphore = asyncio.Semaphore(bench['max_concurrency'])
    started = time.perf_counter()

    async def session(index):
        rng = random.Random((bench.get('seed') or 0) + index)
        prompt = common + rng.choices(vocabulary, k=first - len(common))
        headers = {'X-data-parallel-rank': str(index % config['dp_size'])} if bench['session_affinity'] else {}
        async with semaphore:
            for turn in range(calls):
                before = time.perf_counter()
                body = {
                    'model': config['model'], 'prompt': prompt, 'max_tokens': bench['random_output_len'],
                    'temperature': 0, 'ignore_eos': True, 'return_token_ids': True,
                }
                response = await client.post(config['base_url'] + '/v1/completions', json=body, headers=headers)
                response.raise_for_status()
                result = response.json()
                after = time.perf_counter()
                if 'cached_tokens' not in (result['usage'].get('prompt_tokens_details') or {}):
                    raise ValueError('Server must report cached-token usage')
                tokens = result['choices'][0]['token_ids']
                if tokens is None or len(tokens) != bench['random_output_len']:
                    raise ValueError('Server did not return the requested generated token IDs')
                logger.info(json.dumps({
                    'session': index, 'turn': turn, 'start_s': before - started, 'end_s': after - started,
                    'prompt_length': len(prompt), 'usage': result['usage'],
                    'finish_reason': result['choices'][0]['finish_reason'],
                }))
                prompt.extend(tokens)
                prompt.extend(rng.choices(vocabulary, k=bench['session_new_tokens']))

    # TaskGroup cancels sibling sessions on failure; a partial run never reports success.
    async with asyncio.TaskGroup() as group:
        for index in range(count // calls):
            group.create_task(session(index))
    logger.info(json.dumps({'duration_s': time.perf_counter() - started, 'config': config}))


async def main(config):
    async with httpx.AsyncClient(timeout=3600, limits=httpx.Limits(max_connections=None)) as client:
        bench = config['benchmark']
        for index, concurrency in enumerate(bench['session_concurrencies'] or [bench['max_concurrency']]):
            current = {**config, 'benchmark': {**bench, 'max_concurrency': concurrency,
                       'seed': (bench.get('seed') or 0) + index * bench['num_prompts']}}
            await measure(current, client)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    asyncio.run(main(json.loads(sys.argv[1])))
