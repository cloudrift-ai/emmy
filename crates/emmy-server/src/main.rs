use anyhow::Result;
use clap::Parser;
use emmy_server::{
    api::{App, router},
    worker::Worker,
};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::sync::{Semaphore, mpsc};

const ACTIVE_REQUESTS: usize = 1;
const DEFAULT_CONTEXT: usize = 4096;
const DEFAULT_PORT: u16 = 8000;
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Parser)]
#[command(version, about = "Single-request native Emmy text server")]
struct Args {
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    model: String,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = DEFAULT_PORT)]
    port: u16,
    #[arg(long, default_value_t = DEFAULT_CONTEXT)]
    max_model_len: usize,
}
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let (jobs, receiver) = mpsc::channel(ACTIVE_REQUESTS);
    let ready = Arc::new(AtomicBool::new(false));
    let shutdown = Arc::new(AtomicBool::new(false));
    let app = App {
        model: args.model,
        jobs,
        admission: Arc::new(Semaphore::new(ACTIVE_REQUESTS)),
        ready: ready.clone(),
        shutdown: shutdown.clone(),
    };
    let worker_ready = ready.clone();
    let worker_shutdown = shutdown.clone();
    let worker = std::thread::spawn(move || {
        let _readiness = Readiness(worker_ready.clone());
        match Worker::load(&args.artifact, args.max_model_len) {
            Ok(worker) => worker.run(receiver, worker_ready, worker_shutdown),
            Err(e) => eprintln!("native startup failed: {e:#}"),
        }
    });
    let listener = tokio::net::TcpListener::bind((args.host.as_str(), args.port)).await?;
    let signal = async move {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("signal handler");
        tokio::select! { _ = tokio::signal::ctrl_c() => (), _ = terminate.recv() => () }
        shutdown.store(true, Ordering::Release);
        ready.store(false, Ordering::Release);
        // A stuck driver operation cannot be cancelled safely in-process.
        tokio::spawn(async {
            tokio::time::sleep(SHUTDOWN_TIMEOUT).await;
            std::process::exit(1);
        });
    };
    axum::serve(listener, router(app))
        .with_graceful_shutdown(signal)
        .await?;
    tokio::task::spawn_blocking(move || worker.join())
        .await?
        .map_err(|_| anyhow::anyhow!("runtime thread panicked"))?;
    Ok(())
}

struct Readiness(Arc<AtomicBool>);
impl Drop for Readiness {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
