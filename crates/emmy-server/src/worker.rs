//! One thread owns text processing and CUDA state. Channel closure cancels future steps.
use anyhow::{Result, anyhow};
use emmy_runtime::{cuda::Device, generation::{Generator, Sampling}};
use std::{path::Path, sync::{Arc, atomic::{AtomicBool, Ordering}}, time::Duration};
use tokio::sync::mpsc;
use crate::{api::{ApiError, Job, Output}, text::{Stops, Text}};

const OUTPUT_POLL: Duration = Duration::from_millis(5);

pub struct Worker {
    generator: Generator,
    text: Text,
    context: usize,
    eos: Vec<i64>,
}
impl Worker {
    pub fn load(root: &Path, context: usize) -> Result<Self> {
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let config = &manifest["key"]["generation"];
        anyhow::ensure!(context > 0 && context <= config["context_length"].as_u64().unwrap_or(0) as usize, "invalid context capacity");
        let eos = serde_json::from_value(config["eos_ids"].clone())?;
        let text = Text::load(root)?;
        let generator = Generator::load(&Device::new(0)?, root)?;
        Ok(Self { generator, text, context, eos })
    }
    pub fn run(mut self, mut jobs: mpsc::Receiver<Job>, ready: Arc<AtomicBool>, shutdown: Arc<AtomicBool>) {
        ready.store(true, Ordering::Release);
        while let Some(job) = jobs.blocking_recv() {
            if shutdown.load(Ordering::Acquire) { break; }
            if let Err(error) = self.request(&job, &shutdown) {
                // No retry and no new work after any execution/text-decoding failure.
                eprintln!("native request failed: {error:#}");
                ready.store(false, Ordering::Release);
                send(&job, Output::Error(ApiError::unavailable()), &shutdown);
                break;
            }
            // advance synchronizes before returning; only now may admission be released.
        }
        ready.store(false, Ordering::Release);
    }
    fn request(&mut self, job: &Job, shutdown: &AtomicBool) -> Result<()> {
        let r = &job.request;
        let prompt = if let Some(messages) = &r.messages { self.text.render(messages) }
            else { Ok(r.prompt.clone().unwrap_or_default()) };
        let ids = prompt.and_then(|p| self.text.encode(&p, r.messages.is_some()));
        let ids = match ids {
            Ok(ids) if !ids.is_empty() && ids.len() <= self.context && r.budget() <= self.context - ids.len() => ids,
            Ok(_) => { send(job, Output::Error(ApiError::invalid("prompt and output budget must fit the context")), shutdown); return Ok(()); }
            Err(e) => { send(job, Output::Error(ApiError::invalid(e.to_string())), shutdown); return Ok(()); }
        };
        if !send(job, Output::Started(ids.len()), shutdown) { return Ok(()); }
        if r.budget() == 0 {
            send(job, Output::Finished { completion_tokens: 0, reason: "length" }, shutdown); return Ok(());
        }
        self.generator.start(&ids, Sampling { temperature:r.temperature.unwrap_or(0.0), top_p:r.top_p.unwrap_or(1.0), seed:r.seed.unwrap_or(0) })?;
        let mut decoder = self.text.tokenizer.decode_stream(true);
        let mut stops = Stops::new(r.stops());
        let mut tokens = Vec::new();
        let mut emitted = String::new();
        let mut reason = "length";
        while tokens.len() < r.budget() && !cancelled(job, shutdown) {
            if let Some(token) = self.generator.advance(true)? {
                tokens.push(token as u32);
                if self.eos.contains(&token) { reason = "stop"; break; }
                if let Some(text) = decoder.step(token as u32).map_err(|e| anyhow!("decode: {e}"))? {
                    emitted += &text;
                    let text = stops.push(&text);
                    if !text.is_empty() && !send(job, Output::Text(text), shutdown) { return Ok(()); }
                    if stops.stopped { reason = "stop"; break; }
                }
            }
        }
        if cancelled(job, shutdown) { return Ok(()); }
        if !stops.stopped {
            // decode_stream holds incomplete UTF-8. At EOS/length, match complete decoding,
            // including a replacement character for a genuinely unfinished byte sequence.
            let full = self.text.tokenizer.decode(&tokens, true).map_err(|e| anyhow!("decode: {e}"))?;
            let tail = full.strip_prefix(&emitted).ok_or_else(|| anyhow!("non-monotonic text decoder"))?;
            let mut text = stops.push(tail);
            if stops.stopped { reason = "stop"; } else { text += &stops.finish(); }
            if !text.is_empty() && !send(job, Output::Text(text), shutdown) { return Ok(()); }
        }
        send(job, Output::Finished { completion_tokens:tokens.len(), reason }, shutdown);
        Ok(())
    }
}
fn cancelled(job: &Job, shutdown: &AtomicBool) -> bool { job.output.is_closed() || shutdown.load(Ordering::Acquire) }
fn send(job: &Job, mut item: Output, shutdown: &AtomicBool) -> bool {
    loop {
        if cancelled(job, shutdown) { return false; }
        match job.output.try_send(item) {
            Ok(()) => return true,
            Err(mpsc::error::TrySendError::Closed(_)) => return false,
            Err(mpsc::error::TrySendError::Full(value)) => { item = value; std::thread::sleep(OUTPUT_POLL); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Request;
    use tokio::sync::Semaphore;

    #[test]
    fn bounded_output_cancels_without_releasing_admission_early() {
        let semaphore = Arc::new(Semaphore::new(1));
        let (output, receiver) = mpsc::channel(1);
        let job = Job { request:serde_json::from_str::<Request>(r#"{"model":"test","prompt":"hi"}"#).unwrap(), output,
            _permit:semaphore.clone().try_acquire_owned().unwrap() };
        job.output.try_send(Output::Started(1)).unwrap_or_else(|_| panic!("empty channel"));
        let shutdown = Arc::new(AtomicBool::new(false));
        let flag = shutdown.clone();
        let worker = std::thread::spawn(move || {
            assert!(!send(&job, Output::Text("blocked".into()), &flag));
            assert_eq!(job._permit.num_permits(), 1);
        });
        assert_eq!(semaphore.available_permits(), 0);
        drop(receiver);
        worker.join().unwrap();
        assert_eq!(semaphore.available_permits(), 1);
    }
}
