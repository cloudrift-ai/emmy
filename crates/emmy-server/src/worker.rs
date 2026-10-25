//! One thread owns text processing and CUDA state. Channel closure cancels future steps.
use crate::{
    api::{ApiError, Job, Output, Request},
    text::{Stops, Text},
    vision::Processor,
};
use anyhow::{Result, anyhow, ensure};
use emmy_runtime::{cuda::Device, generation::Generator, vision::Image};
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::mpsc;

const OUTPUT_POLL: Duration = Duration::from_millis(5);

pub struct Worker {
    generator: Generator,
    text: Text,
    /// A pack that answers images: the processor that prepares them for its tower.
    images: Option<Processor>,
    context: usize,
    eos: Vec<i64>,
}
impl Worker {
    /// Load the pack at `root`, serving requests of at most `context` tokens: the pack's own when
    /// `None`.
    pub fn load(root: &Path, context: Option<usize>) -> Result<Self> {
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let config = &manifest["key"]["generation"];
        let capacity = config["context_length"].as_u64().unwrap_or(0) as usize;
        let context = context.unwrap_or(capacity);
        anyhow::ensure!(
            context > 0 && context <= capacity,
            "invalid context capacity"
        );
        let eos = serde_json::from_value(config["eos_ids"].clone())?;
        let text = Text::load(root)?;
        let images = Processor::load(root, serde_json::from_value(config["vision"].clone())?)?;
        let generator = Generator::load(&Device::new(0)?, root)?;
        Ok(Self {
            generator,
            text,
            images,
            context,
            eos,
        })
    }

    /// A request's prompt tokens, and its image when a message carries one: the tower's inputs
    /// and the prompt's M-RoPE positions, its placeholder grown to the image's tokens.
    fn prompt(&self, r: &Request) -> Result<(Vec<i64>, Option<Image>)> {
        let Some(messages) = &r.messages else {
            return Ok((
                self.text
                    .encode(r.prompt.as_deref().unwrap_or_default(), false)?,
                None,
            ));
        };
        let urls: Vec<&str> = messages.iter().flat_map(|m| m.images()).collect();
        let ids = self.text.encode(&self.text.render(messages)?, true)?;
        let Some(&url) = urls.first() else {
            return Ok((ids, None));
        };
        let processor = self
            .images
            .as_ref()
            .ok_or_else(|| anyhow!("this model serves no images"))?;
        ensure!(urls.len() == 1, "a request carries one image at most");
        let (ids, image) = processor.expand(&ids, processor.prepare(url)?)?;
        Ok((ids, Some(image)))
    }
    pub fn run(
        mut self,
        mut jobs: mpsc::Receiver<Job>,
        ready: Arc<AtomicBool>,
        shutdown: Arc<AtomicBool>,
    ) {
        ready.store(true, Ordering::Release);
        while let Some(job) = jobs.blocking_recv() {
            if shutdown.load(Ordering::Acquire) {
                break;
            }
            if let Err(error) = self.request(&job, &shutdown) {
                // No retry and no new work after any execution/text-decoding failure.
                eprintln!("native request failed: {error:#}");
                ready.store(false, Ordering::Release);
                drop(self); // Synchronize/release device state before the job releases admission.
                send(&job, Output::Error(ApiError::unavailable()), &shutdown);
                return;
            }
            // advance synchronizes before returning; only now may admission be released.
        }
        ready.store(false, Ordering::Release);
    }
    fn request(&mut self, job: &Job, shutdown: &AtomicBool) -> Result<()> {
        let r = &job.request;
        let started = Instant::now();
        let (ids, image) = match self.prompt(r) {
            Ok((ids, image))
                if !ids.is_empty()
                    && ids.len() <= self.context
                    && r.budget() <= self.context - ids.len() =>
            {
                (ids, image)
            }
            Ok(_) => {
                send(
                    job,
                    Output::Error(ApiError::invalid(
                        "prompt and output budget must fit the context",
                    )),
                    shutdown,
                );
                return Ok(());
            }
            Err(e) => {
                send(
                    job,
                    Output::Error(ApiError::invalid(e.to_string())),
                    shutdown,
                );
                return Ok(());
            }
        };
        if !send(job, Output::Started(ids.len()), shutdown) {
            return Ok(());
        }
        if r.budget() == 0 {
            send(
                job,
                Output::Finished {
                    completion_tokens: 0,
                    reason: "length",
                },
                shutdown,
            );
            return Ok(());
        }
        match &image {
            Some(image) => {
                let prepared = started.elapsed();
                self.generator
                    .start_image(&ids, image, true, r.sampling())?;
                // What an image costs before prefill: the adapter's processing and the tower's run.
                eprintln!(
                    "image: {} tokens, prepared in {:.1} ms, tower {:.1} ms",
                    image.tokens,
                    prepared.as_secs_f64() * 1e3,
                    (started.elapsed() - prepared).as_secs_f64() * 1e3
                );
            }
            None => self.generator.start(&ids, r.sampling())?,
        }
        let mut decoder = self.text.tokenizer.decode_stream(true);
        let mut stops = Stops::new(r.stops());
        let mut tokens = Vec::new();
        let mut emitted = String::new();
        let mut reason = "length";
        while tokens.len() < r.budget() && !cancelled(job, shutdown) {
            if let Some(token) = self.generator.advance(true, r.ignore_eos)? {
                tokens.push(token as u32);
                if !r.ignore_eos && self.eos.contains(&token) {
                    reason = "stop";
                    break;
                }
                if let Some(text) = decoder
                    .step(token as u32)
                    .map_err(|e| anyhow!("decode: {e}"))?
                {
                    emitted += &text;
                    let text = stops.push(&text);
                    if !text.is_empty() && !send(job, Output::Text(text), shutdown) {
                        return Ok(());
                    }
                    if stops.stopped {
                        reason = "stop";
                        break;
                    }
                }
            }
        }
        if cancelled(job, shutdown) {
            return Ok(());
        }
        if !stops.stopped {
            // decode_stream holds incomplete UTF-8. At EOS/length, match complete decoding,
            // including a replacement character for a genuinely unfinished byte sequence.
            let full = self
                .text
                .tokenizer
                .decode(&tokens, true)
                .map_err(|e| anyhow!("decode: {e}"))?;
            let tail = full
                .strip_prefix(&emitted)
                .ok_or_else(|| anyhow!("non-monotonic text decoder"))?;
            let mut text = stops.push(tail);
            if stops.stopped {
                reason = "stop";
            } else {
                text += &stops.finish();
            }
            if !text.is_empty() && !send(job, Output::Text(text), shutdown) {
                return Ok(());
            }
        }
        send(
            job,
            Output::Finished {
                completion_tokens: tokens.len(),
                reason,
            },
            shutdown,
        );
        Ok(())
    }
}
fn cancelled(job: &Job, shutdown: &AtomicBool) -> bool {
    job.output.is_closed() || shutdown.load(Ordering::Acquire)
}
fn send(job: &Job, mut item: Output, shutdown: &AtomicBool) -> bool {
    loop {
        if cancelled(job, shutdown) {
            return false;
        }
        match job.output.try_send(item) {
            Ok(()) => return true,
            Err(mpsc::error::TrySendError::Closed(_)) => return false,
            Err(mpsc::error::TrySendError::Full(value)) => {
                item = value;
                std::thread::sleep(OUTPUT_POLL);
            }
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
        let job = Job {
            request: serde_json::from_str::<Request>(r#"{"model":"test","prompt":"hi"}"#).unwrap(),
            output,
            _permit: semaphore.clone().try_acquire_owned().unwrap(),
        };
        job.output
            .try_send(Output::Started(1))
            .unwrap_or_else(|_| panic!("empty channel"));
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
