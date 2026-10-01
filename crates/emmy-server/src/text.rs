//! Checkpoint-owned text processing. No model execution or tokenizer fallback.
use anyhow::{Result, anyhow, ensure};
use minijinja::{Environment, context};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub struct Text {
    pub tokenizer: Tokenizer,
    templates: Environment<'static>,
}

impl Text {
    pub fn load(root: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(root.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        let template = std::fs::read_to_string(root.join("chat_template.jinja"))?;
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        env.add_template_owned("chat", template)?;
        Ok(Self {
            tokenizer,
            templates: env,
        })
    }

    pub fn render(&self, messages: &[Message]) -> Result<String> {
        ensure!(!messages.is_empty(), "messages must not be empty");
        ensure!(
            messages
                .iter()
                .all(|m| matches!(m.role.as_str(), "system" | "user" | "assistant")),
            "only system, user, and assistant text messages are supported"
        );
        Ok(self.templates.get_template("chat")?.render(context! {
            messages => messages, add_generation_prompt => true, enable_thinking => false
        })?)
    }

    pub fn encode(&self, prompt: &str, chat: bool) -> Result<Vec<i64>> {
        Ok(self
            .tokenizer
            .encode(prompt, !chat)
            .map_err(|e| anyhow!("tokenizer: {e}"))?
            .get_ids()
            .iter()
            .map(|&id| i64::from(id))
            .collect())
    }
}

/// Hold only suffixes that can still become a stop string. Never split UTF-8.
#[derive(Default)]
pub struct Stops {
    pending: String,
    stops: Vec<String>,
    pub stopped: bool,
}

impl Stops {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            ..Self::default()
        }
    }

    pub fn push(&mut self, chunk: &str) -> String {
        self.pending.push_str(chunk);
        if let Some(end) = self.stops.iter().filter_map(|s| self.pending.find(s)).min() {
            self.stopped = true;
            let result = self.pending[..end].to_owned();
            self.pending.clear();
            return result;
        }
        let hold = self
            .stops
            .iter()
            .flat_map(|stop| {
                stop.char_indices()
                    .map(|(n, _)| n)
                    .filter(|&n| n > 0 && self.pending.ends_with(&stop[..n]))
            })
            .max()
            .unwrap_or(0);
        let rest = self.pending.split_off(self.pending.len() - hold);
        std::mem::replace(&mut self.pending, rest)
    }

    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_across_every_unicode_boundary() {
        let text = "Hello 🦀終STOP trailing";
        for (split, _) in text.char_indices() {
            let mut stops = Stops::new(vec!["終STOP".into(), "STOP".into()]);
            let mut got = stops.push(&text[..split]);
            if !stops.stopped {
                got += &stops.push(&text[split..]);
            }
            assert_eq!(got, "Hello 🦀");
            assert!(stops.stopped);
            assert_eq!(stops.finish(), "");
        }
    }

    #[test]
    fn incomplete_stop_is_flushed_at_length_limit() {
        let mut stops = Stops::new(vec!["🦀end".into()]);
        assert_eq!(stops.push("hello🦀e"), "hello");
        assert_eq!(stops.finish(), "🦀e");
    }
}
