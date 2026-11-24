//! Checkpoint-owned text processing. No model execution or tokenizer fallback.
use anyhow::{Result, anyhow, ensure};
use minijinja::{Environment, context};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::{
    DecodeStream, Tokenizer, decoders::DecoderWrapper, models::ModelWrapper,
    normalizers::NormalizerWrapper, pre_tokenizers::PreTokenizerWrapper,
    processors::PostProcessorWrapper,
};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    pub content: Content,
}

/// A message's content: its text, or the OpenAI parts of a user message that carries images.
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<Part>),
}

/// The checkpoint's template renders an image part as its image placeholder.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Part {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageUrl {
    /// Not rendered: the template writes the placeholder alone, so the image's bytes stay out
    /// of the render.
    #[serde(skip_serializing)]
    pub url: String,
    #[serde(default, skip_serializing)]
    pub detail: Option<String>,
}

impl Message {
    /// The URLs of the images the message carries, in order.
    pub fn images(&self) -> impl Iterator<Item = &str> {
        let parts = match &self.content {
            Content::Text(_) => &[][..],
            Content::Parts(parts) => parts.as_slice(),
        };
        parts.iter().filter_map(|part| match part {
            Part::ImageUrl { image_url } => Some(image_url.url.as_str()),
            Part::Text { .. } => None,
        })
    }
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

/// Decode one response, retaining incomplete Unicode and stop strings until they can be emitted.
pub(crate) struct Decoder<'a> {
    tokenizer: &'a Tokenizer,
    stream: DecodeStream<
        'a,
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
    eos: &'a [i64],
    ended: bool,
    tokens: Vec<u32>,
    emitted: String,
    stops: Stops,
}

impl<'a> Decoder<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer, eos: &'a [i64], stops: Vec<String>) -> Self {
        Self {
            tokenizer,
            stream: tokenizer.decode_stream(true),
            eos,
            ended: false,
            tokens: Vec::new(),
            emitted: String::new(),
            stops: Stops::new(stops),
        }
    }

    pub(crate) fn push(&mut self, token: i64) -> Result<String> {
        self.tokens.push(token as u32);
        if self.eos.contains(&token) {
            self.ended = true;
            return Ok(String::new());
        }
        let text = self
            .stream
            .step(token as u32)
            .map_err(|e| anyhow!("decode: {e}"))?
            .unwrap_or_default();
        self.emitted += &text;
        Ok(self.stops.push(&text))
    }

    pub(crate) fn finish(&mut self) -> Result<String> {
        if self.stops.stopped {
            return Ok(String::new());
        }
        // Flush incomplete UTF-8 as complete decoding would, including genuine replacement characters.
        let full = self
            .tokenizer
            .decode(&self.tokens, true)
            .map_err(|e| anyhow!("decode: {e}"))?;
        let tail = full
            .strip_prefix(&self.emitted)
            .ok_or_else(|| anyhow!("non-monotonic text decoder"))?;
        let mut text = self.stops.push(tail);
        if !self.stops.stopped {
            text += &self.stops.finish();
        }
        Ok(text)
    }

    pub(crate) fn stopped(&self) -> bool {
        self.ended || self.stops.stopped
    }

    pub(crate) fn token_count(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn reason(&self) -> &'static str {
        if self.stopped() { "stop" } else { "length" }
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
    use tokenizers::{AddedToken, decoders::byte_fallback::ByteFallback, models::bpe::BPE};

    fn tokenizer() -> Tokenizer {
        let vocab: [(String, u32); 256] =
            std::array::from_fn(|byte| (format!("<0x{byte:02X}>"), byte as u32));
        let model = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .byte_fallback(true)
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_decoder(Some(ByteFallback::new()));
        tokenizer.add_special_tokens(&[AddedToken::from("<eos>", true)]);
        tokenizer
    }

    #[test]
    fn decoder_preserves_split_unicode_and_flushes_incomplete_bytes() {
        let tokenizer = tokenizer();
        for text in ["é", "🦀", "�"] {
            let bytes = text.as_bytes();
            for count in 1..=bytes.len() {
                let mut decoder = Decoder::new(&tokenizer, &[], vec![]);
                let mut output = String::new();
                for &byte in &bytes[..count] {
                    output += &decoder.push(byte.into()).unwrap();
                }
                if count < bytes.len() {
                    assert!(output.is_empty());
                }
                output += &decoder.finish().unwrap();
                let ids: Vec<u32> = bytes[..count].iter().map(|&byte| byte.into()).collect();
                assert_eq!(output, tokenizer.decode(&ids, true).unwrap());
                if count == bytes.len() {
                    assert_eq!(output, text);
                }
                assert_eq!(decoder.token_count(), count);
                assert_eq!(decoder.reason(), "length");
            }
        }
    }

    #[test]
    fn decoder_stops_across_tokens_and_counts_consumed_tokens() {
        let tokenizer = tokenizer();
        let mut decoder = Decoder::new(&tokenizer, &[], vec!["END".into()]);
        let mut output = String::new();
        for byte in b"hiENDtrailing" {
            output += &decoder.push((*byte).into()).unwrap();
            if decoder.stopped() {
                break;
            }
        }
        assert_eq!(output, "hi");
        assert_eq!(decoder.finish().unwrap(), "");
        assert_eq!(decoder.token_count(), 5);
        assert_eq!(decoder.reason(), "stop");
    }

    #[test]
    fn decoder_flushes_partial_stop_at_length_or_eos() {
        let tokenizer = tokenizer();
        let eos = i64::from(tokenizer.token_to_id("<eos>").unwrap());
        for ended in [false, true] {
            let mut decoder =
                Decoder::new(&tokenizer, std::slice::from_ref(&eos), vec!["END".into()]);
            let mut output = String::new();
            for byte in b"hiEN" {
                output += &decoder.push((*byte).into()).unwrap();
            }
            assert_eq!(output, "hi");
            if ended {
                assert_eq!(decoder.push(eos).unwrap(), "");
            }
            assert_eq!(decoder.finish().unwrap(), "EN");
            assert_eq!(decoder.token_count(), if ended { 5 } else { 4 });
            assert_eq!(decoder.reason(), if ended { "stop" } else { "length" });
        }
    }

    #[test]
    fn decoder_can_ignore_eos_and_finish_without_tokens() {
        let tokenizer = tokenizer();
        let mut decoder = Decoder::new(&tokenizer, &[], vec![]);
        assert_eq!(decoder.finish().unwrap(), "");
        assert_eq!(decoder.token_count(), 0);
        assert_eq!(decoder.reason(), "length");
        let mut decoder = Decoder::new(&tokenizer, &[], vec![]);
        assert_eq!(
            decoder
                .push(tokenizer.token_to_id("<eos>").unwrap().into())
                .unwrap(),
            ""
        );
        assert!(!decoder.stopped());
        assert_eq!(decoder.push(b'A'.into()).unwrap(), "A");
        assert_eq!(decoder.finish().unwrap(), "");
        assert_eq!(decoder.token_count(), 2);
    }

    #[test]
    fn decoder_checks_stop_strings_when_flushing_unicode() {
        let tokenizer = tokenizer();
        let mut decoder = Decoder::new(&tokenizer, &[], vec!["�".into()]);
        assert_eq!(decoder.push(0xc3).unwrap(), "");
        assert!(!decoder.stopped());
        assert_eq!(decoder.finish().unwrap(), "");
        assert_eq!(decoder.reason(), "stop");
        assert_eq!(decoder.token_count(), 1);
    }

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
