//! Narrow OpenAI text API. A worker owns admission until its last GPU step completes.
use crate::text::Message;
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::{get, post},
};
use emmy_runtime::generation::Sampling;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};

const MAX_BODY_BYTES: usize = 1024 * 1024;
const OUTPUT_CHANNEL_CAPACITY: usize = 8;
const DEFAULT_OUTPUT_TOKENS: usize = 128;
const MAX_STOP_STRINGS: usize = 4;
const MAX_STOP_BYTES: usize = 256;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub model: String,
    pub prompt: Option<String>,
    pub messages: Option<Vec<Message>>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
    pub stop: Option<Stop>,
    pub repetition_penalty: Option<f64>,
    pub logprobs: Option<usize>,
    #[serde(default)]
    pub ignore_eos: bool,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Request {
    pub fn stops(&self) -> Vec<String> {
        match &self.stop {
            None => vec![],
            Some(Stop::One(s)) => vec![s.clone()],
            Some(Stop::Many(s)) => s.clone(),
        }
    }
    pub fn budget(&self) -> usize {
        self.max_tokens.unwrap_or(DEFAULT_OUTPUT_TOKENS)
    }
    pub fn sampling(&self) -> Sampling {
        let defaults = Sampling::default();
        Sampling {
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            seed: self.seed.unwrap_or(defaults.seed),
        }
    }
    fn validate(&self, model: &str, chat: bool) -> Result<(), ApiError> {
        if self.model != model {
            return Err(ApiError(StatusCode::NOT_FOUND, "unknown model".into()));
        }
        if (chat && (self.messages.is_none() || self.prompt.is_some()))
            || (!chat && (self.prompt.is_none() || self.messages.is_some()))
        {
            return Err(ApiError::invalid(
                "supply messages for chat or a text prompt for completions",
            ));
        }
        if self.stream_options.is_some() && !self.stream {
            return Err(ApiError::invalid("stream_options requires stream"));
        }
        let sampling = self.sampling();
        let t = sampling.temperature;
        let p = sampling.top_p;
        if !t.is_finite() || t < 0.0 || !p.is_finite() || p <= 0.0 || p > 1.0 {
            return Err(ApiError::invalid("invalid temperature or top_p"));
        }
        if self.repetition_penalty.is_some_and(|p| p != 1.0) || self.logprobs.is_some() {
            return Err(ApiError::invalid(
                "repetition penalties and logprobs are unsupported",
            ));
        }
        let stops = self.stops();
        if stops.len() > MAX_STOP_STRINGS
            || stops
                .iter()
                .any(|s| s.is_empty() || s.len() > MAX_STOP_BYTES)
        {
            return Err(ApiError::invalid(
                "stop requires up to four nonempty strings of at most 256 bytes",
            ));
        }
        Ok(())
    }
}

pub struct ApiError(pub StatusCode, pub String);
impl ApiError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self(StatusCode::BAD_REQUEST, message.into())
    }
    pub fn unavailable() -> Self {
        Self(
            StatusCode::SERVICE_UNAVAILABLE,
            "runtime unavailable; restart required".into(),
        )
    }
    pub fn value(&self) -> Value {
        json!({"error":{"message":self.1,"type": if self.0.is_server_error() {"server_error"} else {"invalid_request_error"},"param":null,"code":self.0.as_u16()}})
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(self.value())).into_response()
    }
}

pub enum Output {
    Started(usize),
    Text(String),
    Finished {
        completion_tokens: usize,
        reason: &'static str,
    },
    Error(ApiError),
}
pub struct Job {
    pub request: Request,
    pub output: mpsc::Sender<Output>,
    pub _permit: OwnedSemaphorePermit,
}
#[derive(Clone)]
pub struct App {
    pub model: String,
    pub jobs: mpsc::Sender<Job>,
    pub admission: Arc<Semaphore>,
    pub ready: Arc<AtomicBool>,
    pub shutdown: Arc<AtomicBool>,
}
impl App {
    pub fn available(&self) -> bool {
        self.ready.load(Ordering::Acquire) && !self.shutdown.load(Ordering::Acquire)
    }
}

pub fn router(app: App) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat))
        .fallback(|| async { ApiError(StatusCode::NOT_FOUND, "unknown endpoint".into()) })
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(app)
}
async fn health(State(app): State<App>) -> Response {
    if app.available() {
        StatusCode::OK.into_response()
    } else {
        ApiError::unavailable().into_response()
    }
}
async fn models(State(app): State<App>) -> Json<Value> {
    Json(
        json!({"object":"list","data":[{"id":app.model,"object":"model","created":0,"owned_by":"emmy"}]}),
    )
}
async fn completions(
    State(app): State<App>,
    request: Result<Json<Request>, JsonRejection>,
) -> Response {
    handle(app, request, false).await
}
async fn chat(State(app): State<App>, request: Result<Json<Request>, JsonRejection>) -> Response {
    handle(app, request, true).await
}

async fn handle(app: App, request: Result<Json<Request>, JsonRejection>, chat: bool) -> Response {
    let request = match request {
        Ok(Json(r)) => r,
        Err(e) => return ApiError(e.status(), e.body_text()).into_response(),
    };
    if let Err(e) = request.validate(&app.model, chat) {
        return e.into_response();
    }
    if !app.available() {
        return ApiError::unavailable().into_response();
    }
    let permit = match app.admission.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return ApiError(
                StatusCode::TOO_MANY_REQUESTS,
                "model busy; one active request is supported".into(),
            )
            .into_response();
        }
    };
    let stream = request.stream;
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|o| o.include_usage);
    let (output, mut receiver) = mpsc::channel(OUTPUT_CHANNEL_CAPACITY);
    if app
        .jobs
        .try_send(Job {
            request,
            output,
            _permit: permit,
        })
        .is_err()
    {
        return ApiError::unavailable().into_response();
    }
    let prompt_tokens = match receiver.recv().await {
        Some(Output::Started(n)) => n,
        Some(Output::Error(e)) => return e.into_response(),
        _ => return ApiError::unavailable().into_response(),
    };
    let id = format!("cmpl-{}", NEXT_ID.fetch_add(1, Ordering::Relaxed));
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let model = app.model;
    if stream {
        // Receiver ownership is the cancellation signal: dropping the body closes the channel.
        let initial = if chat {
            json!({"role":"assistant","content":""})
        } else {
            json!("")
        };
        let first = chunk(&id, &model, created, chat, initial, Value::Null);
        let events = tokio_stream::iter(if chat { vec![first] } else { vec![] }).chain(tokio_stream::wrappers::ReceiverStream::new(receiver)
            .flat_map(move |item| {
                let values = match item {
                    Output::Text(text) => vec![chunk(&id, &model, created, chat, if chat {json!({"content":text})} else {json!(text)}, Value::Null)],
                    Output::Finished { completion_tokens, reason } => {
                        let mut values = vec![chunk(&id, &model, created, chat, if chat {json!({})} else {json!("")}, json!(reason))];
                        if include_usage { values.push(json!({"id":id,"object":if chat {"chat.completion.chunk"} else {"text_completion"},"created":created,"model":model,"choices":[],"usage":usage(prompt_tokens,completion_tokens)})); }
                        values.push(Value::String("[DONE]".into())); values
                    }
                    Output::Error(e) => vec![e.value(), Value::String("[DONE]".into())],
                    Output::Started(_) => vec![],
                };
                tokio_stream::iter(values)
            })).map(|value| Ok::<_, Infallible>(Event::default().data(if value == "[DONE]" {"[DONE]".into()} else {value.to_string()})));
        Sse::new(events).into_response()
    } else {
        let mut text = String::new();
        while let Some(item) = receiver.recv().await {
            match item {
                Output::Text(s) => text += &s,
                Output::Finished {
                    completion_tokens,
                    reason,
                } => {
                    let choice = if chat {
                        json!({"index":0,"message":{"role":"assistant","content":text},"finish_reason":reason})
                    } else {
                        json!({"index":0,"text":text,"logprobs":null,"finish_reason":reason})
                    };
                    return Json(json!({"id":id,"object":if chat {"chat.completion"} else {"text_completion"},"created":created,"model":model,"choices":[choice],"usage":usage(prompt_tokens,completion_tokens)})).into_response();
                }
                Output::Error(e) => return e.into_response(),
                Output::Started(_) => (),
            }
        }
        ApiError::unavailable().into_response()
    }
}
fn usage(prompt: usize, completion: usize) -> Value {
    json!({"prompt_tokens":prompt,"completion_tokens":completion,"total_tokens":prompt+completion})
}
fn chunk(id: &str, model: &str, created: u64, chat: bool, text: Value, reason: Value) -> Value {
    let choice = if chat {
        json!({"index":0,"delta":text,"finish_reason":reason})
    } else {
        json!({"index":0,"text":text,"logprobs":null,"finish_reason":reason})
    };
    json!({"id":id,"object":if chat {"chat.completion.chunk"} else {"text_completion"},"created":created,"model":model,"choices":[choice]})
}
