use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use emmy_server::api::{App, Output, router};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{Semaphore, mpsc};
use tower::ServiceExt;

fn app() -> (App, mpsc::Receiver<emmy_server::api::Job>) {
    let (jobs, receiver) = mpsc::channel(1);
    (
        App {
            model: "test".into(),
            jobs,
            admission: Arc::new(Semaphore::new(1)),
            ready: Arc::new(AtomicBool::new(true)),
            shutdown: Arc::new(AtomicBool::new(false)),
        },
        receiver,
    )
}
fn request(value: Value, chat: bool) -> Request<Body> {
    Request::post(if chat {
        "/v1/chat/completions"
    } else {
        "/v1/completions"
    })
    .header("content-type", "application/json")
    .body(Body::from(value.to_string()))
    .unwrap()
}

#[tokio::test]
async fn stream_and_nonstream_usage_finish_and_unicode() {
    for chat in [false, true] {
        for (stream, include_usage) in [
            (false, None),
            (true, None),
            (true, Some(false)),
            (true, Some(true)),
        ] {
            let (app, mut receiver) = app();
            let mut body = json!({"model":"test","stream":stream});
            if let Some(include_usage) = include_usage {
                body["stream_options"] = json!({"include_usage":include_usage});
            }
            if chat {
                body["messages"] = json!([{"role":"user","content":"hi"}]);
            } else {
                body["prompt"] = json!("hi");
            }
            let worker = tokio::spawn(async move {
                let job = receiver.recv().await.unwrap();
                job.output.send(Output::Started(3)).await.unwrap();
                job.output
                    .send(Output::Text("你好🦀".into()))
                    .await
                    .unwrap();
                job.output
                    .send(Output::Finished {
                        completion_tokens: 2,
                        reason: "stop",
                    })
                    .await
                    .unwrap();
            });
            let response = router(app).oneshot(request(body, chat)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let text = std::str::from_utf8(&bytes).unwrap();
            assert!(text.contains("你好🦀"));
            if stream {
                assert!(text.ends_with("data: [DONE]\n\n"));
                let events: Vec<Value> = text
                    .lines()
                    .filter_map(|s| s.strip_prefix("data: "))
                    .filter(|s| *s != "[DONE]")
                    .map(|s| serde_json::from_str(s).unwrap())
                    .collect();
                let usage_events: Vec<_> = events
                    .iter()
                    .filter(|event| event.get("usage").is_some())
                    .collect();
                assert_eq!(usage_events.len(), usize::from(include_usage == Some(true)));
                if include_usage == Some(true) {
                    assert_eq!(events.last().unwrap()["usage"]["total_tokens"], 5);
                    assert_eq!(events.last().unwrap()["choices"], json!([]));
                }
                assert_eq!(
                    events[events.len() - 1 - usage_events.len()]["choices"][0]["finish_reason"],
                    "stop"
                );
                assert_eq!(events.len(), 2 + usize::from(chat) + usage_events.len());
                if chat {
                    assert_eq!(
                        events[0]["choices"][0]["delta"],
                        json!({"role":"assistant","content":""})
                    );
                    assert_eq!(events[1]["choices"][0]["delta"]["content"], "你好🦀");
                } else {
                    assert_eq!(events[0]["choices"][0]["text"], "你好🦀");
                }
                for event in &events {
                    assert_eq!(event["id"], events[0]["id"]);
                    assert_eq!(event["created"], events[0]["created"]);
                    assert_eq!(event["model"], "test");
                    assert_eq!(
                        event["object"],
                        if chat {
                            "chat.completion.chunk"
                        } else {
                            "text_completion"
                        }
                    );
                }
            } else {
                let value: Value = serde_json::from_str(text).unwrap();
                assert_eq!(value["usage"]["total_tokens"], 5);
                assert_eq!(value["choices"][0]["finish_reason"], "stop");
                assert_eq!(value["model"], "test");
                assert_eq!(
                    value["object"],
                    if chat {
                        "chat.completion"
                    } else {
                        "text_completion"
                    }
                );
                if chat {
                    assert_eq!(
                        value["choices"][0]["message"],
                        json!({"role":"assistant","content":"你好🦀"})
                    );
                } else {
                    assert_eq!(value["choices"][0]["text"], "你好🦀");
                }
            }
            worker.await.unwrap();
        }
    }
}

#[tokio::test]
async fn disconnect_keeps_admission_until_worker_releases_it() {
    for (stream, started) in [(true, false), (true, true), (false, true)] {
        let (app, mut receiver) = app();
        let route = router(app.clone());
        let response = tokio::spawn(route.clone().oneshot(request(
            json!({"model":"test","prompt":"hi","stream":stream}),
            false,
        )));
        let job = receiver.recv().await.unwrap();
        if started {
            job.output.send(Output::Started(1)).await.unwrap();
        }
        if stream && started {
            drop(response.await.unwrap().unwrap());
        } else {
            response.abort();
            assert!(response.await.unwrap_err().is_cancelled());
        }
        assert!(job.output.is_closed());
        assert_eq!(
            route
                .clone()
                .oneshot(request(json!({"model":"test","prompt":"hi"}), false))
                .await
                .unwrap()
                .status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        drop(job);
        assert_eq!(app.admission.available_permits(), 1);
    }
}

#[tokio::test]
async fn validation_and_readiness() {
    let (app, _receiver) = app();
    let route = router(app.clone());
    for body in [
        json!({"model":"test","prompt":"hi","top_p":0}),
        json!({"model":"test","prompt":"hi","tools":[]}),
        json!({"model":"test","prompt":"hi","stop":""}),
        json!({"model":"test","prompt":[1,2]}),
    ] {
        assert!(
            route
                .clone()
                .oneshot(request(body, false))
                .await
                .unwrap()
                .status()
                .is_client_error()
        );
    }
    app.ready.store(false, Ordering::Release);
    assert_eq!(
        route
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        route
            .oneshot(request(json!({"model":"test","prompt":"hi"}), false))
            .await
            .unwrap()
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn failure_is_visible_and_never_retried() {
    for stream in [false, true] {
        let (app, mut receiver) = app();
        let state = app.clone();
        let route = router(app);
        let pending = tokio::spawn(route.clone().oneshot(request(
            json!({"model":"test","prompt":"hi","stream":stream}),
            false,
        )));
        let job = receiver.recv().await.unwrap();
        job.output.send(Output::Started(1)).await.unwrap();
        state.ready.store(false, Ordering::Release);
        job.output
            .send(Output::Error(emmy_server::api::ApiError::unavailable()))
            .await
            .unwrap();
        drop(job);
        let response = pending.await.unwrap().unwrap();
        assert_eq!(
            response.status(),
            if stream {
                StatusCode::OK
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            }
        );
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert!(
            std::str::from_utf8(&body)
                .unwrap()
                .contains("restart required")
        );
        if stream {
            assert!(
                std::str::from_utf8(&body)
                    .unwrap()
                    .ends_with("data: [DONE]\n\n")
            );
        } else {
            let value: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(value["error"]["code"], 503);
            assert!(value.get("choices").is_none());
        }
        assert!(receiver.try_recv().is_err());
        assert_eq!(
            route
                .oneshot(request(json!({"model":"test","prompt":"hi"}), false))
                .await
                .unwrap()
                .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}

#[tokio::test]
async fn worker_rejection_before_start_keeps_http_status() {
    for stream in [false, true] {
        let (app, mut receiver) = app();
        let pending = tokio::spawn(router(app).oneshot(request(
            json!({"model":"test","prompt":"hi","stream":stream}),
            false,
        )));
        let job = receiver.recv().await.unwrap();
        job.output
            .send(Output::Error(emmy_server::api::ApiError::invalid(
                "prompt exceeds context",
            )))
            .await
            .unwrap();
        drop(job);
        let response = pending.await.unwrap().unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["error"]["message"], "prompt exceeds context");
    }
}
