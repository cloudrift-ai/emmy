use emmy_server::text::Text;
use serde_json::Value;
use std::path::Path;

#[test]
fn checkpoint_text_parity() {
    let Ok(path) = std::env::var("NATIVE_TEXT_FIXTURE") else { return; };
    let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let text = Text::load(Path::new(fixture["root"].as_str().unwrap())).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let chat = case.get("messages").is_some();
        let prompt = if chat {
            text.render(&serde_json::from_value::<Vec<_>>(case["messages"].clone()).unwrap()).unwrap()
        } else { case["prompt"].as_str().unwrap().into() };
        assert_eq!(prompt, case["prompt"].as_str().unwrap());
        let ids = text.encode(&prompt, chat).unwrap();
        assert_eq!(serde_json::to_value(&ids).unwrap(), case["ids"]);
        let mut decoder = text.tokenizer.decode_stream(true);
        let mut decoded = String::new();
        for id in &ids {
            if let Some(chunk) = decoder.step(*id as u32).unwrap() { decoded += &chunk; }
        }
        let full = text.tokenizer.decode(&ids.iter().map(|&id| id as u32).collect::<Vec<_>>(), true).unwrap();
        assert!(full.starts_with(&decoded));
        decoded += &full[decoded.len()..];
        assert_eq!(decoded, case["decoded"].as_str().unwrap());
    }
}
