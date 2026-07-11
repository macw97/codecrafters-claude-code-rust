use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json, from_value};
use serde::Deserialize;
use std::{env, process};


#[derive(Debug, Deserialize)]
pub struct ChatCompletion {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    #[serde(default)]
    pub index: u32,
    pub message: Message,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)] // only present on some providers (Nvidia/OpenRouter)
    pub reasoning: Option<String>,
    #[serde(default)] // absent when the model returns plain text
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type", default)]
    pub tool_type: String,
    pub function: Function,
}

#[derive(Debug, Deserialize)]
pub struct Function {
    pub name: String,
    pub arguments: String, // JSON-encoded string in BOTH your examples
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}
#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);

    let ai_model = env::var("LOCAL_MODEL").unwrap_or("anthropic/claude-haiku-4.5".to_string());

    #[allow(unused_variables)]
    let response: Value = client
        .chat()
        .create_byot(json!({
            "messages": [
                {
                    "role": "user",
                    "content": args.prompt,

                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "description": "Read and return the contents of a file",
                        "parameters": {
                        "type": "object",
                        "properties": {
                          "file_path": {
                           "type": "string",
                           "description": "The path to the file to read"
                          }
                        },
                       "required": ["file_path"]
                        }
                    }
                }
            ],
            "model": ai_model,
        }))
        .await?;

    
    // You can use print statements as follows for debugging, they'll be visible when running tests.
    eprintln!("Logs from your program will appear here!");

    let body: ChatCompletion = serde_json::from_value(response)?;

    if let Some(content) = body.choices.first().and_then(|c| c.message.tool_calls.first()) {
        let name = &content.function.name;
        let args: serde_json::Value = serde_json::from_str(&content.function.arguments)?;
        let file_path = args.get("file_path").and_then(|v| v.as_str());

        match name.as_str() {
            "Read" => {
                
                if let Some(path) = file_path {
                    let f = std::fs::read_to_string(path)?;
                    println!("{}", f);
                }
            }
            _ => { println!("Not implemented operation yet!"); }
        }
    } else if let Some(resp) = body.choices.first().and_then(|c| c.message.content.as_ref()) {
        println!("{}", resp);
    }


    Ok(())
}
