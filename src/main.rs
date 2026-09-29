use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json, from_value};
use serde::{Deserialize, Serialize};
use std::{env, process};
use std::process::Command;
use walkdir::WalkDir;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;


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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    #[serde(default)]
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] // only present on some providers (Nvidia/OpenRouter)
    pub reasoning: Option<String>,
    #[serde(default)] // absent when the model returns plain text
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type", default)]
    pub tool_type: String,
    pub function: Function,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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

#[derive(Debug, Deserialize)]
pub struct SkillMeta {
    pub name: String,
    pub description: String,
}

#[derive(Debug)]
pub struct Skill {
    pub meta: SkillMeta,
    pub body: String,
    pub path: PathBuf,
}

#[derive(Debug, Error)]
pub enum SkillError {
    #[error("failed to read {path}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid frontmatter in {path}: {reason}")]
    Frontmatter { path: PathBuf, reason: &'static str },
    #[error("invalid YAML in {path}")]
    Yaml {
        path: PathBuf,
        #[source]
        source: serde_norway::Error,
    },
    #[error("failed to walk skills directory")]
    Walk(#[from] walkdir::Error),
}

#[derive(Debug, Default)]
pub struct LoadReport {
    pub skills: Vec<Skill>,
    pub errors: Vec<SkillError>,
}

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

fn split_frontmatter(src: &str) -> Result<(&str, &str), &'static str> {
    let src = src.strip_prefix('\u{feff}').unwrap_or(src);
    let rest = src.strip_prefix("---").ok_or("missing opening `---`")?;
    let end = rest.find("\n---").ok_or("missing closing `---`")?;
    let yaml = &rest[..end];
    let after_fence = &rest[end + "\n---".len()..];
    let body = after_fence.split_once('\n').map_or("", |(_, body)| body);
    Ok((yaml, body))
}

pub fn parse_skill(src: &str, path: &Path) -> Result<Skill, SkillError> {
    let (yaml, body) = split_frontmatter(src).map_err(|reason| SkillError::Frontmatter {
        path: path.to_path_buf(),
        reason,
    })?;
    let meta: SkillMeta = serde_norway::from_str(yaml).map_err(|source| SkillError::Yaml {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(Skill {
        meta,
        body: body.trim().to_owned(),
        path: path.to_path_buf(),
    })
}
 
/// Reads and parses a single `SKILL.md` from disk.
pub fn read_skill(path: &Path) -> Result<Skill, SkillError> {
    let src = fs::read_to_string(path).map_err(|source| SkillError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_skill(&src, path)
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

    let mut messages: Vec<Message> = vec![
        Message { role: "user".to_string(), tool_call_id: None, content: Some(args.prompt), reasoning: None, tool_calls: Vec::new()}
    ];

    // You can use print statements as follows for debugging, they'll be visible when running tests.
    eprintln!("Logs from your program will appear here!");
    let mut skills: Vec<Skill> = Vec::new();

    for entry in WalkDir::new(".claude/skills").min_depth(2).max_depth(2) {
    
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                continue;
            }
        };


        let is_skill_file = entry.file_type().is_file() && entry.file_name().eq_ignore_ascii_case("SKILL.md");
         
        if !is_skill_file {
            continue;
        }

        match read_skill(entry.path()) {
            Ok(skill) => skills.push(skill),
            Err(err) => println!("{}", err),
        }
    }

    let mut system = String::from("You have access to the following skills:\n");
    for skill in &skills {
        system.push_str(&format!("\n- {}: {}", skill.meta.name, skill.meta.description));
    } 

    messages.push(Message { role: "system".to_string(), tool_call_id: None, content: Some(system), reasoning: None, tool_calls: Vec::new()});

    'outer: loop {
        
        #[allow(unused_variables)]
        let response: Value = client
            .chat()
            .create_byot(json!({
            "messages": messages,
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
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "description": "Write content to a file",
                        "parameters": {
                        "type": "object",
                        "required": ["file_path", "content"],
                        "properties": {
                            "file_path": {
                            "type": "string",
                            "description": "The path of the file to write to"
                            },
                            "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                            }
                        }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "description": "Execute a shell command",
                        "parameters": {
                        "type": "object",
                        "required": ["command"],
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute"
                            }
                        }
                        }
                    }
                }
            ],
            "model": ai_model,
        }))
        .await?;

        let body: ChatCompletion = serde_json::from_value(response)?;
        
        let choice = match body.choices.into_iter().next() {
            Some(c) => c,
            None => break 'outer,
        };
        
        let message = choice.message;

        if message.tool_calls.is_empty() {
            let m = message;
            println!("{}", m.content.as_deref().unwrap_or("Error my dude"));
            break 'outer;
        }

        let tool_calls = message.tool_calls.clone();
        
        messages.push(message);
            

        for tool_call in tool_calls {
            let id = tool_call.id;
            let name = tool_call.function.name;
            let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)?;
            let file_path = args.get("file_path").and_then(|v| v.as_str());

            match name.as_str() {
                "Read" => {
                    
                    if let Some(path) = file_path {
                        let f = std::fs::read_to_string(path)?;
                        messages.push(Message { role: "tool".to_string(), tool_call_id: Some(id), content: Some(f), reasoning: None, tool_calls: Vec::new()})
                    }
                }
                "Write" => {
                    if let Some(path) = file_path {
                        let text = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        let _ = std::fs::write(path, text);
                        messages.push(Message { role: "tool".to_string(), tool_call_id: Some(id), content: Some(text.to_string()), reasoning: None, tool_calls: Vec::new()})
                    }
                }
                "Bash" => {
                    let command = args.get("command").and_then(|v| v.as_str());
                    if let Some(cmd) = command {
                        let out = Command::new("sh")
                            .arg("-c")
                            .arg(&cmd)
                            .output()
                            .expect("Command failed to execute");

                        let mut result = String::from_utf8_lossy(&out.stdout).into_owned();
                        if !out.stderr.is_empty() {
                            result.push_str(&String::from_utf8_lossy(&out.stderr));
                        }
                        messages.push(Message { role: "tool".to_string(), tool_call_id: Some(id), content: Some(result), reasoning: None, tool_calls: Vec::new()})
                    }
                }
                _ => { println!("Not implemented operation yet!"); }
            }
        }
            
    
    }
    


    Ok(())
}
