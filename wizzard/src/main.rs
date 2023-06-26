use std::error::Error;

use async_openai::{
    config::AzureConfig,
    types::{
        ChatCompletionRequestMessageArgs, CreateChatCompletionRequestArgs,
        CreateEmbeddingRequestArgs, Role,
    },
    Client,
};

use polars::*;
use polars::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs::File;
use std::io::Write;

async fn chat_completion_example(client: &Client<AzureConfig>) -> Result<(), Box<dyn Error>> {
    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(512u16)
        .model("gpt-35-turbo")
        .messages([
            ChatCompletionRequestMessageArgs::default()
                .role(Role::System)
                .content("You are a helpful assistant.")
                .build()?,
            ChatCompletionRequestMessageArgs::default()
                .role(Role::User)
                .content("How does large language model work?")
                .build()?,
        ])
        .build()?;

    let response = client.chat().create(request).await?;

    println!("\nResponse:\n");
    for choice in response.choices {
        println!(
            "{}: Role: {}  Content: {:?}",
            choice.index, choice.message.role, choice.message.content
        );
    }
    Ok(())
}

// chat completion given prompt
async fn chat_completion_given_prompt_example(client: &Client<AzureConfig>,prompt: &String) -> Result<String, Box<dyn Error>> {
    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(512u16)
        .model("gpt-35-turbo")
        .messages([
            ChatCompletionRequestMessageArgs::default()
                .role(Role::System)
                .content("You are a helpful assistant.")
                .build()?,
            ChatCompletionRequestMessageArgs::default()
                .role(Role::User)
                .content("How does large language model work?")
                .build()?,
        ])
        .build()?;

    let response = client.chat().create(request).await.unwrap();

    println!("\nResponse:\n");
    for choice in response.choices {
        println!(
            "{}: Role: {}  Content: {:?}",
            choice.index, choice.message.role, choice.message.content
        );
    }
    // return ok and the last message content
    Ok(&response.choices.last().unwrap().message.content.clone().unwrap())
   
}


fn add_constraints(prompt: &str) -> String {
    format!(
        "I want you act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems
(e.g., ChatGPT and GPT4) a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please
do not omit the input in #Given Prompt#.
You SHOULD complicate the given prompt using the following method:
Please add one more constraints/requirements into #Given Prompt#
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only
add 10 to 20 words into #Given Prompt#.
‘#Given Prompt#’, ‘#Rewritten Prompt#’, ‘given prompt’ and ‘rewritten prompt’ are not allowed to appear in
#Rewritten Prompt#
#Given Prompt#:
{}
#Rewritten Prompt#:
",
        prompt
    )
}



async fn evolve(client: &Client<AzureConfig>, dataset_path: &str, evolved_dataset_path: &str) {
    let mut it = 0;
    let dataset = CsvReader::from_path(dataset_path)
        .unwrap()
        .finish()
        .unwrap();
    let evolution_list = vec![add_constraints];
    let mut rng = thread_rng();
    

    let mut evolved_dataset = dataset.clone();

    // get collumn text
    let text_column = dataset.column("text").unwrap();

    // new vector for evolved text
    let mut evolved_text_vec = Vec::new();


    for text in text_column.iter() {
        it += 1;
        if it > 5 {
            break;
        }
        //let evolution_prompt =add_constraints(sampl  //evolutions[index](sample.get("text").unwrap().as_str().unwrap());
        //let mut conversation = Conversation::new();
        //let evolved_text = "booo";//conversation.user_message(evolution_prompt, false);
        let extracted_text = text.get_str().unwrap();
        println!("before: {}", extracted_text);
        
        let evolved_test = chat_completion_given_prompt_example(client, extracted_text).await;
        match evolved_test {
            Some(text) => {
                
                println!("after: {}", text);
                evolved_text_vec.push(text);
            }
            None => {
                println!("after: {}", "None");
            }
        };

        
        
    }

    // store vector in txt file
    let mut file = File::create("evolved_text.txt").unwrap();
    for text in evolved_text_vec {
        file.write_all(text.as_bytes()).unwrap();
    }


    // // add evolved text to dataset
    // let evolved_text_series = Series::new("evolved_text", evolved_text_vec);
    // evolved_dataset.hstack(&[evolved_text_series]).unwrap();

    // // write evolved dataset to csv
    // let mut file = File::create(evolved_dataset_path).unwrap();
    // CsvWriter::new(&mut file)
    //     .has_header(true)
    //     .finish(&mut evolved_dataset)
    //     .unwrap();
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config = AzureConfig::new()
        .with_api_base("https://nlpschool.openai.azure.com")
        .with_api_key("")
        .with_deployment_id("cim")
        .with_api_version("2023-03-15-preview");

    let client = Client::with_config(config);

    evolve(&client,"/mnt/g/Mega/Documents/ML/NaturalLanguageProcessing/WizzardLM_UW/oasst1-train-tree.csv", "evolved_prompts.csv");

    // Run embedding Example
    //embedding_example(&client).await?;

    // Run completions stream Example
    // Bug (help wanted): https://github.com/64bit/async-openai/pull/67#issuecomment-1555165805
    //completions_stream_example(&client).await?;

    // Run chat completion example
    //chat_completion_example(&client).await?;

    Ok(())
}