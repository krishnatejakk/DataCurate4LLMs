import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def generate_instruction(icl=False, response_type="clear, accurate, and concise", 
                         considerations="relevant facts and context", num_examples=None):
    base_instruction = f"Provide a {response_type} response to the following user query. Consider {considerations} in your answer."
    
    if icl:
        icl_instruction = f"""You will be presented with {num_examples} example{'s' if num_examples != 1 else ''} of user queries and the corresponding assistant responses, followed by a new user query. These examples serve as a guide for the structure, style, and depth of your response. Pay close attention to:

1. The format and organization of the responses
2. The level of detail provided
3. Any specific patterns or techniques used in addressing the queries

After the examples, you will receive a new user query. Apply the insights gained from the examples to formulate your response, while also adhering to the following guideline:

{base_instruction}

Remember, while the examples are meant to guide you, each query is unique. Tailor your response to the specific needs of the new query while maintaining the general approach demonstrated in the examples.

Examples:
"""
        return "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible" + icl_instruction
    else:
        return"You are a helpful, respectful and honest assistant. Always answer as helpfully as possible" + base_instruction

class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, system_message, examples=None, max_length=2048):
        self.prompts = prompts
        self.tokenizer = tokenizer
        # Check if tokenizer has a padding token otherwise use eos token
        if tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.system_message = system_message
        self.examples = examples

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        messages = [{"role": "system", "content": self.system_message}]
        
        if self.examples:
            for example in self.examples:
                messages.append({"role": "user", "content": example['query']})
                messages.append({"role": "assistant", "content": example['response']})
        
        messages.append({"role": "user", "content": self.prompts[idx]})
        
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

def perform_inference(model, tokenizer, prompts, batch_size=3, icl=False, 
                      response_type="clear, accurate, and concise", 
                      considerations="relevant facts and context",
                      examples=None, **gen_kwargs):
    """
    Performs inference on prompts and returns the generated text.

    Args:
        model: AutoModelForCausalLM instance
        tokenizer: AutoTokenizer instance
        prompts: list of prompts
        batch_size: batch size for inference
        icl: boolean indicating whether to include in-context learning instruction
        response_type: string describing the desired response type
        considerations: string describing what to consider in the response
        examples: list of dictionaries containing 'query' and 'response' for in-context learning
        **gen_kwargs: keyword arguments for model.generate()

    Return:
        generated_text: list of generated responses from the given LM
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert tokenizer.padding_side == "left", "Tokenizer padding side must be 'left' for Inference"
    num_examples = len(examples) if examples else None
    system_message = generate_instruction(icl, response_type, considerations, num_examples)
    dataset = PromptDataset(prompts, tokenizer, system_message, examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    generated_text = []

    for batch in tqdm(dataloader):
        tokenized_output = tokenizer(batch, padding=True, return_tensors='pt', truncation=True).to(device)
        default_gen_kwargs = {
            "max_length": tokenizer.model_max_length,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": True,
            "top_k": 100,
            "top_p": 0.7,
            "temperature": 0.8
        }
        default_gen_kwargs.update(gen_kwargs)

        with torch.no_grad():
            gen_tokens = model.generate(**tokenized_output, **default_gen_kwargs)

        for i, tokens in enumerate(gen_tokens):
            prompt_length = len(tokenized_output.input_ids[i])
            gen_text = tokenizer.decode(tokens[prompt_length:], skip_special_tokens=True).strip()
            generated_text.append(gen_text)
 
    return generated_text

if __name__ == "__main__":
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Set up the model and tokenizer
    model_name =  "Qwen/Qwen2-72B"  # You can change this to any model you want to test
    max_memory_mapping = {i: "60GB" for i in range(8)}
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", max_memory=max_memory_mapping)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    # Ensure the model is in evaluation mode
    model.eval()

    # Set up test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain the concept of machine learning in simple terms.",
        "Who wrote the play 'Romeo and Juliet'?"
    ]

    # Set up example prompts for in-context learning
    icl_examples = [
        {"query": "What is the capital of Spain?", 
         "response": "The capital of Spain is Madrid. It is located in the center of the country and is known for its rich history, art museums, and beautiful parks."},
        {"query": "Explain the concept of artificial intelligence in simple terms.", 
         "response": "Artificial Intelligence (AI) is the field of computer science focused on creating machines that can perform tasks that typically require human intelligence. This includes things like problem-solving, learning, and understanding natural language."}
    ]

    print("Running inference without in-context learning:")
    results_without_icl = perform_inference(
        model,
        tokenizer,
        test_prompts,
        batch_size=3,
        icl=False,
        response_type="concise and informative",
        considerations="accuracy and clarity"
    )

    for prompt, result in zip(test_prompts, results_without_icl):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {result}")

    print("\n" + "="*50 + "\n")

    print("Running inference with in-context learning:")
    results_with_icl = perform_inference(
        model,
        tokenizer,
        test_prompts,
        batch_size=3,
        icl=True,
        response_type="detailed and informative",
        considerations="accuracy, clarity, and context",
        examples=icl_examples
    )

    for prompt, result in zip(test_prompts, results_with_icl):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {result}")

    print("\nTest completed. Please review the outputs to ensure they meet the expected criteria.")