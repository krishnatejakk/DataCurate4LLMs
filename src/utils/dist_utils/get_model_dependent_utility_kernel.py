import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def generate_instruction(icl=False, response_type="detailed and informative", 
                         considerations="accuracy, clarity, and context"):
    base_instruction = f"Provide a {response_type} response to the following user query. Consider {considerations} in your answer."
    
    if icl:
        return f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. You will be presented with an example of user query and the corresponding assistant response, followed by a new user query. The given example serve as a guide for the structure, style, and depth of your response. Pay close attention to:

1. The format and organization of the response
2. The level of detail provided
3. Any specific patterns or techniques used in addressing the query

After the example, you will receive a new user query. Apply the insights gained from the example to formulate your response, while also adhering to the following guideline:

{base_instruction}

Remember, while the example is meant to guide you, each user query is unique. Tailor your response to the specific needs of the new user query while maintaining the general approach demonstrated in the example.

Example:
"""
        # return f"You will be presented with an example query and its response, followed by a new query. Use the example as a guide for your response. {base_instruction}\nExample:\n" 
        
    else:
        return "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. " + base_instruction
        # return "Please generate a response to the following query."

class ModelDependentICLUtility:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"Using {self.num_gpus} GPUs")
            self.model = torch.nn.DataParallel(self.model)

        # Set default ChatML template if tokenizer has no set chat template
        if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
        # else:
        #     self.chat_template = self.tokenizer.chat_template
        
        # Compile the Jinja2 template once for efficiency
        # self.compiled_template = jinja2.Template(self.chat_template)

    def compute_distances(self, logits, input_ids, token_type_ids):
        distances = []
        for logit, input_id, token_type_id in zip(logits, input_ids, token_type_ids):
            shifted_logits = logit[:-1, :]
            shifted_input_ids = input_id[1:]
            shifted_token_type_ids = token_type_id[1:]
            valid_positions = (shifted_token_type_ids == 1)
            
            if valid_positions.sum() == 0:
                distances.append(torch.tensor(0.0, device=logit.device))
                continue
            
            valid_logits = shifted_logits[valid_positions]
            valid_labels = shifted_input_ids[valid_positions]
            probs = F.softmax(valid_logits, dim=-1)
            pred_probs = probs.gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)
            num_valid_tokens = valid_positions.sum().float()
            distance = torch.norm(pred_probs - 1.0) / torch.sqrt(num_valid_tokens)
            distances.append(distance)
        
        return torch.stack(distances)

    def compute_utility(self, train_prompts, train_responses, 
                              valid_prompts=None, valid_responses=None,
                              scaling='additive', batch_size=32, 
                              response_type="clear, accurate, and concise", 
                              considerations="relevant facts and context"):
        if valid_prompts is None or valid_responses is None:
            valid_prompts, valid_responses = train_prompts, train_responses

        instruction_no_icl = generate_instruction(icl=False, response_type=response_type, considerations=considerations)
        instruction_with_icl = generate_instruction(icl=True, response_type=response_type, considerations=considerations)

        dataset = ICLDataset(valid_prompts, valid_responses, train_prompts, train_responses, 
                             self.tokenizer, instruction_no_icl, instruction_with_icl)
        
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

        n_train = len(train_prompts)
        n_valid = len(valid_prompts)
        utility_kernel = np.zeros((n_valid, n_train))
        distances_without_icl = np.zeros(n_valid)

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing ICL Utility"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                valid_idxs = batch['valid_idxs']
                train_idxs = batch['train_idxs']

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                distances = self.compute_distances(logits, input_ids, token_type_ids)

                for i in range(len(distances)):
                    valid_idx = valid_idxs[i].item()
                    train_idx = train_idxs[i].item()
                    distance = distances[i].item()
                    if train_idx == -1:
                        distances_without_icl[valid_idx] = distance
                    else:
                        utility = distances_without_icl[valid_idx] - distance
                        utility_kernel[valid_idx, train_idx] = utility

        if scaling == 'min-max':
            min_val = utility_kernel.min()
            max_val = utility_kernel.max()
            utility_kernel = (utility_kernel - min_val) / (max_val - min_val)
        elif scaling == 'additive':
            utility_kernel = utility_kernel - utility_kernel.min()

        return utility_kernel

class ICLDataset(Dataset):
    def __init__(self, valid_prompts, valid_responses, 
                 train_prompts, train_responses, tokenizer, 
                 instruction_no_icl, instruction_with_icl, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_no_icl = instruction_no_icl
        self.instruction_with_icl = instruction_with_icl
        self.valid_prompts = valid_prompts
        self.valid_responses = valid_responses
        self.train_prompts = train_prompts
        self.train_responses = train_responses

    def __len__(self):
        return len(self.valid_prompts) * (len(self.train_prompts) + 1)

    def __getitem__(self, idx):
        num_train_examples = len(self.train_prompts)
        valid_idx = idx // (num_train_examples + 1)
        train_idx = idx % (num_train_examples + 1) - 1

        if train_idx == -1:
            messages = [
                {"role": "system", "content": self.instruction_no_icl},
                {"role": "user", "content": self.valid_prompts[valid_idx]},
                {"role": "assistant", "content": self.valid_responses[valid_idx]}
            ]
        else:
            messages = [
                {"role": "system", "content": self.instruction_with_icl},
                {"role": "user", "content": self.train_prompts[train_idx]},
                {"role": "assistant", "content": self.train_responses[train_idx]},
                {"role": "user", "content": self.valid_prompts[valid_idx]},
                {"role": "assistant", "content": self.valid_responses[valid_idx]}
            ]

        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        encoding = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offsets = encoding['offset_mapping']

        assistant_response_text = self.valid_responses[valid_idx]
        token_type_ids = self.generate_token_type_ids(formatted_text, assistant_response_text, offsets)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "valid_idx": valid_idx,
            "train_idx": train_idx
        }

    def generate_token_type_ids(self, formatted_text, assistant_response_text, offsets):
        assistant_start = formatted_text.rfind(assistant_response_text)
        if assistant_start == -1:
            raise ValueError("Assistant's response not found in the formatted text.")

        token_type_ids = torch.zeros(len(offsets[0]), dtype=torch.long)
        for idx, (start_char, _) in enumerate(offsets[0]):
            if start_char >= assistant_start:
                token_type_ids[idx] = 1

        return token_type_ids

    def collate_fn(self, batch):
        input_ids_list, attention_masks_list, token_type_ids_list = [], [], []
        valid_idxs_list, train_idxs_list = [], []
        # Find the longest sequence in the batch, capped by max_length
        batch_max_length = min(max([item["input_ids"].size(0) for item in batch]), self.max_length)
        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            token_type_ids = item["token_type_ids"]
            valid_idx = item["valid_idx"]
            train_idx = item["train_idx"]

            seq_len = input_ids.size(0)
            if seq_len > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                token_type_ids = token_type_ids[:self.max_length]
            else:
                padding_length = batch_max_length - seq_len
                pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                input_ids = torch.cat([input_ids, torch.full((padding_length,), pad_id, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
                token_type_ids = torch.cat([token_type_ids, torch.zeros(padding_length, dtype=torch.long)])

            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)
            token_type_ids_list.append(token_type_ids)
            valid_idxs_list.append(valid_idx)
            train_idxs_list.append(train_idx)

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_masks_list),
            "token_type_ids": torch.stack(token_type_ids_list),
            "valid_idxs": torch.tensor(valid_idxs_list, dtype=torch.long),
            "train_idxs": torch.tensor(train_idxs_list, dtype=torch.long),
        }

# Example usage
if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Define some example prompts and responses for testing
    train_prompts = [
        "Translate 'Hello, how are you?' to French.",
        "What is the capital of Japan?",
        "Calculate the area of a circle with radius 5 cm.",
        "Explain the concept of photosynthesis in simple terms.",
        "Write a haiku about autumn.",
        "Solve the equation: 2x + 5 = 13",
        "List three renewable energy sources.",
        "What is the difference between a simile and a metaphor?",
        "Convert 150 pounds to kilograms.",
        "Describe the water cycle in 3 steps."
    ]

    train_responses = [
        "The French translation of 'Hello, how are you?' is 'Bonjour, comment allez-vous?'",
        "The capital of Japan is Tokyo.",
        "The area of a circle with radius 5 cm is π * r^2 = π * 5^2 ≈ 78.54 cm².",
        "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar.",
        "Crisp leaves falling down\nGolden hues paint the landscape\nAutumn whispers soft",
        "To solve 2x + 5 = 13, subtract 5 from both sides: 2x = 8. Then divide both sides by 2: x = 4.",
        "Three renewable energy sources are: solar energy, wind energy, and hydroelectric power.",
        "A simile compares two things using 'like' or 'as', while a metaphor directly states that one thing is another.",
        "To convert 150 pounds to kilograms, multiply by 0.45359237. 150 * 0.45359237 ≈ 68.04 kg.",
        "The water cycle in 3 steps: 1. Evaporation: water turns into vapor. 2. Condensation: vapor forms clouds. 3. Precipitation: water falls as rain or snow."
    ]

    valid_prompts = [
        "What is the chemical formula for water?",
        "Calculate the volume of a cube with side length 4 meters.",
        "Translate 'Good morning' to Spanish and German.",
        "Explain the difference between DNA and RNA.",
        "Write a limerick about a cat.",
        "Solve the quadratic equation: x^2 - 5x + 6 = 0",
        "List the first 5 elements of the periodic table.",
        "What are the three states of matter? Give an example of each.",
        "Convert 30°C to Fahrenheit.",
        "Describe the process of mitosis in 4 stages."
    ]

    valid_responses = [
        "The chemical formula for water is H2O.",
        "The volume of a cube with side length 4 meters is 4^3 = 64 cubic meters.",
        "In Spanish: 'Buenos días'. In German: 'Guten Morgen'.",
        "DNA is double-stranded and contains thymine, while RNA is single-stranded and contains uracil instead of thymine. DNA stores genetic information, while RNA helps in protein synthesis.",
        "There once was a cat named Lou\nWho always knew just what to do\nHe'd purr and he'd play\nAll night and all day\nAnd nap in the sun when he's through",
        "Using the quadratic formula: x = [-(-5) ± √((-5)^2 - 4*1*6)] / (2*1)\nx = (5 ± √25 - 24) / 2\nx = (5 ± 1) / 2\nx1 = 3, x2 = 2",
        "The first 5 elements of the periodic table are: 1. Hydrogen (H), 2. Helium (He), 3. Lithium (Li), 4. Beryllium (Be), 5. Boron (B).",
        "The three states of matter are solid, liquid, and gas. Examples: Ice (solid), water (liquid), and water vapor (gas).",
        "To convert 30°C to Fahrenheit: (30°C * 9/5) + 32 = 86°F",
        "Mitosis in 4 stages: 1. Prophase: chromosomes condense. 2. Metaphase: chromosomes align at the equator. 3. Anaphase: sister chromatids separate. 4. Telophase: nuclear envelopes reform, and the cell divides."
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # Ensure use_fast=True

    utility_calculator = ModelDependentICLUtility(model, tokenizer)

    # Define example prompts and responses (same as provided)

    utility_kernel = utility_calculator.compute_utility(
        train_prompts, train_responses, valid_prompts, valid_responses, 
        response_type="clear, accurate, and concise",
        considerations="relevant facts and context"
    )

    print("ICL Utility Kernel:")
    print(utility_kernel)
