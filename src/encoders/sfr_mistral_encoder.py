import token
from torch.utils.data import BatchSampler, SequentialSampler
from src.encoders.base_encoder import BaseEncoder
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# # Each query must come with a one-sentence instruction that describes the task
# task = 'Given a web search query, retrieve relevant passages that answer the query'

class SFRMistralEncoder(BaseEncoder):
    def __init__(self, model_name='Salesforce/SFR-Embedding-Mistral', 
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                 batch_size=20,
                 max_length=4096,
                 use_fp16=False):
        super().__init__(model_name, device, batch_size, tokenizer=True, use_fp16=use_fp16)
        self.max_length = max_length
    
    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode(self, inputs, return_tensors=True):
        if self.num_gpus > 0:
            batch_size = self.batch_size * self.num_gpus
        else:
            batch_size = self.batch_size
        # get the embeddings 
        sampler = BatchSampler(SequentialSampler(range(len(inputs))), 
                               batch_size=batch_size, drop_last=False)
        
        # Progress bar for tokenization
        tokenization_progress = tqdm(total=len(inputs), desc="Tokenizing", position=0)
        
        batched_tokenized_inputs = []
        for indices in sampler:
            inputs_batch = [inputs[x] for x in indices]
            batched_tokenized_inputs.append(self.tokenizer(inputs_batch, 
                                                           max_length=self.max_length, 
                                                           padding=True, 
                                                           truncation=True, 
                                                           return_tensors="pt"))
            tokenization_progress.update(len(indices))
        
        tokenization_progress.close()
        
        # Progress bar for embedding generation
        embedding_progress = tqdm(total=len(batched_tokenized_inputs), desc="Generating Embeddings", position=0)
        
        embeddings = []
        for batch_tokens in batched_tokenized_inputs:
            batch_tokens = {k: v.to(self.device) for k, v in batch_tokens.items()}
            with torch.no_grad():
                outputs = self.model(**batch_tokens)
            batch_embeddings = self.last_token_pool(outputs.last_hidden_state, 
                                              batch_tokens['attention_mask']).cpu()
            embeddings.append(batch_embeddings)
            embedding_progress.update(1)
        
        embedding_progress.close()
        
        print("Concatenating and normalizing embeddings...")
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        if return_tensors:
            return embeddings
        return embeddings.cpu().numpy()
    
    def embed_dataset(self, dataset, column_name='first_user_message'):
        texts = dataset[column_name]
        embeddings = self.encode(texts)
        # Convert numpy array to list of lists
        embeddings = embeddings.tolist()
        return dataset.add_column("embedding", embeddings)
    
if __name__ == "__main__":
    from datasets import Dataset
    import torch

    # Test case for the SFRMistralEncoder class
    # Initialize the SFRMistralEncoder
    encoder = SFRMistralEncoder(
        model_name='Salesforce/SFR-Embedding-Mistral',  # The model name
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # Use GPU if available
        batch_size=2,  # Small batch size for testing
        max_length=512  # Adjust max length for testing
    )

    # Sample input text data for testing
    task_description = "Given a web search query, retrieve relevant passages."
    test_queries = [
        "What is the capital of France?",
        "How does artificial intelligence work?"
    ]

    # Create detailed instructions for each query using the static method
    test_instructions = [get_detailed_instruct(task_description, query) for query in test_queries]

    # Run the encoder to get embeddings for the test instructions
    print("\nTesting the SFRMistralEncoder with task instructions:")
    embeddings = encoder.encode(test_instructions, return_tensors=True)
    print("Embeddings:\n", embeddings)

    # Testing the embedding of a dataset using the encoder
    test_dataset = Dataset.from_dict({
        'first_user_message': [
            "What is the weather like today?",
            "Explain the theory of relativity."
        ]
    })

    print("\nEmbedding a dataset column using SFRMistralEncoder:")
    embedded_dataset = encoder.embed_dataset(test_dataset)
    print(embedded_dataset)
