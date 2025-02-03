import torch
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional
from tqdm import tqdm
import os

# Set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class NVEmbedEncoder():
    def __init__(
        self,
        model_name: str = 'nvidia/NV-Embed-v2',
        device: Optional[torch.device] = None,
        batch_size: int = 8,
        max_length: int = 4096,
        use_fp16: bool = True
    ):
        # Initialize base class without DataParallel (we'll apply it only to embedding_model)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "right"
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Apply fp16 if requested
        if use_fp16:
            self.model = self.model.half()
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Apply DataParallel only to the embedding model if multiple GPUs are available
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Using {device_count} GPUs")
            for module_key, module in self.model._modules.items():
               self.model._modules[module_key] = DataParallel(module)

        self.num_gpus = device_count

    def get_detailed_instruct(self, task_description: str, query_description: str, query: str) -> str:
        return f'Instruct: {task_description}\n{query_description}: {query}{self.tokenizer.eos_token}'

    def encode(
        self,
        inputs: List[str],
        instruction: str = "",
        query_description: str = "",
        return_tensors: bool = True,
        show_progress: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode input texts into embeddings using the model's native encode function.
        """
        # Adjust batch size for multi-GPU if embedding_model is using DataParallel
        if hasattr(self.model, 'embedding_model') and isinstance(self.model.embedding_model, torch.nn.DataParallel):
            batch_size = self.batch_size * self.num_gpus
        else:
            batch_size = self.batch_size

        if instruction:
            inputs = [self.get_detailed_instruct(instruction, query_description, query) for query in inputs]
        else:
            inputs = [query + self.tokenizer.eos_token for query in inputs]
            
        # Use the model's built-in encode method
        with torch.no_grad():
            embeddings = self.model._do_encode(
                prompts=inputs,
                batch_size=batch_size,
                instruction=instruction,
                max_length=self.max_length,
                **kwargs
            )
            
            # Normalize embeddings if they aren't already
            if not kwargs.get('skip_normalize', False):
                embeddings = F.normalize(embeddings, p=2, dim=1)

        if return_tensors:
            return embeddings
        return embeddings.cpu().numpy()

    def embed_dataset(self, dataset, column_name='text', instruction: str = "", query_description: str = ""):
        """
        Embed an entire dataset column in batches to minimize memory usage.

        Args:
            dataset (Dataset): The dataset to embed.
            column_name (str): The name of the text column to embed.
            instruction (str): Instruction to guide the embedding model.
            query_description (str): Description of the query.

        Returns:
            Dataset: The dataset with an added 'embedding' column.
        """
        texts = dataset[column_name]
        embeddings = self.encode(inputs=texts, 
                                instruction=instruction,
                                query_description=query_description,)
        
        # Convert numpy array to list of lists
        embeddings = embeddings.tolist()
        return dataset.add_column("embedding", embeddings)
    
if __name__ == "__main__":
    # Test the encoder
    encoder = NVEmbedEncoder(
        model_name='nvidia/NV-Embed-v2',
        batch_size=2,
        max_length=512
    )

    # Test queries
    test_queries = [
        "What is the capital of France?",
        "How does artificial intelligence work?"
    ]

    # Test with instruction
    instruction = "Given a question, retrieve relevant passages that answer it.\n"
    embeddings = encoder.encode(test_queries, instruction=instruction)
    print("Generated embeddings shape:", embeddings.shape)

    # Test with dataset
    from datasets import Dataset
    test_dataset = Dataset.from_dict({
        'text': [
            "What is the weather like today?",
            "Explain the theory of relativity."
        ]
    })

    embedded_dataset = encoder.embed_dataset(test_dataset, instruction=instruction)
    print("\nEmbedded dataset:", embedded_dataset)