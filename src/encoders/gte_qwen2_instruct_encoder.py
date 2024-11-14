import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional
from tqdm import tqdm
import os

# Set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Qwen2EmbedEncoder():
    def __init__(
        self,
        model_name: str = 'Alibaba-NLP/gte-Qwen2-7B-instruct',
        device: Optional[torch.device] = None,
        batch_size: int = 2,
        max_length: int = 4096,
        use_fp16: bool = True
    ):
        # Initialize device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Apply fp16 if requested
        if use_fp16:
            self.model = self.model.half()
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Apply DataParallel if multiple GPUs are available
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Using {device_count} GPUs")
            self.model = torch.nn.DataParallel(self.model)
        
        self.num_gpus = device_count
        print(f"Number of GPUs: {self.num_gpus}")

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query_description: str, query: str) -> str:
        return f'{task_description}\n{query_description}\n{query}'
    
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
        Encode input texts into embeddings.
        """
        # Adjust batch size for multi-GPU
        batch_size = self.batch_size * self.num_gpus if self.num_gpus > 1 else self.batch_size

        # Prepare inputs with instruction
        if instruction:
            inputs = [self.get_detailed_instruct(instruction, query_description, input_text) for input_text in inputs]
        
        # Tokenize inputs
        inputs_encodings = self.tokenizer(
            inputs,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs_encodings = {k: v.to(self.device) for k, v in inputs_encodings.items()}
        
        embeddings_list = []
        # Process in batches
        with torch.no_grad():
            for i in tqdm(range(0, len(inputs), batch_size), disable=not show_progress):
                batch = {k: v[i:i+batch_size] for k, v in inputs_encodings.items()}
                outputs = self.model(**batch)
                last_hidden_state = outputs.last_hidden_state
                attention_mask = batch['attention_mask']
                embeddings = self.last_token_pool(last_hidden_state, attention_mask)
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1).cpu()
                embeddings_list.append(embeddings)
        
        embeddings = torch.cat(embeddings_list, dim=0)
        
        if return_tensors:
            return embeddings
        else:
            return embeddings.cpu().numpy()

    def embed_dataset(self, dataset, column_name='text', instruction: str = "", query_description: str = "") -> Dataset:
        """
        Embed an entire dataset column in batches to minimize memory usage.

        Args:
            dataset (Dataset): The dataset to embed.
            column_name (str): The name of the text column to embed.
            instruction (str): Instruction to guide the embedding model.

        Returns:
            Dataset: The dataset with an added 'embedding' column.
        """
        texts = dataset[column_name]
        embeddings = self.encode(inputs=texts, instruction=instruction, query_description=query_description)
        
        # Convert tensors to lists
        embeddings = embeddings.cpu().numpy().tolist()
        return dataset.add_column("embedding", embeddings)
        
if __name__ == "__main__":
    # Test the encoder
    encoder = Qwen2EmbedDecoder(
        model_name='Alibaba-NLP/gte-Qwen2-7B-instruct',
        batch_size=2,
        max_length=512
    )

    # Test queries
    test_queries = [
        "What is the capital of France?",
        "How does artificial intelligence work?"
    ]

    # Test with instruction
    instruction = "Given a question, retrieve relevant passages that answer it."
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
# from typing import List, Dict, Any
# import ray
# from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
# from vllm import LLM
# import torch

# class Qwen2EmbedDecoder:
#     def __init__(
#         self,
#         model_name: str = 'Alibaba-NLP/gte-Qwen2-7B-instruct',
#         max_model_len: int = 16384,
#         use_fp16: bool = True,
#         num_instances: int = torch.cuda.device_count() # ADJUST TO NUMBER OF AVAILABLE GPUS
#     ):
#         """
#         Initialize the Qwen2EmbedDecoder with the specified model and configuration.

#         Args:
#             model_name (str): The name or path of the model to load.
#             max_model_len (int): Maximum sequence length for input texts.
#             use_fp16 (bool): Whether to use FP16 precision.
#             num_instances (int): Number of parallel instances for inference.
#         """
#         self.model_name = model_name
#         self.max_model_len = max_model_len
#         self.use_fp16 = use_fp16
#         self.num_instances = num_instances

#         # Initialize Ray
#         if not ray.is_initialized():
#             ray.init()

#         # Determine the number of available GPUs
#         self.num_gpus = torch.cuda.device_count()
#         if self.num_gpus == 0:
#             raise ValueError("No GPUs available. Please ensure that your system has GPUs and CUDA is properly configured.")

#         # Set tensor parallelism per instance
#         self.tensor_parallel_size = 1

#         # Create placement groups for each instance
#         self.placement_groups = [
#             ray.util.placement_group(
#                 [{
#                     "GPU": self.tensor_parallel_size,
#                     "CPU": 1
#                 }] * self.tensor_parallel_size,
#                 strategy="STRICT_PACK",
#             ) for _ in range(self.num_instances)
#         ]

#         # Wait for placement groups to be ready
#         ray.get([pg.ready() for pg in self.placement_groups])

#         # Initialize LLM instances
#         self.llm_actors = [
#             self._create_llm_actor(pg) for pg in self.placement_groups
#         ]

#     def _create_llm_actor(self, placement_group):
#         @ray.remote(num_gpus=self.tensor_parallel_size, scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group))
#         class LLMPredictor:
#             def __init__(self, model_name, tensor_parallel_size, max_model_len, use_fp16):
#                 self.llm = LLM(
#                     model=model_name,
#                     tensor_parallel_size=tensor_parallel_size,
#                     max_model_len=max_model_len,
#                     dtype="half" if use_fp16 else "float32",
#                     task="embedding"
#                 )

#             def encode(self, inputs: List[str]) -> List[List[float]]:
#                 outputs = self.llm.encode(inputs)
#                 embeddings = [output.outputs.embedding for output in outputs]
#                 return embeddings

#         return LLMPredictor.remote(self.model_name, self.tensor_parallel_size, self.max_model_len, self.use_fp16)

#     def encode(
#         self,
#         inputs: List[str],
#         instruction: str = "",
#         query_description: str = "",
#         show_progress: bool = True,
#     ) -> List[List[float]]:
#         """
#         Encode input texts into embeddings.

#         Args:
#             inputs (List[str]): A list of input texts to encode.
#             instruction (str): Instruction to guide the embedding model.
#             query_description (str): Description of the query.
#             show_progress (bool): Whether to display a progress bar during encoding.

#         Returns:
#             List[List[float]]: A list of embeddings corresponding to the input texts.
#         """
#         # Prepare inputs with instruction if provided
#         if instruction and query_description:
#             inputs = [f'{instruction}\n{query_description}\n{input_text}' for input_text in inputs]

#         # Split inputs among instances
#         input_batches = [inputs[i::self.num_instances] for i in range(self.num_instances)]

#         # Perform inference in parallel
#         futures = [
#             actor.encode.remote(batch) for actor, batch in zip(self.llm_actors, input_batches)
#         ]
#         results = ray.get(futures)

#         # Flatten the list of embeddings
#         embeddings = [embedding for result in results for embedding in result]

#         return embeddings

#     def embed_dataset(self, dataset, column_name='text', instruction: str = "", query_description: str = ""):
#         """
#         Embed an entire dataset column.

#         Args:
#             dataset: The dataset to embed.
#             column_name (str): The name of the text column to embed.
#             instruction (str): Instruction to guide the embedding model.
#             query_description (str): Description of the query.

#         Returns:
#             The dataset with an added 'embedding' column.
#         """
#         texts = dataset[column_name]
#         embeddings = self.encode(inputs=texts, instruction=instruction, query_description=query_description)

#         # Add embeddings to the dataset
#         dataset = dataset.add_column("embedding", embeddings)
#         return dataset

# # Example usage
# if __name__ == "__main__":
#     # Initialize the encoder
#     encoder = Qwen2EmbedDecoder(
#         model_name='Alibaba-NLP/gte-Qwen2-7B-instruct',
#         max_model_len=16384,
#         num_instances=8  # Adjust based on your available resources
#     )

#     # Test queries
#     test_queries = [
#         "What is the capital of France?",
#         "How does artificial intelligence work?"
#     ]

#     # Generate embeddings with instruction
#     instruction = "Given a question, retrieve relevant passages that answer it."
#     query_description = "This is a general knowledge question."
#     embeddings = encoder.encode(test_queries, instruction=instruction, query_description=query_description)
#     for i, emb in enumerate(embeddings):
#         print(f"Embedding for query {i+1}: {emb[:5]}...")  # Displaying the first 5 dimensions for brevity
