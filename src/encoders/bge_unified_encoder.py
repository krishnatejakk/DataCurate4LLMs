import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional, Union, Dict
from tqdm import tqdm
import os
import numpy as np

# Model-specific configurations
MODEL_CONFIGS = {
    'BAAI/bge-base-en': {
        'pooling_method': 'cls',
        'normalize_embeddings': True,
        'max_length': 512,
        'default_instruction': 'Represent this sentence for searching relevant passages:' 
    },
    'BAAI/bge-base-en-v1.5': {
        'pooling_method': 'cls',
        'normalize_embeddings': True,
        'max_length': 512,
        'default_instruction': 'Represent this sentence for searching relevant passages:'
    },
    'BAAI/bge-large-en': {
        'pooling_method': 'cls',
        'normalize_embeddings': True,
        'max_length': 512,
        'default_instruction': 'Represent this sentence for searching relevant passages:'
    },
    'BAAI/bge-large-en-v1.5': {
        'pooling_method': 'cls',
        'normalize_embeddings': True,
        'max_length': 512,
        'default_instruction': 'Represent this sentence for searching relevant passages:'
    },
    'BAAI/bge-m3': {
        'pooling_method': 'cls',
        'normalize_embeddings': True,
        'max_length': 4096,
        'default_instruction': 'Use the following sentences to search for relevant passages:'
    },
    'BAAI/bge-multilingual-gemma2': {
        'pooling_method': 'last_token',
        'normalize_embeddings': True,
        'max_length': 4096,
        'default_instruction': 'Represent this for searching:'
    }
}

class UnifiedBGEEncoder:
    def __init__(
        self,
        model_name: str = 'BAAI/bge-multilingual-gemma2',
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        use_fp16: bool = True,
        use_default_instruction: bool = True
    ):
        """
        Unified encoder supporting all BGE model variants with model-specific configurations.
        
        Args:
            model_name: Model identifier from supported BGE variants
            device: Computation device
            batch_size: Base batch size (will be adjusted for multi-GPU)
            use_fp16: Whether to use half-precision
            use_default_instruction: Whether to use the model's default instruction
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} not supported. Supported models: {list(MODEL_CONFIGS.keys())}")
            
        # Load model configuration
        self.config = MODEL_CONFIGS[model_name]
        self.model_name = model_name
        self.use_default_instruction = use_default_instruction
        
        # Initialize device and basic parameters
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up GPU configuration
        self.num_gpus = torch.cuda.device_count()
        self.batch_size = batch_size * self.num_gpus if self.num_gpus > 0 else batch_size
        
        # Suppress tokenizer warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Model configuration
        self.model.eval()
        if use_fp16:
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        
        # Multi-GPU setup
        if self.num_gpus > 1:
            print(f"Using {self.num_gpus} GPUs")
            self.model = torch.nn.DataParallel(self.model)
            
    def _pool_embeddings(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Model-specific pooling method"""
        if self.config['pooling_method'] == 'cls':
            return hidden_states[:, 0]
        elif self.config['pooling_method'] == 'mean':
            s = torch.sum(hidden_states * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.config['pooling_method'] == 'last_token':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return hidden_states[:, -1]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
            
    def _prepare_inputs(self, texts: Union[str, List[str]], instruction: str = "", query_description: str = "") -> List[str]:
        """Prepare inputs with model-specific formatting"""
        if isinstance(texts, str):
            texts = [texts]
            
        # Use model's default instruction if enabled and no custom instruction provided
        if not instruction and self.use_default_instruction and self.config['default_instruction']:
            instruction = self.config['default_instruction']
            
        if instruction:
            if 'bge-multilingual' in self.model_name.lower():
                texts = [f'<instruct>{instruction}\n{query_description}{text}' for text in texts]
            elif 'bge-m3' not in self.model_name.lower():
                texts = [f'{instruction} {text}' for text in texts]
        return texts
        
    @torch.no_grad()
    def encode(
        self,
        inputs: Union[str, List[str]],
        instruction: str = "",
        query_description: str = "",
        show_progress: bool = True,
        return_tensors: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode texts into embeddings.
        
        Args:
            inputs: Input text or list of texts
            instruction: Optional instruction (overrides default if provided)
            query_description: Optional query description (used for specialized instructions)
            show_progress: Whether to show progress bar
            return_tensors: Whether to return pytorch tensors (True) or numpy arrays (False)
        """
        # Input preparation
        input_was_string = isinstance(inputs, str)
        inputs = self._prepare_inputs(inputs, instruction, query_description)
        
        # Tokenization
        encodings = self.tokenizer(
            inputs,
            max_length=self.config['max_length'],
            padding=True,
            truncation=True,
            return_tensors='pt',
            pad_to_multiple_of=8
        ).to(self.device)
        
        # Batch processing
        embeddings_list = []
        for i in tqdm(range(0, len(inputs), self.batch_size), disable=not show_progress or len(inputs) < 256):
            # Prepare batch
            batch = {
                k: v[i:i + self.batch_size] 
                for k, v in encodings.items()
            }
            
            # Model forward pass
            outputs = self.model(**batch)
            hidden_states = outputs.last_hidden_state
            
            # Pool embeddings
            embeddings = self._pool_embeddings(hidden_states, batch['attention_mask'])
            
            # Normalize if configured
            if self.config['normalize_embeddings']:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            embeddings_list.append(embeddings.cpu())
            
            # Clean up GPU memory
            del outputs, hidden_states, embeddings, batch
            torch.cuda.empty_cache()
            
        # Combine embeddings
        embeddings = torch.cat(embeddings_list, dim=0)
        
        # Handle single input case
        if input_was_string:
            embeddings = embeddings[0]
            
        # Return appropriate format
        if return_tensors:
            return embeddings
        return embeddings.numpy()
    
    def encode_queries(self, queries: Union[str, List[str]], instruction: str = "", **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """Specialized method for encoding queries"""
        return self.encode(queries, instruction=instruction, **kwargs)
    
    def encode_corpus(self, corpus: Union[str, List[str]], instruction: str = "", **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """Specialized method for encoding corpus documents"""
        return self.encode(corpus, instruction=instruction, **kwargs)
    
    def embed_dataset(
        self,
        dataset: Dataset,
        column_name: str = 'text',
        instruction: str = "",
        query_description: str = ""
    ) -> Dataset:
        """Embed an entire dataset column"""
        texts = dataset[column_name]
        embeddings = self.encode(
            texts,
            instruction=instruction,
            query_description=query_description,
            return_tensors=False
        )
        return dataset.add_column("embedding", embeddings.tolist())

if __name__ == "__main__":
    # Test with different BGE variants
    test_queries = [
        "What is the capital of France?",
        "How does artificial intelligence work?"
    ]
    
    # Test each model variant
    for model_name in MODEL_CONFIGS.keys():
        print(f"\nTesting with model: {model_name}")
        
        try:
            # Initialize encoder
            encoder = UnifiedBGEEncoder(
                model_name=model_name,
                batch_size=2,
                use_default_instruction=True  # Use model's default instruction
            )
            
            # Test encoding
            embeddings = encoder.encode_queries(test_queries)
            print(f"Query embeddings: {embeddings}")
            print(f"Using pooling method: {encoder.config['pooling_method']}")
            print(f"Default instruction: {encoder.config['default_instruction']}")
            
            # Test with dataset
            from datasets import Dataset
            test_dataset = Dataset.from_dict({
                'text': [
                    "What is the weather like today?",
                    "Explain the theory of relativity."
                ]
            })
            
            embedded_dataset = encoder.embed_dataset(
                test_dataset,
                instruction="Represent the document for retrieval"
            )
            print("\nEmbedded dataset:", embedded_dataset)
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")