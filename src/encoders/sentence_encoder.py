from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Union, Optional
from tqdm import tqdm
from datasets import Dataset

class SentenceEncoder:
    def __init__(
        self,
        model_name: str = 'BAAI/bge-large-en-v1.5',
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        use_fp16: bool = True,
        use_default_instruction: bool = True
    ):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.use_default_instruction = use_default_instruction
        
        # Initialize model
        self.model = SentenceTransformer(model_name, device=self.device)
        if use_fp16:
            self.model.half()
            
    def _prepare_inputs(self, texts: Union[str, List[str]], instruction: str = "", query_description: str = "") -> List[str]:
        """Match BGE input preparation logic"""
        if isinstance(texts, str):
            texts = [texts]
            
        if instruction:
            if 'bge-multilingual' in self.model_name.lower():
                texts = [f'<instruct>{instruction}\n{query_description}{text}' for text in texts]
            elif 'bge-m3' not in self.model_name.lower():
                texts = [f'{instruction} {text}' for text in texts]
        return texts

    def encode(
        self,
        texts: Union[str, List[str]],
        instruction: str = "",
        query_description: str = "",
        show_progress: bool = True,
        return_tensors: bool = True,
        normalize_embeddings: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """Enhanced encoding method matching BGE functionality"""
        input_was_string = isinstance(texts, str)
        texts = self._prepare_inputs(texts, instruction, query_description)
        
        # Encode with SentenceTransformer
        embeddings = self.model.encode(
            texts,
            device=self.device,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
            normalize_embeddings=normalize_embeddings
        )
        
        # Handle single input case
        if input_was_string:
            embeddings = embeddings[0]
            
        if return_tensors:
            return embeddings.cpu()
        return embeddings.cpu().numpy()
    
    def encode_queries(self, queries: Union[str, List[str]], instruction: str = "", **kwargs):
        """Query encoding wrapper"""
        return self.encode(queries, instruction=instruction, **kwargs)
    
    def encode_corpus(self, corpus: Union[str, List[str]], instruction: str = "", **kwargs):
        """Corpus encoding wrapper"""
        return self.encode(corpus, instruction=instruction, **kwargs)
    
    def embed_dataset(
        self,
        dataset: Dataset,
        column_name: str = 'text',
        instruction: str = "",
        query_description: str = ""
    ) -> Dataset:
        """Dataset embedding method"""
        texts = dataset[column_name]
        embeddings = self.encode(
            texts,
            instruction=instruction,
            query_description=query_description,
            return_tensors=False
        )
        return dataset.add_column("embedding", embeddings.tolist())

if __name__ == "__main__":
    # Test with same examples as BGE
    test_queries = [
        "What is the capital of France?",
        "How does artificial intelligence work?"
    ]
    
    # Initialize encoder
    encoder = SentenceEncoder(
        model_name='BAAI/bge-m3',
        batch_size=2,
        use_default_instruction=True
    )
    
    # Test encoding
    embeddings = encoder.encode_queries(
        test_queries,
        instruction="Represent this sentence for searching relevant passages:"
    )
    print(f"Query embeddings: {embeddings}")
    
    # Test with dataset
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