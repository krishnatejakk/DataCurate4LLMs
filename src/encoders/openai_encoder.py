import os
import time
import torch
from tqdm import tqdm
import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
from src.encoders.base_encoder import BaseEncoder

class OpenAIEncoder(BaseEncoder):
    def __init__(self, model_name="text-embedding-3-large", device="cpu", batch_size=20000, max_attempts=3, timeout=10):
        super().__init__(model_name, device, batch_size)
        self.model = self.initialize_model(model_name, device)
        self.max_attempts = max_attempts
        self.timeout = timeout

    def get_api_key(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return api_key

    def initialize_model(self, model_name, device, use_fp16=False):
        # Model initialized once with api_key already retrieved in __init__
        return OpenAIEmbeddings(model=model_name, api_key=self.get_api_key())

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def embed_with_retry(self, texts):
        try:
            return self.model.embed_documents(texts)
        except Exception as e:
            print(f"Retryable error embedding texts: {e}")
            raise  # Let tenacity handle retry

    def encode(self, inputs, return_tensors=True, return_numpy=False, **kwargs):
        embeddings = []
        for i in tqdm(range(0, len(inputs), self.batch_size)):
            batch = inputs[i:i + self.batch_size]
            batch_embeddings = self.embed_batch(batch, i)
            embeddings.extend(batch_embeddings)
        
        if return_tensors:
            return torch.tensor(embeddings)
        
        if return_numpy:
            return np.array(embeddings)
        
        return embeddings

    def embed_batch(self, batch, batch_index):
        try:
            return self.embed_with_retry(batch)
        except Exception as e:
            print(f"Failed to embed batch at index {batch_index} after retries: {e}")
            return [[0.0] * 3072] * len(batch)  # Assuming 768-dim embeddings as fallback

    def embed_dataset(self, dataset, column_name='first_user_message'):
        texts = dataset[column_name]
        embeddings = self.encode(texts)
        return dataset.add_column("embedding", embeddings.tolist())

if __name__ == "__main__":
    from datasets import Dataset
    encoder = OpenAIEncoder(
        model_name="text-embedding-3-large",
        device="cpu",
        batch_size=2,
        max_attempts=3,
        timeout=10
    )

    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial Intelligence is transforming the world.",
        "What is the capital of France?"
    ]
    print("\nTesting the OpenAIEncoder:")
    embeddings = encoder.encode(test_texts)
    print("Embeddings:\n", embeddings)

    test_dataset = Dataset.from_dict({
        'first_user_message': [
            "The sky is blue.",
            "OpenAI is advancing research in AI.",
            "How do you solve a complex problem?"
        ]
    })
    print("\nEmbedding a dataset column:")
    embedded_dataset = encoder.embed_dataset(test_dataset)
    print(embedded_dataset)
