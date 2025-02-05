import os
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModel, AutoTokenizer, logging as hf_logging
from typing import List, Optional
from tqdm import tqdm
from datasets import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def safe_print(rank, msg):
    """Only print from rank 0."""
    if rank == 0:
        print(msg, flush=True)


class ArcticEmbedEncoder:
    def __init__(
        self,
        model_name: str = 'Snowflake/snowflake-arctic-embed-l-v2.0',
        batch_size: int = 24,
        max_length: int = 4096,
        use_fp16: bool = False,
    ):
        # Initialize distributed process group if not already done.
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_name = model_name
        self.use_fp16 = use_fp16

        self.device = torch.device(f"cuda:{self.rank}")
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model with rank-specific logging."""
        original_verbosity = hf_logging.get_verbosity()
        try:
            # Suppress Hugging Face warnings on non-root ranks
            if self.rank != 0:
                hf_logging.set_verbosity_error()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                add_pooling_layer=False,
                trust_remote_code=True
            )

            # Use DDP if using multiple GPUs
            if self.world_size > 1:
                self.model = DDP(self.model.to(self.device), device_ids=[self.rank])
            else:
                self.model = self.model.to(self.device)

            if self.use_fp16:
                self.model = self.model.half()

            self.model.eval()
            safe_print(self.rank, "Model loaded successfully.")
        finally:
            # Restore original logging level
            if self.rank != 0:
                hf_logging.set_verbosity(original_verbosity)

    def get_detailed_instruct(self, task: str, desc: str, query: str) -> str:
        return f'Instruct: {task}\n{desc}: {query}{self.tokenizer.eos_token}'

    def encode(self, inputs: List[str], instruction: str = "", return_tensors: bool = True, **kwargs) -> torch.Tensor:
        # Create a dataset that also tracks the original order via an "idx" field.
        dataset = Dataset.from_dict({
            "text": inputs,
            "idx": list(range(len(inputs)))
        })
        # Use embed_dataset to produce the embeddings in the correct order.
        return self.embed_dataset(
            dataset,
            instruction=instruction,
            text_column_name="text",
            embedding_column_name="embedding",
            return_tensors=return_tensors,
            add_to_dataset=False
        )

    def embed_dataset(
        self,
        dataset: Dataset,
        instruction: str = "",
        text_column_name: str = "text",
        embedding_column_name: str = "embedding",
        return_tensors: bool = True,
        add_to_dataset: bool = True,
    ) -> Dataset:
        # Use a DistributedSampler with no shuffling so that order can be restored.
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        # The collate_fn returns both text and original index
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            collate_fn=lambda x: (
                [item[text_column_name] for item in x],
                [item["idx"] for item in x]
            ),
        )

        local_embeds = []
        local_indices = []
        for batch_texts, batch_indices in tqdm(
            dataloader,
            desc=f"Processing on Rank {self.rank}",
            disable=self.rank != 0
        ):
            # If instruction provided, modify the text accordingly.
            if instruction:
                batch_texts = [instruction + ": " + text for text in batch_texts]

            tokens = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**tokens)
                # Take the first token embedding (or [CLS]) and normalize it.
                embeds = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
                local_embeds.append(embeds)
                local_indices.append(torch.tensor(batch_indices, device=self.device))

        # Concatenate the local embeddings and indices
        local_embeds = torch.cat(local_embeds, dim=0) if local_embeds else torch.tensor([], device=self.device)
        local_indices = torch.cat(local_indices, dim=0) if local_indices else torch.tensor([], device=self.device)

        # Gather embeddings and indices from all ranks
        if self.world_size > 1:
            gathered_embeds = [torch.zeros_like(local_embeds) for _ in range(self.world_size)]
            gathered_indices = [torch.zeros_like(local_indices) for _ in range(self.world_size)]
            dist.all_gather(gathered_embeds, local_embeds)
            dist.all_gather(gathered_indices, local_indices)
            all_embeds = torch.cat(gathered_embeds, dim=0)
            all_indices = torch.cat(gathered_indices, dim=0)
        else:
            all_embeds = local_embeds
            all_indices = local_indices

        # Move tensors to CPU after gathering
        all_embeds = all_embeds.cpu()
        all_indices = all_indices.cpu()

        # Reorder the embeddings based on the original indices.
        sorted_order = torch.argsort(all_indices)
        sorted_embeds = all_embeds[sorted_order]

        if not return_tensors:
            sorted_embeds = sorted_embeds.numpy()

        if add_to_dataset:
            # Convert embeddings to list for Hugging Face datasets.
            dataset = dataset.add_column(embedding_column_name, sorted_embeds.tolist())
            return dataset
        return sorted_embeds


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def run_demo():
    try:
        encoder = ArcticEmbedEncoder(batch_size=2, max_length=512)
        # Create some sample conversation texts. Multiply to have enough samples.
        conversations = [
            "User: I've been feeling really down lately...",
            "User: I have a big presentation tomorrow...",
            "User: I just read about the rapid decline in bee populations...",
            "User: I'm planning a trip to Japan next year...",
        ] * 10  # Adjust the number as needed

        if encoder.rank == 0:
            print("Last four conversations:")
            print(conversations)

        # Encode the texts using the encoder.encode method.
        embeddings = encoder.encode(conversations, instruction="Retrieve relevant passages.")
        if encoder.rank == 0:
            print("\nEncode results:")
            for i, (text, emb) in enumerate(zip(conversations, embeddings)):
                print(f"{i+1}. {text[:50]}... -> Embedding shape: {emb.shape}")

        # Demonstrate using embed_dataset directly.
        dataset = Dataset.from_dict({
            "text": conversations,
            "idx": list(range(len(conversations)))
        })
        embedded_ds = encoder.embed_dataset(
            dataset,
            instruction="Retrieve relevant passages.",
            add_to_dataset=True
        )
        if encoder.rank == 0:
            print("\nDataset results:")
            print(embedded_ds)

        # Also show an example of returning numpy arrays.
        embeddings_np = encoder.encode(conversations, instruction="Retrieve relevant passages.", return_tensors=False)
        if encoder.rank == 0:
            print("\nNumpy array results:")
            print(embeddings_np, embeddings_np.shape)
    except Exception as e:
        safe_print(dist.get_rank(), f"Demo failed: {str(e)}")
    finally:
        cleanup()


if __name__ == "__main__":
    run_demo()
