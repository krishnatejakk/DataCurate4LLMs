import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, logging as hf_logging
from typing import List, Optional
from tqdm import tqdm
from datasets import Dataset

# Ensure tokenizers do not spawn extra workers.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ArcticEmbedEncoder:
    def __init__(
        self,
        model_name: str = 'Snowflake/snowflake-arctic-embed-l-v2.0',
        batch_size: int = 32,
        max_length: int = 4096,
        use_fp16: bool = False,
        device: Optional[torch.device] = None   # NEW parameter
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_name = model_name
        self.use_fp16 = use_fp16
        # Use the provided device or default to cuda:0 if not provided.
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._initialize_model()

    def _initialize_model(self):
        """Load the model and tokenizer."""
        hf_logging.set_verbosity_info()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            add_pooling_layer=False,
            trust_remote_code=True
        )
        # Move model to the provided device.
        self.model = self.model.to(self.device)
        if self.use_fp16:
            self.model = self.model.half()
        self.model.eval()
        print("Model loaded successfully on device", self.device)

    def get_detailed_instruct(self, task: str, desc: str, query: str) -> str:
        return f'Instruct: {task}\n{desc}: {query}{self.tokenizer.eos_token}'

    def encode(self, inputs: List[str], instruction: str = "", return_tensors: bool = True, **kwargs) -> torch.Tensor:
        dataset = Dataset.from_dict({
            "text": inputs,
            "idx": list(range(len(inputs)))
        })
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
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: (
                [item[text_column_name] for item in x],
                [item["idx"] for item in x]
            ),
        )

        all_embeds = []
        all_indices = []
        for batch_texts, batch_indices in tqdm(
            dataloader,
            desc="Encoding batch"
        ):
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
                embeds = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
                all_embeds.append(embeds.cpu())
                all_indices.extend(batch_indices)

        if all_embeds:
            all_embeds = torch.cat(all_embeds, dim=0)
        else:
            all_embeds = torch.tensor([])

        if return_tensors:
            return all_embeds
        else:
            return all_embeds.numpy()
