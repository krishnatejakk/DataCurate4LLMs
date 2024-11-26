import torch
from src.utils.compute_pairwise_similarity import compute_pairwise_dense
from evaluate import Metric, MetricInfo
from typing import Dict, Any

class EmbeddingAverageDiversity(Metric):
    """
    Computes the diversity metric as the average dissimilarity over all pairs.
    Diversity = (1 / n^2) * sum_{i,j} (1 - s_{ij})
    """
    def __init__(self, batch_size=10000):
        """
        Initializes the embedding diversity metric using cosine similarity with additive scaling.

        Args:
            batch_size (int): Batch size for computing pairwise similarities.
        """
        self.batch_size = batch_size
        super().__init__()

    def _info(self) -> MetricInfo:
        return MetricInfo(
            description="Computes the average diversity metric for a set of embeddings using cosine similarity with additive scaling.",
            citation="",
            inputs_description="A list or array of embeddings.",
            features=None,
            codebase_urls=[],
            reference_urls=[],
        )

    def _compute(self, embeddings) -> Dict[str, Any]:
        """
        Compute the diversity metric for the provided embeddings.

        Args:
            embeddings (Tensor): A torch tensor of shape (n_samples, embedding_dim).

        Returns:
            dict: A dictionary containing the diversity score.
        """
        # Ensure the input is a torch tensor of type float32
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        else:
            embeddings = embeddings.float()

        # Compute pairwise cosine similarities with additive scaling
        pairwise_similarity = compute_pairwise_dense(
            tensor1=embeddings,
            batch_size=self.batch_size,
            metric='cosine',
        )

        # pairwise_similarity is on CPU, ensure it's a torch tensor
        if not isinstance(pairwise_similarity, torch.Tensor):
            pairwise_similarity = torch.tensor(pairwise_similarity, dtype=torch.float32)

        # Calculate diversity as 1/n^2 * sum(1 - s_ij)
        n = pairwise_similarity.size(0)
        diversity_score = torch.sum(1 - pairwise_similarity) / (n * n)

        return {"diversity": diversity_score.item()}

# Example usage
if __name__ == "__main__":
    # Generate example embeddings tensor
    embeddings = torch.randn(100, 64)  # Replace with actual embeddings
    metric = EmbeddingAverageDiversity(batch_size=32)
    result = metric._compute(embeddings)
    print("Diversity score (Version 1):", result["diversity"])
