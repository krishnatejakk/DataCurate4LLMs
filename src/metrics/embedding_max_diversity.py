import torch
from src.utils.compute_pairwise_similarity import compute_pairwise_dense
from evaluate import Metric, MetricInfo
from typing import Dict, Any

class EmbeddingMaxDiversity(Metric):
    """
    Computes the diversity metric by summing over each embedding the complement of its maximum similarity with any other embedding (excluding itself).
    Diversity = 1/n sum_{i=1}^{n} (1 - max_{j != i} s_{ij})
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
            description="Computes the diversity metric by averaging the complement of maximum similarities excluding self-similarity.",
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

        n = embeddings.size(0)
        batch_size = self.batch_size

        # Initialize a tensor to store max similarities for each embedding
        max_similarities = torch.empty(n, dtype=torch.float32)

        # Process embeddings in batches to compute max similarities
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_embeddings = embeddings[start:end]

            # Compute pairwise similarities between batch and all embeddings
            similarities = compute_pairwise_dense(
                tensor1=batch_embeddings,
                tensor2=embeddings,
                batch_size=self.batch_size,
                metric='cosine',
                scaling='additive'
            )

            # similarities is of shape (batch_size_current, n)
            batch_size_current = end - start

            # Convert similarities to torch tensor if not already
            if not isinstance(similarities, torch.Tensor):
                similarities = torch.tensor(similarities, dtype=torch.float32)

            # Exclude self-similarities
            batch_row_indices = torch.arange(batch_size_current)
            batch_col_indices = torch.arange(start, end)
            similarities[batch_row_indices, batch_col_indices] = -float('inf')

            # Compute max similarity for each embedding in the batch
            max_sim, _ = torch.max(similarities, dim=1)

            # Store the max similarities
            max_similarities[start:end] = max_sim.cpu()

        # Compute diversity score
        diversity_score = torch.mean(1 - max_similarities).item()

        return {"diversity": diversity_score}

# Example usage
if __name__ == "__main__":
    # Generate example embeddings tensor
    embeddings = torch.randn(100, 64)  # Replace with actual embeddings
    metric = EmbeddingMaxDiversity(batch_size=32)
    result = metric._compute(embeddings)
    print("Diversity score (Version 2):", result["diversity"])
