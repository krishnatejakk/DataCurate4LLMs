import os
import logging
import json
import torch
import argparse
import numpy as np
import gc
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from src.encoders.bge_unified_encoder import UnifiedBGEEncoder
from src.encoders.gte_qwen2_instruct_encoder import Qwen2EmbedEncoder
from src.encoders.nvembed_encoder import NVEmbedEncoder
from src.encoders.openai_encoder import OpenAIEncoder
from src.encoders.sfr_mistral_encoder import SFRMistralEncoder
from src.utils.compute_pairwise_similarity import compute_pairwise_dense
from src.metrics.embedding_max_diversity import EmbeddingMaxDiversity
from src.metrics.embedding_average_diversity import EmbeddingAverageDiversity
from submodlib import FacilityLocationFunction
from jinja2 import Environment, BaseLoader
from functools import wraps
from kneed import KneeLocator
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def retry_on_exception(func):
    """
    Decorator to retry a function upon exception up to a maximum number of retries.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        last_exception = None
        for attempt in range(self.config.retry_attempts):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.error(f"Attempt {attempt + 1} failed with error: {str(e)}")
                if attempt < self.config.retry_attempts - 1:
                    logger.info(f"Retrying in {self.config.retry_delay} seconds...")
                    time.sleep(self.config.retry_delay)
                    gc.collect()
                    torch.cuda.empty_cache()
        raise last_exception
    return wrapper

@dataclass
class ClusteringConfig:
    """
    Configuration for clustering synthetic data.
    """
    input_file: str
    output_dir: str
    encoder_model: str = "BAAI/bge-large-en-v1.5"
    encoder_type: str = "bge"
    query_instruction: str = "Represent the given query and result pairs for searching similar pairs:\n"
    num_clusters: Optional[int] = None
    template: str = "{% set last_input = item.input | last %}{{ last_input.speaker }}: {{ last_input.text }}\nResult: {{ item.output['reworded version'] }}"
    retry_attempts: int = 3
    retry_delay: int = 10
    seed: int = 42
    batch_size: int = 10000  # Added batch_size attribute

class DataClustering:
    """
    Synthetic data clustering class with enhanced features.
    """
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.env = Environment(loader=BaseLoader())
        self.template = self.env.from_string(config.template)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        print(f"Encoder type: {config.encoder_type}")
        if config.encoder_type == "bge":
            self.encoder = UnifiedBGEEncoder(
                model_name=config.encoder_model
            )
        elif config.encoder_type == "qwen2":
            self.encoder = Qwen2EmbedEncoder(
                model_name=config.encoder_model
            )
        elif config.encoder_type == "nvembed":
            self.encoder = NVEmbedEncoder(
                model_name=config.encoder_model
            )
        elif config.encoder_type == "openai":
            self.encoder = OpenAIEncoder(
                model_name=config.encoder_model
            )
        elif config.encoder_type == "sfr_mistral":
            self.encoder = SFRMistralEncoder(
                model_name=config.encoder_model
            )
        else:
            raise ValueError(f"Invalid encoder type: {config.encoder_type}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @retry_on_exception
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from the input JSON file.
        """
        logger.info(f"Loading data from {self.config.input_file}")
        with open(self.config.input_file, "r") as f:
            data = json.load(f)
        return data

    def format_texts(self, data: List[Dict[str, Any]]) -> List[str]:
        """
        Format texts using the Jinja2 template.
        """
        texts = []
        for idx, item in enumerate(data):
            text = self.template.render(item=item)
            if idx < 5:
                logger.info(f"Formatted text {idx + 1}:\n{text}")
            texts.append(text.strip())
        return texts

    @retry_on_exception
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts into embeddings.
        """
        logger.info("Encoding texts into embeddings")
        embeddings = self.encoder.encode(texts, instruction=self.config.query_instruction)
        return embeddings

    def compute_similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute the similarity matrix.
        """
        logger.info("Computing similarity matrix")
        return compute_pairwise_dense(embeddings, metric="cosine", scaling="additive", device=self.device)

    def facility_location_clustering(self, similarity: torch.Tensor, budget: int):
        """
        Perform facility location clustering.
        """
        logger.info(f"Performing facility location clustering with budget {budget}")
        func = FacilityLocationFunction(
            n=similarity.shape[0],
            mode="dense",
            sijs=similarity.cpu().numpy(),
            separate_rep=False
        )
        return func.maximize(budget=budget, optimizer="LazyGreedy")

    def assign_to_clusters(self, similarity: torch.Tensor, indices: List[int], num_clusters: int) -> Tuple[List[int], List[float]]:
        """
        Assign data points to clusters and record their similarities to the cluster centers.
        """
        logger.info(f"Assigning data points to {num_clusters} clusters")
        centers = torch.tensor(indices[:num_clusters], dtype=torch.long)
        assignments = []
        similarities_to_center = []
        for i in range(similarity.shape[0]):
            similarities = similarity[i, centers]
            max_similarity, closest_center_idx = torch.max(similarities, dim=0)
            assignments.append(closest_center_idx.item())
            similarities_to_center.append(max_similarity.item())
        return assignments, similarities_to_center


    def detect_optimal_clusters(self, gains: List[float]) -> int:
        """
        Detect the optimal number of clusters using the KneeLocator.
        """
        logger.info("Detecting optimal number of clusters using KneeLocator")
        x = np.arange(2, len(gains) + 1)
        kn = KneeLocator(x, gains[1:], curve='convex', direction='decreasing')
        optimal_clusters = kn.knee
        if optimal_clusters is not None:
            optimal_clusters += 1 # Add a buffer of 10 clusters
            optimal_clusters = min(optimal_clusters, len(gains))
            logger.info(f"Optimal number of clusters detected: {optimal_clusters}")
        else:
            logger.warning("KneeLocator did not detect a knee point. Using default max clusters.")
            optimal_clusters = 50
        return optimal_clusters

    def group_by_clusters(self, assignments: List[int], data: List[Dict[str, Any]], similarities: List[float]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Group data samples into clusters based on assignments and sort them by similarity to cluster center.
        """
        logger.info("Grouping data samples into clusters and sorting them by similarity to cluster center")
        clusters = {}
        for idx, (cluster_id, similarity) in enumerate(zip(assignments, similarities)):
            sample = data[idx].copy()  # Copy to avoid modifying the original data
            sample['similarity_to_center'] = similarity  # Add similarity to the data sample
            clusters.setdefault(cluster_id, []).append(sample)
        # Now, sort each cluster's samples by similarity
        for cluster_id in clusters:
            clusters[cluster_id].sort(key=lambda x: x['similarity_to_center'], reverse=True)
        return clusters

    def save_clusters(self, clusters: Dict[int, List[Dict[str, Any]]], output_file: str):
        """
        Save clusters with data samples to a JSON file.
        """
        with open(output_file, "w") as f:
            json.dump(clusters, f, indent=4)
        logger.info(f"Clusters saved to {output_file}")


    def save_elbow_plot(self, gains: List[float], plot_file: str):
        """
        Save the gains plot to visualize the elbow point.
        """
        import matplotlib.pyplot as plt
        cluster_numbers = np.arange(1, len(gains) + 1)
        plt.figure()
        plt.plot(cluster_numbers, gains, marker='o')
        plt.xlabel("Number of Clusters")
        plt.ylabel("Gains")
        plt.title("Submodular Gains vs Number of Clusters")
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"Gains plot saved to {plot_file}")

    def process(self):
        """
        Main processing function.
        """
        try:
            data = self.load_data()
            texts = self.format_texts(data)
            embeddings = self.encode_texts(texts)

            # Compute diversity metrics
            diversity_metric = EmbeddingMaxDiversity(batch_size=self.config.batch_size)
            diversity = diversity_metric._compute(embeddings)
            logger.info(f"Max Diversity metric: {diversity}")

            diversity_metric = EmbeddingAverageDiversity(batch_size=self.config.batch_size)
            diversity = diversity_metric._compute(embeddings)
            logger.info(f"Average Diversity metric: {diversity}")

            # Continue with the rest of the processing
            similarity = self.compute_similarity_matrix(embeddings)

            # Compute submodular gains
            budget = embeddings.shape[0] - 1
            greedy_list = self.facility_location_clustering(similarity, budget)
            indices, gains = zip(*greedy_list)
            gains = np.array(gains)

            # Detect or use predefined number of clusters
            if self.config.num_clusters is None:
                num_clusters = self.detect_optimal_clusters(gains)
                if num_clusters is None:
                    num_clusters = len(gains)
            else:
                num_clusters = self.config.num_clusters
                logger.info(f"Using specified number of clusters: {num_clusters}")

            # Assign clusters and get similarities
            assignments, similarities_to_center = self.assign_to_clusters(similarity, indices, num_clusters)

            # Group data samples into clusters and sort by similarity
            clusters = self.group_by_clusters(assignments, data, similarities_to_center)

            # Save results
            os.makedirs(self.config.output_dir, exist_ok=True)
            # Extract the filename from the input file path
            file_name = os.path.basename(self.config.input_file).split('.')[0]
            file_extension = os.path.basename(self.config.input_file).split('.')[1]
            output_file = os.path.join(self.config.output_dir, file_name + "_clusters." + file_extension)
            self.save_clusters(clusters, output_file)
            plot_file = os.path.join(self.config.output_dir, file_name + "_gains_plot.png")
            self.save_elbow_plot(gains[1:], plot_file)

        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            raise


def main():
    """
    Main function to load configuration and start processing.
    """
    parser = argparse.ArgumentParser(description="Synthetic Data Clustering")
    parser.add_argument("--config_file", help="Path to JSON configuration file",
                        required=True, type=str
                        )
    # Optional command-line arguments to override config file
    parser.add_argument("--num_clusters", type=int, help="Number of clusters (optional, overrides config)")
    parser.add_argument("--retry_attempts", type=int, help="Number of retry attempts (overrides config)")
    parser.add_argument("--retry_delay", type=int, help="Delay between retries in seconds (overrides config)")
    args = parser.parse_args()

    # Load configuration from file
    with open(args.config_file, 'r') as f:
        config_dict = json.load(f)

    # Override config with command-line arguments if provided
    if args.num_clusters is not None:
        config_dict['num_clusters'] = args.num_clusters
    if args.retry_attempts is not None:
        config_dict['retry_attempts'] = args.retry_attempts
    if args.retry_delay is not None:
        config_dict['retry_delay'] = args.retry_delay

    # log the configuration
    logger.info(f"Configuration: {config_dict}")
    # Load template from config or default
    if 'template' in config_dict and config_dict['template']:
        template = config_dict['template']
    else:
        # Default template
        raise ValueError("Template is required in the configuration file")

    # Create ClusteringConfig instance
    config = ClusteringConfig(
        input_file=config_dict['input_file'],
        output_dir=config_dict['output_dir'],
        encoder_model=config_dict.get('encoder_model', 'BAAI/bge-large-en-v1.5'),
        encoder_type=config_dict.get('encoder_type', 'bge'),
        batch_size=config_dict.get('batch_size', 10000),  # Added batch_size attribute
        query_instruction=config_dict.get('query_instruction', 'Represent the given query and result pairs for searching similar pairs:\n'),
        num_clusters=config_dict.get('num_clusters'),
        template=template,
        retry_attempts=config_dict.get('retry_attempts', 3),
        retry_delay=config_dict.get('retry_delay', 10),
        seed=config_dict.get('seed', 42)
    )

    processor = DataClustering(config)
    processor.process()

if __name__ == "__main__":
    main()
