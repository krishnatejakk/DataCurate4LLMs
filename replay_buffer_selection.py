import os
import logging
import h5py
import torch
import numpy as np
import gc
import glob
import argparse
from datasets import load_dataset
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from src.encoders.bge_unified_encoder import UnifiedBGEEncoder
from submodlib import FacilityLocationFunction
from multiprocessing import Pool, set_start_method
from src.utils.compute_pairwise_similarity import compute_pairwise_dense
from tqdm import tqdm
from jinja2 import Environment, BaseLoader
import json
import re
import time
from functools import wraps

# Configure logging to display timestamp, module name, log level, and message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mapping from subset sizes to descriptive names
SIZE_TO_REPLAY_DESC = {1: "tiny", 5: "small", 10: "medium", 25: "large", 50: "verylarge"}

@dataclass
class ProcessingConfig:
    """
    Configuration for data processing.
    """
    instruction: str
    query_description: str
    templates: Dict[str, str]
    batch_size: int = 100000
    num_folds: int = 1
    subset_sizes: List[int] = None
    num_gpus: int = 8
    seed: int = 42
    max_retries: int = 3
    retry_delay: int = 30
    output_dir: str = 'output'  # Output directory for saving results

def retry_on_exception(func):
    """
    Decorator to retry a function upon exception up to a maximum number of retries.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.error(f"Attempt {attempt + 1} failed with error: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    logger.info(f"Retrying in {self.config.retry_delay} seconds...")
                    time.sleep(self.config.retry_delay)
                    gc.collect()
                    torch.cuda.empty_cache()
        raise last_exception
    return wrapper

class DataProcessor:
    """
    Processes data by generating embeddings and selecting subsets based on embeddings.
    """
    def __init__(self, config: ProcessingConfig, encoder_cls):
        """
        Initializes the DataProcessor with the given configuration and encoder class.

        Args:
            config (ProcessingConfig): The processing configuration.
            encoder_cls: The encoder class to use for generating embeddings.
        """
        self.config = config
        self.encoder = encoder_cls()
        self.env = Environment(loader=BaseLoader())
        # Compile templates for text formatting
        self.templates = {k: self.env.from_string(v) for k, v in config.templates.items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    def format_text(self, example: Dict[str, Any], format_type: str) -> str:
        """
        Formats the text of an example using the specified template.

        Args:
            example (Dict[str, Any]): The data example to format.
            format_type (str): The key of the template to use.

        Returns:
            str: The formatted text.
        """
        template = self.templates.get(format_type)
        if not template:
            raise ValueError(f"Unknown format type: {format_type}")
        return template.render(**example)

    def get_last_processed_batch(self, output_dir: str) -> Tuple[int, Optional[str]]:
        """
        Retrieves the last processed batch number and its file path from the output directory.

        Args:
            output_dir (str): The directory where batch files are stored.

        Returns:
            Tuple[int, Optional[str]]: The last batch number and its file path.
        """
        batch_files = glob.glob(os.path.join(output_dir, 'batch_*.h5'))
        if not batch_files:
            return -1, None

        # Sort batch files by batch number
        batch_files.sort(key=lambda x: self.extract_batch_number(x))
        max_batch_file = batch_files[-1]
        max_batch_number = self.extract_batch_number(max_batch_file)

        # Return the max batch number and the corresponding batch file path
        return max_batch_number, max_batch_file

    @retry_on_exception
    def process_batch(self, batch_texts: List[str], output_file: str) -> int:
        """
        Processes a batch of texts by generating embeddings and saving them to a file.

        Args:
            batch_texts (List[str]): The list of texts in the batch.
            output_file (str): The path to the output file where embeddings will be saved.

        Returns:
            int: The dimension of the embeddings generated.
        """
        embeddings = self.encoder.encode(
            inputs=batch_texts,
            instruction=self.config.instruction,
            query_description=self.config.query_description
        ).cpu().numpy()

        if embeddings.size == 0:
            logger.warning(f"No embeddings generated for batch, skipping file {output_file}")
            return None  # Return None if there are no embeddings

        embedding_dim = embeddings.shape[1]
        logger.info(f"Embedding dimension for batch: {embedding_dim}")

        # Write embeddings to HDF5 file
        with h5py.File(output_file, 'w') as h5f:
            h5f.create_dataset('embeddings', data=embeddings, dtype='float32', chunks=True)
            h5f.flush()

        return embedding_dim

    @retry_on_exception
    def generate_embeddings(self, dataset, output_dir: str) -> str:
        """
        Generates embeddings for the dataset and saves them to the output directory.

        Args:
            dataset: The dataset to process.
            output_dir (str): The directory where embeddings will be saved.

        Returns:
            str: The path to the merged embeddings file.
        """
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(os.path.join(output_dir, 'embeddings.h5')):
            logger.info(f"Embeddings file already exists in {output_dir}, skipping")
            return os.path.join(output_dir, 'embeddings.h5')
        last_batch, last_batch_file = self.get_last_processed_batch(output_dir)
        if last_batch >= 0:
            logger.info(f"Resuming from batch {last_batch} in {last_batch_file}")
        else:
            logger.info("Starting from scratch")
        batch_texts = []

        # Initialize total_processed based on last batch
        total_processed = (last_batch) * self.config.batch_size if last_batch >= 0 else 0
        if last_batch >= 0:
            # For the last batch, we need to check the number of samples processed
            embedding_size, _ = self.get_embedding_size_dim_from_file(last_batch_file)
            total_processed += embedding_size

        batch_number = last_batch + 1

        # Initialize progress bar
        progress_bar = tqdm(
            desc="Generating embeddings",
            initial=total_processed,
            unit=" samples",
            total=len(dataset)
        )

        # Iterate over dataset examples
        for i, example in enumerate(dataset):
            if i < total_processed:
                continue  # Skip already processed samples

            text = self.format_text(example, dataset.config_name or 'default')
            batch_texts.append(text)

            if len(batch_texts) == self.config.batch_size:
                # Process batch
                batch_file = os.path.join(output_dir, f'batch_{batch_number}.h5')
                self.process_batch(batch_texts, batch_file)
                total_processed += len(batch_texts)
                progress_bar.update(len(batch_texts))
                batch_texts = []
                batch_number += 1
                gc.collect()
                torch.cuda.empty_cache()

        # Process any remaining texts in the final batch
        if batch_texts:
            batch_file = os.path.join(output_dir, f'batch_{batch_number}.h5')
            self.process_batch(batch_texts, batch_file)
            total_processed += len(batch_texts)
            progress_bar.update(len(batch_texts))

        progress_bar.close()

        # Merge all batch embeddings into a single file
        merged_file = os.path.join(output_dir, 'embeddings.h5')
        self.merge_embeddings(output_dir, merged_file, total_samples=total_processed)
        return merged_file

    def extract_batch_number(self, filename):
        """
        Extracts the batch number from the filename.
        Assumes the filename is in the format 'batch_<number>.h5'.

        Args:
            filename (str): The filename from which to extract the batch number.

        Returns:
            int: The batch number extracted from the filename.
        """
        basename = os.path.basename(filename)
        match = re.search(r'batch_(\d+)\.h5$', basename)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Filename {filename} does not match expected pattern.")

    def get_embedding_size_dim_from_file(self, batch_file: str) -> Tuple[int, int]:
        """
        Reads the batch file to determine the embedding size (number of embeddings) and dimension.

        Args:
            batch_file (str): The path to the batch file.

        Returns:
            Tuple[int, int]: A tuple containing the number of embeddings and the embedding dimension.
        """
        with h5py.File(batch_file, 'r') as h5f:
            if 'embeddings' not in h5f:
                raise ValueError(f"The file {batch_file} does not contain 'embeddings' dataset.")
            embeddings = h5f['embeddings']
            embedding_size = embeddings.shape[0]  # Get the number of embeddings
            embedding_dim = embeddings.shape[1]  # Get the embedding dimension
            logger.info(f"Embedding dimension from {batch_file}: {embedding_dim}")
        return embedding_size, embedding_dim

    def merge_embeddings(self, output_dir, merged_file, total_samples):
        """
        Merges all batch embedding files into a single embeddings file.

        Args:
            output_dir (str): The directory where batch embedding files are stored.
            merged_file (str): The path to the merged embeddings file.
            total_samples (int): The total number of samples (embeddings).

        """
        # Find all batch files
        batch_files = glob.glob(os.path.join(output_dir, 'batch_*.h5'))
        if not batch_files:
            logger.warning("No batch files found to merge")
            return

        # Sort batch files by batch number
        batch_files.sort(key=lambda x: self.extract_batch_number(x))

        # Retrieve embedding_dim from the first batch file
        _, embedding_dim = self.get_embedding_size_dim_from_file(batch_files[0])

        if os.path.exists(merged_file):
            logger.info(f"Merged file {merged_file} already exists, skipping merge")
            return

        logger.info(f"Merging {len(batch_files)} batch files into {merged_file} with {total_samples} samples")

        with h5py.File(merged_file, 'w') as h5f_merged:
            # Initialize the dataset in the merged file with the retrieved embedding dimension
            embeddings_ds = h5f_merged.create_dataset(
                'embeddings',
                shape=(total_samples, embedding_dim),
                dtype='float32'
            )

            start_idx = 0
            for batch_file in batch_files:
                with h5py.File(batch_file, 'r') as h5f_batch:
                    if 'embeddings' not in h5f_batch:
                        logger.error(f"File {batch_file} does not contain 'embeddings' dataset")
                        continue

                    embeddings = h5f_batch['embeddings'][:]
                    batch_size = embeddings.shape[0]
                    end_idx = start_idx + batch_size

                    # Check that each file's embedding dimension matches the retrieved embedding_dim
                    if embeddings.shape[1] != embedding_dim:
                        logger.error(f"Embedding dimension mismatch in {batch_file}. Expected {embedding_dim}, got {embeddings.shape[1]}")
                        continue

                    # Copy embeddings into the merged dataset
                    embeddings_ds[start_idx:end_idx] = embeddings
                    start_idx = end_idx

                # Remove the batch file after processing
                os.remove(batch_file)
                logger.info(f"Processed and removed {batch_file}")

            gc.collect()

    def select_subsets(self, input_path: str, embeddings: torch.Tensor) -> Dict[int, List[int]]:
        """
        Selects subsets of the data based on embeddings using submodular optimization.

        Args:
            input_path (str): The path to the input data file.
            embeddings (torch.Tensor): The embeddings tensor.

        Returns:
            Dict[int, List[int]]: A dictionary mapping subset sizes to lists of selected indices.
        """
        # Initialize indices and shuffle
        indices = np.arange(len(embeddings))
        np.random.shuffle(indices)
        
        # Partition data into folds
        fold_size = len(embeddings) // self.config.num_folds
        remainder = len(embeddings) % self.config.num_folds

        folds = []
        start_idx = 0
        for i in range(self.config.num_folds):
            extra = 1 if i < remainder else 0  # Distribute the remainder among the first folds
            end_idx = start_idx + fold_size + extra
            folds.append(indices[start_idx:end_idx])
            start_idx = end_idx

        # Distribute folds across GPUs
        gpu_assignments = []
        folds_per_gpu = self.config.num_folds // self.config.num_gpus
        extra_folds = self.config.num_folds % self.config.num_gpus

        start_fold = 0
        for gpu_id in range(self.config.num_gpus):
            num_folds_this_gpu = folds_per_gpu + (1 if gpu_id < extra_folds else 0)
            end_fold = start_fold + num_folds_this_gpu
            gpu_folds_info = [(fold_idx, folds[fold_idx]) for fold_idx in range(start_fold, end_fold)]
            
            gpu_assignments.append((
                gpu_id,
                gpu_folds_info,
                embeddings,
                self.config.subset_sizes
            ))
            start_fold = end_fold

        # Process folds in parallel using multiprocessing
        with Pool(processes=self.config.num_gpus) as pool:
            gpu_results = pool.map(process_folds_with_gpu, gpu_assignments)

        # Combine results from all GPUs
        all_results = []
        for gpu_result in gpu_results:
            all_results.extend(gpu_result)

        # Initialize combined subsets
        combined_subsets = {size: {"indices": [], "gains": []} for size in self.config.subset_sizes}

        # Aggregate indices and gains from all folds
        for fold_idx, result in all_results:
            for size in self.config.subset_sizes:
                combined_subsets[size]["indices"].extend(result[size]["indices"])
                combined_subsets[size]["gains"].extend(result[size]["gains"])

        # Save metadata with indices and gains
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        subsets = {}
        for size in self.config.subset_sizes:
            # Sort indices by gains in descending order
            sorted_indices_gains = sorted(
                zip(combined_subsets[size]["indices"], combined_subsets[size]["gains"]),
                key=lambda x: x[1],
                reverse=True
            )
            sorted_indices = [x[0] for x in sorted_indices_gains]

            # Save metadata to file
            metadata_file = os.path.join(
                self.config.output_dir, 
                f"{base_name}_fl_{self.config.num_folds}_partitions_{size}_subset_metadata.npz"
            )

            np.savez(
                metadata_file,
                indices=sorted_indices,
                gains=[x[1] for x in sorted_indices_gains]
            )
            logger.info(f"Saved metadata to {metadata_file}")
            subsets[size] = sorted_indices

        return subsets

    def process_file(self, input_path: str, output_dir: str):
        """
        Processes a single input file by generating embeddings and selecting subsets.

        Args:
            input_path (str): The path to the input data file.
            output_dir (str): The directory where outputs will be saved.
        """
        try:
            # Load dataset from input file
            dataset = load_dataset(
                input_path.split('.')[-1],
                data_files=input_path,
                split='train',
                cache_dir=None
            )
            
            logger.info(f"Generating embeddings for {input_path}")
            # Generate embeddings
            embedding_file = self.generate_embeddings(
                dataset, 
                os.path.join(output_dir, 'embeddings')
            )
            
            logger.info("Loading embeddings for subset selection")
            with h5py.File(embedding_file, 'r') as f:
                embeddings = torch.tensor(f['embeddings'][:], dtype=torch.float32)
            
            logger.info("Selecting subsets")
            # Select subsets based on embeddings
            subsets = self.select_subsets(input_path, embeddings)
            
            logger.info("Saving subsets")
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            for size, indices in subsets.items():
                # Create subset of the dataset
                subset_data = dataset.select(indices)

                # Determine size description
                size_desc = SIZE_TO_REPLAY_DESC.get(size, "custom")
                output_file = os.path.join(
                    output_dir, 
                    f"{base_name}_{size_desc}_subset.{input_path.split('.')[-1]}"
                )
                # Save subset data to same format as input
                extension = input_path.split('.')[-1]
                if extension in ['json', 'jsonl']:
                    subset_data.to_json(output_file, orient='records', lines=True)
                elif extension == 'csv':
                    subset_data.to_csv(output_file, index=False)
                elif extension == 'parquet':
                    subset_data.to_parquet(output_file)
                logger.info(f"Saved {size}% subset to {output_file}")
                    
            # Clean up
            os.remove(embedding_file)
            del dataset, embeddings
            gc.collect()
            torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error processing file {input_path}: {str(e)}")
            raise

def process_folds_with_gpu(args):
    """
    Processes folds assigned to a GPU, computing the similarity matrix during subset selection.

    Args:
        args (Tuple): A tuple containing:
            - gpu_id (int): The GPU ID to use.
            - gpu_folds_info (List[Tuple[int, np.ndarray]]): List of fold indices and their corresponding data indices.
            - embeddings (torch.Tensor): The embeddings tensor.
            - subset_sizes (List[int]): List of subset sizes to generate.

    Returns:
        List[Tuple[int, Dict]]: A list of results for each fold, containing the fold index and subset information.
    """
    gpu_id, gpu_folds_info, embeddings, subset_sizes = args
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    results = []
    for fold_idx, fold_indices in gpu_folds_info:
        try:
            logger.info(f"Processing fold {fold_idx + 1} on GPU {gpu_id}")
            
            # Compute similarity matrix for this fold
            fold_embeddings = embeddings[fold_indices].to(device)
            
            logger.info(f"Computing similarity matrix for fold {fold_idx + 1} on GPU {gpu_id}")
            max_sim_mat = compute_pairwise_dense(
                fold_embeddings,
                batch_size=50000,
                metric='cosine',
                device=device,
                scaling="additive"
            )
            similarity_matrix = max_sim_mat.cpu().numpy()
            
            subsets = {}
            ds_func = FacilityLocationFunction(
                n=similarity_matrix.shape[0],
                sijs=similarity_matrix,
                mode="dense",
                separate_rep=False
            )
            
            for size in subset_sizes:
                logger.info(f"Selecting {size}% subset for fold {fold_idx + 1}")
                budget = max(1, int(size / 100 * similarity_matrix.shape[0]))  # Ensure minimum budget of 1
                
                subset_result = ds_func.maximize(
                    budget=budget,
                    optimizer="LazierThanLazyGreedy",
                    epsilon=160,
                    stopIfZeroGain=False,
                    stopIfNegativeGain=False,
                    verbose=False
                )
                
                subset_indices = [fold_indices[x[0]] for x in subset_result]
                subset_gains = [x[1] for x in subset_result]
                subsets[size] = {
                    "indices": subset_indices,
                    "gains": subset_gains
                }
                logger.info(f"Completed {size}% subset selection for fold {fold_idx + 1}")
            
            results.append((fold_idx, subsets))
        except Exception as e:
            logger.error(f"Error processing fold {fold_idx + 1} on GPU {gpu_id}: {str(e)}")
            raise
        finally:
            # Cleanup
            del ds_func, similarity_matrix, fold_embeddings
            gc.collect()
            torch.cuda.empty_cache()
    return results

def main():
    """
    Main function to parse arguments and initiate data processing.
    """
    parser = argparse.ArgumentParser(description='Data Processing with Embedding Generation and Subset Selection')
    parser.add_argument('--input_files', nargs='+', required=True,
                        help='List of input files to process')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save output files')
    parser.add_argument('--config', required=True,
                        help='Path to config JSON file')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs to use')
    parser.add_argument('--max_retries', type=int, default=3,
                        help='Maximum number of retries for failed operations')
    parser.add_argument('--retry_delay', type=int, default=30,
                        help='Delay between retries in seconds')
    args = parser.parse_args()

    try:
        # Load processing configuration from JSON file
        with open(args.config) as f:
            config_dict = json.load(f)

        config = ProcessingConfig(**config_dict)
        # Update config with command-line arguments
        config.num_gpus = min(args.num_gpus, torch.cuda.device_count())
        config.max_retries = args.max_retries
        config.retry_delay = args.retry_delay
        config.output_dir = args.output_dir  # Set output_dir in config

        os.makedirs(args.output_dir, exist_ok=True)

        processor = DataProcessor(config, UnifiedBGEEncoder)

        for input_file in args.input_files:
            logger.info(f"Processing file: {input_file}")
            processor.process_file(input_file, args.output_dir)

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    set_start_method('spawn', force=True)
    main()
