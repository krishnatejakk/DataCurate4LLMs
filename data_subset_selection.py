import os
import math
import logging
import h5py
import torch
import numpy as np
import gc
import glob
import argparse
from datasets import load_dataset, concatenate_datasets
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union
from src.encoders.bge_unified_encoder import UnifiedBGEEncoder
from src.encoders.gte_qwen2_instruct_encoder import Qwen2EmbedEncoder
from src.encoders.nvembed_encoder import NVEmbedEncoder
from src.encoders.openai_encoder import OpenAIEncoder
from src.encoders.sfr_mistral_encoder import SFRMistralEncoder
from submodlib import FacilityLocationFunction
from multiprocessing import Pool, set_start_method
from src.utils.compute_pairwise_similarity import compute_pairwise_dense
from tqdm import tqdm
from jinja2 import Environment, BaseLoader
import json
import re
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """
    Enhanced configuration for data processing.
    """
    instruction: str
    query_description: str
    templates: Dict[str, str]
    batch_size: int = 100000
    num_folds: int = 1
    subset_sizes: List[Union[int, float]] = None  # Can be percentages or absolute numbers
    num_gpus: int = 8
    seed: int = 42
    max_retries: int = 3
    retry_delay: int = 30
    output_dir: str = 'output'
    template_name: str = 'conversation'
    combine_files: bool = False  # New parameter to control file combination , if True, combine all input files before selectig a subset from combined file, otherwise select a subset from each file separately
    encoder_type: str = 'bge' # Encoder Family
    encoder_model: str = 'BAAI/bge-m3' # Encoder Model

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
    Enhanced data processor with support for combined files and multiple selection methods.
    """
    def __init__(self, config: ProcessingConfig, encoder_cls):
        """
        Initializes the DataProcessor with the given configuration and encoder class.

        Args:
            config (ProcessingConfig): The processing configuration.
            encoder_cls: The encoder class to use for generating embeddings.
        """
        self.config = config
        self.encoder = encoder_cls(model_name = config.encoder_model)
        self.env = Environment(loader=BaseLoader())
        self.templates = {k: self.env.from_string(v) for k, v in config.templates.items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds
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

    def load_and_combine_datasets(self, input_files: List[str]) -> Any:
        """
        Load and optionally combine multiple datasets.

        Args:
            input_files (List[str]): List of input file paths.

        Returns:
            Combined dataset or list of individual datasets.
        """
        datasets = []
        
        for input_file in input_files:
            file_extension = input_file.split('.')[-1]
            if file_extension == 'jsonl':
                file_extension = 'json'
            dataset = load_dataset(
                file_extension,
                data_files=input_file,
                split='train',
                cache_dir=None
            )
            datasets.append(dataset)

        if self.config.combine_files:
            logger.info("Combining datasets...")
            return concatenate_datasets(datasets)
        
        if len(datasets) > 1:
            raise ValueError("Multiple datasets provided but combine_files is not enabled")
        return datasets[0]

    def calculate_subset_size(self, total_samples: int, size_spec: Union[int, float]) -> int:
        """
        Calculate the actual subset size based on the specification.

        Args:
            total_samples (int): Total number of samples in the dataset.
            size_spec (Union[int, float]): Size specification (percentage if float, absolute if int).

        Returns:
            int: Actual number of samples to select.
        """
        if isinstance(size_spec, float):
            # Treat as percentage
            return max(1, int(size_spec / 100 * total_samples))
        # Treat as absolute number
        return min(size_spec, total_samples)

    def get_subset_name(self, size_spec: Union[int, float], actual_size: int) -> str:
        """
        Generate appropriate subset name based on selection method.

        Args:
            size_spec (Union[int, float]): Original size specification.
            actual_size (int): Actual number of samples selected.

        Returns:
            str: Descriptive name for the subset.
        """
        if isinstance(size_spec, float):
            return f"percent_{size_spec:.1f}"
        return f"samples_{actual_size}"

    def get_last_processed_batch(self, output_dir: str) -> Tuple[int, Optional[str]]:
        """
        Retrieves the last processed batch number and its file path from the output directory.

        Args:
            output_dir (str): The directory where batch files are stored.

        Returns:
            Tuple[int, Optional[str]]: The last batch number and the corresponding batch file path.
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
        if last_batch >= 0:
            # For the last batch, we need to check the number of samples processed
            embedding_size, _ = self.get_embedding_size_dim_from_file(last_batch_file)
            total_processed = last_batch * self.config.batch_size + embedding_size
        else:
            total_processed = 0

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

            text = self.format_text(example, format_type=self.config.template_name)
            if i < 5:
                logger.info(f"Example {i + 1}: {text}")
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
            
    def select_subsets(self, dataset_name: str, embeddings: torch.Tensor) -> Dict[Union[int, float], List[int]]:
        """
        Enhanced subset selection supporting both percentage and absolute size specifications.
        """
        indices = np.arange(len(embeddings))
        np.random.shuffle(indices)
        
        fold_size = len(embeddings) // self.config.num_folds
        remainder = len(embeddings) % self.config.num_folds

        folds = []
        start_idx = 0
        for i in range(self.config.num_folds):
            extra = 1 if i < remainder else 0
            end_idx = start_idx + fold_size + extra
            folds.append(indices[start_idx:end_idx])
            start_idx = end_idx

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
                self.config.subset_sizes,
                len(embeddings)  # Pass total samples for absolute size calculation
            ))
            start_fold = end_fold

        with Pool(processes=self.config.num_gpus) as pool:
            gpu_results = pool.map(process_folds_with_gpu, gpu_assignments)

        all_results = []
        for gpu_result in gpu_results:
            all_results.extend(gpu_result)

        combined_subsets = {size: {"indices": [], "gains": []} for size in self.config.subset_sizes}
        
        for fold_idx, result in all_results:
            for size in self.config.subset_sizes:
                combined_subsets[size]["indices"].extend(result[size]["indices"])
                combined_subsets[size]["gains"].extend(result[size]["gains"])

        base_name = dataset_name
        subsets = {}
        
        for size_spec in self.config.subset_sizes:
            actual_size = self.calculate_subset_size(len(embeddings), size_spec)
            sorted_indices_gains = sorted(
                zip(combined_subsets[size_spec]["indices"], combined_subsets[size_spec]["gains"]),
                key=lambda x: x[1],
                reverse=True
            )[:actual_size]  # Limit to actual_size

            sorted_indices = [x[0] for x in sorted_indices_gains]
            sorted_gains = [x[1] for x in sorted_indices_gains]

            subset_name = self.get_subset_name(size_spec, actual_size)
            metadata_file = os.path.join(
                self.config.output_dir, 
                f"{base_name}_fl_{self.config.num_folds}_partitions_{subset_name}_metadata.npz"
            )

            np.savez(
                metadata_file,
                indices=sorted_indices,
                gains=sorted_gains
            )
            logger.info(f"Saved metadata to {metadata_file}")
            subsets[size_spec] = sorted_indices

        return subsets
    
    def get_dataset_name(self, input_file: str) -> str:
        """
        Get a clean dataset name from the input file path.
        
        Args:
            input_file (str): Input file path
            
        Returns:
            str: Clean dataset name
        """
        # Extract filename without extension and path
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        # Clean the name to make it filesystem-friendly
        clean_name = re.sub(r'[^\w\-_]', '_', base_name)
        return clean_name

    def process_files(self, input_files: List[str], output_dir: str):
        """
        Process multiple input files with support for both combined and separate processing.
        
        Args:
            input_files (List[str]): List of input files to process
            output_dir (str): Output directory for results
        """
        try:
            if self.config.combine_files:
                # Process combined datasets
                logger.info("Processing combined datasets...")
                dataset = self.load_and_combine_datasets(input_files)
                dataset_name = "combined_dataset"
                
                # Process combined dataset
                self._process_single_dataset(dataset, dataset_name, output_dir, input_files[0])
            else:
                # Process each dataset separately
                logger.info("Processing datasets separately...")
                for input_file in input_files:
                    dataset = self.load_and_combine_datasets([input_file])
                    dataset_name = self.get_dataset_name(input_file)
                    logger.info(f"Processing dataset: {dataset_name}")
                    self._process_single_dataset(dataset, dataset_name, output_dir, input_file)
        
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            raise

    def _process_single_dataset(self, dataset, dataset_name: str, output_dir: str, input_file: str):
        """
        Process a single dataset (either combined or individual).
        
        Args:
            dataset: The dataset to process
            dataset_name (str): Name of the dataset
            output_dir (str): Output directory
            input_file (str): Original input file path (for extension)
        """
        try:
            # Create dataset-specific output directory
            dataset_output_dir = os.path.join(output_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            logger.info(f"Generating embeddings for {dataset_name}")
            embedding_file = self.generate_embeddings(
                dataset, 
                os.path.join(dataset_output_dir, 'embeddings')
            )
            
            logger.info("Loading embeddings for subset selection")
            with h5py.File(embedding_file, 'r') as f:
                embeddings_data = f['embeddings'][:]
                if embeddings_data.size == 0:
                    logger.warning(f"No embeddings generated for dataset {dataset_name}, skipping subset selection")
                    return
                embeddings = torch.tensor(embeddings_data, dtype=torch.float32)
            
            logger.info("Selecting subsets")
            subsets = self.select_subsets(dataset_name, embeddings)
            
            logger.info("Saving subsets")
            for size_spec, indices in subsets.items():
                subset_data = dataset.select(indices)
                subset_name = self.get_subset_name(
                    size_spec,
                    len(indices)
                )
                
                # Create subset filename with dataset name
                output_file = os.path.join(
                    dataset_output_dir, 
                    f"{dataset_name}_{subset_name}_subset.{input_file.split('.')[-1]}"
                )
                
                self._save_subset(subset_data, output_file, input_file)
                logger.info(f"Saved subset with {len(indices)} samples to {output_file}")
            
            # Clean up resources
            del dataset, embeddings
            gc.collect()
            torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
            raise

    def _save_subset(self, subset_data, output_file: str, input_file: str):
        """
        Save subset data to file in appropriate format.
        
        Args:
            subset_data: The dataset subset to save
            output_file (str): Output file path
            input_file (str): Original input file path (for determining format)
        """
        extension = input_file.split('.')[-1]
        if extension in ['json', 'jsonl']:
            subset_data.to_json(output_file, orient='records', lines=True)
        elif extension == 'csv':
            subset_data.to_csv(output_file, index=False)
        elif extension == 'parquet':
            subset_data.to_parquet(output_file)

def process_folds_with_gpu(args):
    """
    Process folds on GPU with support for both percentage and absolute size specifications.
    """
    gpu_id, gpu_folds_info, embeddings, subset_sizes, total_samples = args
    try:
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        results = []
        for fold_idx, fold_indices in gpu_folds_info:
            try:
                logger.info(f"Processing fold {fold_idx + 1} on GPU {gpu_id}")
                
                fold_embeddings = embeddings[fold_indices].to(device)
                
                logger.info(f"Computing similarity matrix for fold {fold_idx + 1}")
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
                
                for size_spec in subset_sizes:
                    if isinstance(size_spec, float):
                        # Percentage-based selection
                        budget = max(1, math.ceil((size_spec / 100) * similarity_matrix.shape[0]))
                    else:
                        # Absolute number-based selection
                        budget = max(1, math.ceil(size_spec * (similarity_matrix.shape[0] / total_samples)))
                    
                    logger.info(f"Selecting subset of size {budget} for fold {fold_idx + 1}")
                    
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
                    subsets[size_spec] = {
                        "indices": subset_indices,
                        "gains": subset_gains
                    }
                    
                results.append((fold_idx, subsets))
                
            except Exception as e:
                logger.error(f"Error processing fold {fold_idx + 1} on GPU {gpu_id}: {str(e)}")
                raise
            finally:
                for var in ['ds_func', 'similarity_matrix', 'fold_embeddings']:
                    if var in locals():
                        del locals()[var]
                gc.collect()
                torch.cuda.empty_cache()
                
        return results
    except Exception as e:
        logger.error(f"Error in process_folds_with_gpu on GPU {gpu_id}: {str(e)}")
        raise

def main():
    """
    Enhanced main function with support for combined processing and multiple selection methods.
    """
    parser = argparse.ArgumentParser(description='Enhanced Data Processing with Combined Files Support')
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
    parser.add_argument('--combine_files', action='store_true',
                        help='Combine all input files before processing')
    parser.add_argument('--subset_sizes', nargs='+', type=str,
                        help='List of subset sizes (use % suffix for percentages, otherwise treated as absolute numbers)')
    args = parser.parse_args()

    try:
        # Load processing configuration from JSON file
        with open(args.config) as f:
            config_dict = json.load(f)

        # Parse subset sizes from command line arguments
        if args.subset_sizes:
            config_dict['subset_sizes'] = args.subset_sizes
        
        # Convert percentage strings to floats and absolute numbers to ints
        subset_sizes = []
        for size in config_dict.get('subset_sizes', []):
            if size.endswith('%'):
                # Convert percentage string to float
                subset_sizes.append(float(size[:-1]))
            else:
                # Convert absolute number string to int
                subset_sizes.append(int(size))
        config_dict['subset_sizes'] = subset_sizes

        config = ProcessingConfig(**config_dict)
        
        # Update config with command-line arguments
        config.num_gpus = min(args.num_gpus, torch.cuda.device_count())
        config.max_retries = args.max_retries
        config.retry_delay = args.retry_delay
        config.output_dir = args.output_dir
        config.combine_files = args.combine_files

        logger.info(f"Processing configuration: {config}")

        # Initialize data processor based on encoder type
        os.makedirs(args.output_dir, exist_ok=True)

        if config.encoder_type == "bge":
            processor = DataProcessor(config, UnifiedBGEEncoder)
        elif config.encoder_type == "openai":
            processor = DataProcessor(config, OpenAIEncoder)
        elif config.encoder_type == "sfr_mistral":
            processor = DataProcessor(config, SFRMistralEncoder)
        elif config.encoder_type == "nvembed":
            processor = DataProcessor(config, NVEmbedEncoder)
        elif config.encoder_type == "qwen2":
            processor = DataProcessor(config, Qwen2EmbedEncoder)
        else:
            raise ValueError(f"Unknown encoder type: {config.encoder_type}")

        processor.process_files(args.input_files, args.output_dir)
 

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

    finally:
        # Cleanup any temporary files or resources
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    set_start_method('spawn', force=True)
    main()
