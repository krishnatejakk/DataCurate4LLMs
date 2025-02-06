import os
import math
import logging
import h5py
import torch
import numpy as np
import gc
import glob
import argparse
import json
import re
import time
from multiprocessing import set_start_method
from functools import wraps
from tqdm import tqdm
from dataclasses import dataclass
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from jinja2 import Environment, BaseLoader

# Import your encoder – here we use ArcticEmbedEncoder.
from src.encoders.arctic_encoder import ArcticEmbedEncoder
# (Import other encoder classes as needed.)

# Import your submodular function modules
from submodlib import FacilityLocationFunction
from src.utils.compute_pairwise_similarity import compute_pairwise_dense

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
if int(os.environ.get("RANK", 0)) != 0:
    logger.setLevel(logging.ERROR)

# Ensure that only GPU 0 prints Hugging Face warnings.
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank != 0:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()


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


@dataclass
class ProcessingConfig:
    """
    Configuration for processing.
    """
    instruction: str
    query_description: str
    templates: dict
    batch_size: int = 100000
    num_folds: int = 4            # Number of folds for subset selection
    subset_sizes: list = None     # e.g. [10, "5%"] (absolute numbers or percentages)
    num_gpus: int = 8
    seed: int = 42
    max_retries: int = 3
    retry_delay: int = 30
    output_dir: str = 'output'
    template_name: str = 'conversation'
    combine_files: bool = False
    encoder_type: str = 'arctic'
    encoder_model: str = 'Snowflake/snowflake-arctic-embed-l2.0'
    max_length: int = 4096
    use_fp16: bool = False


class DataProcessor:
    def __init__(self, config: ProcessingConfig, encoder_cls):
        self.config = config
        # Global rank and world size (if available)
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        # Use LOCAL_RANK for device assignment.
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        self.device = torch.device(f"cuda:{self.local_rank}")

        # Initialize encoder on the proper device.
        self.encoder = encoder_cls(
            model_name=config.encoder_model,
            max_length=config.max_length,
            use_fp16=config.use_fp16,
            device=self.device
        )
        self.encoder.device = self.device
        self.encoder.model = self.encoder.model.to(self.device)
        # If desired, wrap with DDP for synchronized operations:
        # self.encoder.model = DDP(self.encoder.model, device_ids=[self.local_rank])
        
        self.env = Environment(loader=BaseLoader())
        self.templates = {k: self.env.from_string(v) for k, v in config.templates.items()}

        # Set random seeds for reproducibility.
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    def format_text(self, example: dict, format_type: str) -> str:
        template = self.templates.get(format_type)
        if not template:
            raise ValueError(f"Unknown format type: {format_type}")
        return template.render(**example)

    def _get_processed_count(self, output_dir: str) -> int:
        """
        Check how many samples have already been processed (across batch files) for this rank.
        """
        pattern = os.path.join(output_dir, f'batch_rank{self.rank}_*.h5')
        batch_files = sorted(glob.glob(pattern))
        count = 0
        for fname in batch_files:
            with h5py.File(fname, 'r') as h5f:
                count += h5f['embeddings'].shape[0]
        return count

    @retry_on_exception
    def generate_embeddings(self, dataset, output_dir: str) -> str:
        """
        Generate embeddings in batches, storing each batch (with its original indices)
        and resuming from the last processed sample.
        """
        os.makedirs(output_dir, exist_ok=True)
        merged_path = os.path.join(output_dir, 'embeddings.h5')
        
        # Only rank 0 checks if merged file exists.
        if os.path.exists(merged_path):
            logger.info(f"Embeddings file already exists in {output_dir}, skipping generation")
            return merged_path

        # Partition the dataset using a DistributedSampler.
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
        sampler_indices = list(sampler)
        total_samples = len(sampler_indices)
        
        # Resume: count how many samples this rank already processed.
        processed_count = self._get_processed_count(output_dir)
        logger.info(f"Rank {self.rank} resuming from {processed_count} already processed samples out of {total_samples}")

        # Assume sampler_indices is a list of indices and processed_count is already defined.
        remaining = [idx for pos, idx in enumerate(sampler_indices) if pos >= processed_count]
        num_batches = math.ceil(len(remaining) / self.config.batch_size)

        # Create a progress bar with total number of batches
        pbar = tqdm(total=num_batches, desc=f"Rank {self.rank} Processing Batches")

        batch_texts = []
        batch_indices = []
        for idx in remaining:
            example = dataset[idx]
            text = self.format_text(example, self.config.template_name)
            batch_texts.append(text)
            batch_indices.append(idx)
            if len(batch_texts) == self.config.batch_size:
                self._process_distributed_batch(batch_texts, batch_indices, output_dir)
                # Reset the lists for the next batch
                batch_texts = []
                batch_indices = []
                # Update the progress bar by one batch
                pbar.update(1)

        # Process any remaining items that didn't fill a complete batch
        if batch_texts:
            self._process_distributed_batch(batch_texts, batch_indices, output_dir)
            pbar.update(1)

        pbar.close()
        # Synchronize all processes.
        dist.barrier()
        if self.rank == 0:
            self._merge_distributed_embeddings(output_dir, merged_path)
        dist.barrier()
        return merged_path

    def _process_distributed_batch(self, batch_texts, batch_indices, output_dir):
        """
        Process one batch of texts, generate embeddings and store them along with
        their original indices.
        """
        embeddings = self.encoder.encode(batch_texts, self.config.instruction)
        if isinstance(embeddings, torch.Tensor):
            emb_np = embeddings.cpu().numpy()
        else:
            emb_np = embeddings
        # Ensure that the number of embeddings matches the number of indices.
        if emb_np.shape[0] != len(batch_indices):
            raise ValueError("Mismatch between embeddings and indices counts")
        # Name the file with rank and current timestamp (you could also include a batch number).
        batch_file = os.path.join(output_dir, f'batch_rank{self.rank}_{int(time.time())}.h5')
        with h5py.File(batch_file, 'w') as h5f:
            h5f.create_dataset('embeddings', data=emb_np, dtype='float32')
            h5f.create_dataset('indices', data=np.array(batch_indices), dtype='int64')
        logger.info(f"Rank {self.rank} saved batch file: {batch_file}")

    def _merge_distributed_embeddings(self, output_dir, merged_path):
        """
        Merge all batch files (from all ranks) into a single HDF5 file.
        The merged file is sorted by the original dataset indices so that the embeddings
        are stored in the same order as the input.
        """
        all_files = []
        for r in range(self.world_size):
            rank_files = sorted(glob.glob(os.path.join(output_dir, f'batch_rank{r}_*.h5')))
            all_files.extend(rank_files)

        if not all_files:
            logger.warning("No batch files found to merge")
            return

        # Gather indices and embeddings from all batch files.
        all_indices = []
        all_embeddings = []
        for fname in all_files:
            with h5py.File(fname, 'r') as f:
                indices = f['indices'][:]
                embeddings = f['embeddings'][:]
                all_indices.append(indices)
                all_embeddings.append(embeddings)
            os.remove(fname)
            logger.info(f"Processed and removed {fname}")

        all_indices = np.concatenate(all_indices)
        all_embeddings = np.concatenate(all_embeddings)
        # Sort the arrays by the original indices.
        sort_order = np.argsort(all_indices)
        all_indices = all_indices[sort_order]
        all_embeddings = all_embeddings[sort_order]
        total_samples = all_embeddings.shape[0]
        emb_dim = all_embeddings.shape[1]

        logger.info(f"Merging {total_samples} samples with embedding dimension {emb_dim}")

        with h5py.File(merged_path, 'w') as h5f_merged:
            h5f_merged.create_dataset('embeddings', data=all_embeddings, dtype='float32')
            h5f_merged.create_dataset('indices', data=all_indices, dtype='int64')
        logger.info(f"Merged embeddings saved to {merged_path}")

    def select_subsets_ddp(self, dataset_name: str, embeddings: torch.Tensor) -> dict:
        """
        Distributed subset selection over a fixed number of folds.
        Each process computes selections for the folds assigned to it.
        Then, results are gathered from all ranks and merged on rank 0.
        """
        num_folds = self.config.num_folds
        total_samples = embeddings.shape[0]
        fold_size = total_samples // num_folds
        folds = []
        start_idx = 0
        for i in range(num_folds):
            extra = 1 if i < total_samples % num_folds else 0
            end_idx = start_idx + fold_size + extra
            folds.append(np.arange(start_idx, end_idx))
            start_idx = end_idx

        # Each process is assigned folds by round-robin.
        local_results = []
        for fold_idx, fold_indices in enumerate(folds):
            if fold_idx % self.world_size != self.rank:
                continue  # Skip folds not assigned to this rank.
            logger.info(f"Rank {self.rank} processing fold {fold_idx+1}/{num_folds}")
            fold_embeddings = embeddings[fold_indices].to(self.device)
            # Compute similarity matrix (tune batch_size and scaling as needed).
            sim_mat = compute_pairwise_dense(
                fold_embeddings,
                batch_size=50000,
                metric='cosine',
                device=self.device,
                scaling="additive"
            )
            sim_mat = sim_mat.cpu().numpy()
            # For each subset size in the configuration, compute a subset.
            fold_result = {}
            for size_spec in self.config.subset_sizes:
                if isinstance(size_spec, float):
                    # Percentage: convert to absolute budget.
                    budget = max(1, math.ceil((size_spec / 100.0) * len(fold_indices)))
                else:
                    # For absolute numbers, we may scale by fold size relative to total.
                    budget = max(1, math.ceil(size_spec * (len(fold_indices) / total_samples)))
                logger.info(f"Rank {self.rank} fold {fold_idx+1}: selecting budget {budget}")
                ds_func = FacilityLocationFunction(
                    n=sim_mat.shape[0],
                    sijs=sim_mat,
                    mode="dense",
                    separate_rep=False
                )
                subset_result = ds_func.maximize(
                    budget=budget,
                    optimizer="LazierThanLazyGreedy",
                    epsilon=160,
                    stopIfZeroGain=False,
                    stopIfNegativeGain=False,
                    verbose=False
                )
                # Map local fold indices back to global indices.
                subset_indices = [fold_indices[x[0]] for x in subset_result]
                subset_gains = [x[1] for x in subset_result]
                fold_result[size_spec] = {"indices": subset_indices, "gains": subset_gains}
            local_results.append((fold_idx, fold_result))
            # Cleanup for the fold.
            del sim_mat, fold_embeddings
            gc.collect()
            torch.cuda.empty_cache()

        # Gather results from all ranks.
        gathered_results = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_results, local_results)
        logger.info(f"Rank {self.rank} gathered results from all ranks")
        
        # Flatten gathered_results (each rank returns a list of fold results).
        all_results = []
        for rank_result in gathered_results:
            if rank_result is not None:
                all_results.extend(rank_result)

        combined_subsets = {size: {"indices": [], "gains": []} for size in self.config.subset_sizes}
        for fold_idx, fold_result in all_results:
            for size in self.config.subset_sizes:
                combined_subsets[size]["indices"].extend(fold_result[size]["indices"])
                combined_subsets[size]["gains"].extend(fold_result[size]["gains"])

        subsets = {}
        for size_spec in self.config.subset_sizes:
            # Calculate the target number of samples.
            if isinstance(size_spec, float):
                actual_size = max(1, int((size_spec / 100.0) * total_samples))
            else:
                actual_size = min(size_spec, total_samples)
            
            # Sort by gains (largest first) and select the top actual_size.
            sorted_candidates = sorted(
                zip(combined_subsets[size_spec]["indices"], combined_subsets[size_spec]["gains"]),
                key=lambda x: x[1],
                reverse=True
            )[:actual_size]
            
            sorted_indices = [x[0] for x in sorted_candidates]
            # Optionally, save metadata.
            metadata_file = os.path.join(
                self.config.output_dir,
                f"{dataset_name}_fl_{num_folds}_subset_{size_spec}.npz"
            )

            if self.rank == 0:
                np.savez(metadata_file, indices=sorted_indices, gains=[x[1] for x in sorted_candidates])

            logger.info(f"Saved metadata to {metadata_file}")
            subsets[size_spec] = sorted_indices

        return subsets

    def process_files(self, input_files: list, output_dir: str):
        """
        Process input files: load dataset, generate embeddings,
        perform distributed subset selection, and save selected subsets.
        """
        try:
            if self.config.combine_files:
                logger.info("Processing combined datasets...")
                from datasets import load_dataset, concatenate_datasets
                datasets = []
                for infile in input_files:
                    file_extension = infile.split('.')[-1]
                    if file_extension == 'jsonl':
                        file_extension = 'json'
                    ds = load_dataset(file_extension, data_files=infile, split='train')
                    datasets.append(ds)
                dataset = concatenate_datasets(datasets)
                dataset_name = "combined_dataset"
            else:
                from datasets import load_dataset
                infile = input_files[0]
                file_extension = infile.split('.')[-1]
                if file_extension == 'jsonl':
                    file_extension = 'json'
                dataset = load_dataset(file_extension, data_files=infile, split='train')
                dataset_name = os.path.splitext(os.path.basename(infile))[0]
            
            # Generate embeddings (with resume functionality).
            emb_dir = os.path.join(output_dir, dataset_name, "embeddings")
            emb_file = self.generate_embeddings(dataset, emb_dir)
            # Load embeddings.
            with h5py.File(emb_file, 'r') as f:
                embeddings_data = f['embeddings'][:]
                if embeddings_data.size == 0:
                    logger.warning(f"No embeddings generated for dataset {dataset_name}, skipping subset selection")
                    return
                embeddings = torch.tensor(embeddings_data, dtype=torch.float32)
            # Perform distributed subset selection.
            subsets = self.select_subsets_ddp(dataset_name, embeddings)
            logger.info("Subset selection complete.")
            # For each subset size, save the corresponding subset.
            if self.rank == 0:
                for size_spec, indices in subsets.items():
                    subset_data = dataset.select(indices)
                    ext = input_files[0].split('.')[-1]
                    subset_name = f"{dataset_name}_subset_{size_spec}"
                    output_file = os.path.join(output_dir, dataset_name, f"{subset_name}.{ext}")
                    self._save_subset(subset_data, output_file, ext)
                    logger.info(f"Saved subset with {len(indices)} samples to {output_file}")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            raise

    def _save_subset(self, subset_data, output_file: str, extension: str):
        if extension in ['json', 'jsonl']:
            subset_data.to_json(output_file, orient='records', lines=True)
        elif extension == 'csv':
            subset_data.to_csv(output_file, index=False)
        elif extension == 'parquet':
            subset_data.to_parquet(output_file)


def main():
    """
    Main entry point. Initializes distributed processing, loads config,
    and runs the processor.
    """
    # Initialize distributed processing.
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    try:
        parser = argparse.ArgumentParser(
            description='Enhanced Data Processing with DDP Subset Selection and Resume Capability'
        )
        parser.add_argument('--input_files', nargs='+', required=True,
                            help='List of input files to process')
        parser.add_argument('--output_dir', required=True,
                            help='Directory to save output files')
        parser.add_argument('--config', required=True,
                            help='Path to config JSON file')
        # Use parse_known_args() to ignore extra arguments.
        args, unknown = parser.parse_known_args()

        with open(args.config) as f:
            config_dict = json.load(f)
        # Process subset_sizes: if percentages (ending with %), convert to float.
        subset_sizes = []
        for size in config_dict.get('subset_sizes', []):
            if isinstance(size, str) and size.endswith('%'):
                subset_sizes.append(float(size[:-1]))
            else:
                subset_sizes.append(int(size))
        config_dict['subset_sizes'] = subset_sizes

        from dataclasses import dataclass
        config = ProcessingConfig(**config_dict)
        config.num_gpus = int(os.environ.get("WORLD_SIZE", 1))
        config.output_dir = args.output_dir

        logger.info(f"Processing configuration: {config}")

        # For demonstration, we choose the Arctic encoder.
        processor = DataProcessor(config, ArcticEmbedEncoder)
        processor.process_files(args.input_files, args.output_dir)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    set_start_method('spawn', force=True)
    main()