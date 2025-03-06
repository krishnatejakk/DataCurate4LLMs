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
import datetime
from filelock import FileLock
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
        torch.cuda.set_device(self.local_rank)

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
        lock_file = os.path.join(output_dir, 'embedding_generation_done.lock')

        # If a lock file exists, assume the generation is complete and skip processing.
        if os.path.exists(lock_file):
            logger.info(f"Rank {self.rank}: Lock file found in {output_dir}, skipping embedding generation.")
            # Wait for all processes to reach this point.
            dist.barrier()
            return merged_path

        # Partition the dataset using a DistributedSampler.
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
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
        text_count = 0
        for idx in remaining:
            example = dataset[idx]
            text = self.format_text(example, self.config.template_name)
            if text_count < 1:
                logger.info(f"Rank {self.rank} processing text: {text}")
                text_count += 1
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
        logger.info(f"Rank {self.rank} waiting at barrier (pre-embedding merge)")
        torch.cuda.synchronize(self.device)  # Ensure all CUDA work is finished
        dist.barrier()  # Barrier with NCCL using the correct device
        logger.info(f"Rank {self.rank} passed barrier")
        if self.rank == 0:
            self._merge_distributed_embeddings(output_dir, merged_path)
        logger.info(f"Rank {self.rank} waiting at barrier (post-embedding merge)")
        torch.cuda.synchronize(self.device)
        dist.barrier()
        logger.info(f"Rank {self.rank} passed barrier")
        return merged_path

    def _process_distributed_batch(self, batch_texts, batch_indices, output_dir):
        embeddings = self.encoder.encode(batch_texts, self.config.instruction)
        if isinstance(embeddings, torch.Tensor):
            emb_np = embeddings.cpu().numpy()
        else:
            emb_np = embeddings
        if emb_np.shape[0] != len(batch_indices):
            raise ValueError("Mismatch between embeddings and indices counts")
        # Use nanosecond resolution for uniqueness (or use a UUID if preferred)
        batch_file = os.path.join(output_dir, f'batch_rank{self.rank}_{time.time_ns()}.h5')
        with h5py.File(batch_file, 'w') as h5f:
            h5f.create_dataset('embeddings', data=emb_np, dtype='float32')
            h5f.create_dataset('indices', data=np.array(batch_indices), dtype='int64')
        logger.info(f"Rank {self.rank} saved batch file: {batch_file}")

    def _merge_distributed_embeddings(self, output_dir, merged_path):
        """
        Merge all batch files (from all ranks) into a single HDF5 file.
        The merged file is sorted by the original dataset indices so that the embeddings
        are stored in the same order as the input. Duplicate indices (from DDP padding)
        are removed.
        """
        all_files = []
        for r in range(self.world_size):
            rank_files = sorted(glob.glob(os.path.join(output_dir, f'batch_rank{r}_*.h5')))
            all_files.extend(rank_files)

        if not all_files:
            logger.warning("No batch files found to merge")
            return

        # Gather indices and embeddings from all batch files.
        all_indices_list = []
        all_embeddings_list = []
        for fname in all_files:
            with h5py.File(fname, 'r') as f:
                if 'indices' not in f or 'embeddings' not in f:
                    logger.warning(f"Skipping corrupted or incomplete file: {fname}")
                    continue
                indices = f['indices'][:]
                embeddings = f['embeddings'][:]
                if indices.size == 0 or embeddings.size == 0:
                    logger.warning(f"Skipping empty file: {fname}")
                    continue
                all_indices_list.append(indices)
                all_embeddings_list.append(embeddings)

        all_indices = np.concatenate(all_indices_list)
        all_embeddings = np.concatenate(all_embeddings_list)

        # Handle duplicate indices by averaging embeddings
        from collections import defaultdict
        index_to_embeddings = defaultdict(list)
        for idx, emb in zip(all_indices, all_embeddings):
            index_to_embeddings[idx].append(emb)

        unique_indices = []
        unique_embeddings = []
        for idx, emb_list in index_to_embeddings.items():
            unique_indices.append(idx)
            unique_embeddings.append(np.mean(emb_list, axis=0))  # or np.sum(emb_list, axis=0)

        unique_indices = np.array(unique_indices)
        unique_embeddings = np.array(unique_embeddings)

        # Sort the unique indices and embeddings
        sort_order = np.argsort(unique_indices)
        sorted_indices = unique_indices[sort_order]
        sorted_embeddings = unique_embeddings[sort_order]

        # Validate alignment
        assert len(sorted_indices) == len(sorted_embeddings), "Mismatch between sorted indices and embeddings"

        total_samples = sorted_embeddings.shape[0]
        emb_dim = sorted_embeddings.shape[1]

        logger.info(f"Merging {total_samples} unique samples with embedding dimension {emb_dim}")

        with h5py.File(merged_path, 'w') as h5f_merged:
            h5f_merged.create_dataset('embeddings', data=sorted_embeddings, dtype='float32')
            h5f_merged.create_dataset('indices', data=sorted_indices, dtype='int64')
        logger.info(f"Merged embeddings saved to {merged_path}")
        
        # Write a lock/marker file indicating that embedding generation is complete.
        lock_file = os.path.join(output_dir, 'embedding_generation_done.lock')
        with open(lock_file, 'w') as f:
            f.write('done')
        logger.info(f"Lock file created at {lock_file}")
       
        # Delete batch files only after successful merge
        # for fname in all_files:
        #     os.remove(fname)
        #     logger.info(f"Removed batch file: {fname}")
            

    def _load_or_create_folds(self, total_samples, num_folds, folds_file):
        """
        Load stored folds from a file if they exist; otherwise, create the folds
        deterministically and save them.
        """
        if os.path.exists(folds_file):
            logger.info(f"Loading stored folds from {folds_file}")
            loaded = np.load(folds_file)
            folds = [loaded[f"fold_{i}"] for i in range(num_folds)]
            return folds
        else:
            fold_size = total_samples // num_folds
            folds = []
            start_idx = 0
            for i in range(num_folds):
                extra = 1 if i < total_samples % num_folds else 0
                end_idx = start_idx + fold_size + extra
                folds.append(np.arange(start_idx, end_idx))
                start_idx = end_idx
            # Save the folds so that subsequent runs use the same fold splits.
            fold_dict = {f"fold_{i}": folds[i] for i in range(num_folds)}
            np.savez(folds_file, **fold_dict)
            logger.info(f"Stored folds to {folds_file}")
            return folds

    def select_subsets_ddp(self, dataset_name: str, embeddings: torch.Tensor) -> dict:
        """
        Distributed subset selection over a fixed number of folds.
        Each process computes selections for the folds assigned to it.
        The fold splits and per‑fold results are stored to allow resuming.
        Then, results are gathered from all ranks and merged on rank 0.
        """
        total_samples = embeddings.shape[0]
        logger.info(f"Rank {self.rank} starting subset selection for {dataset_name} with {total_samples} samples")
        # If num_folds is -1, determine it based on dataset size.
        if self.config.num_folds == -1:
            num_folds = math.ceil(total_samples / 50000)
        else:
            num_folds = self.config.num_folds
        logger.info(f"Rank {self.rank} using {num_folds} folds for subset selection")

        # Define the folds directory and file paths.
        folds_dir = os.path.join(self.config.output_dir, dataset_name, "folds")
        os.makedirs(folds_dir, exist_ok=True)
        folds_file = os.path.join(folds_dir, "folds.npz")
        lock_file = folds_file + ".lock"  # This will be our lock file.

        if self.rank == 0:
            with FileLock(lock_file):
                folds = self._load_or_create_folds(total_samples, num_folds, folds_file)
        # At this point, rank 0 has released the lock, and others can acquire it if needed.
        dist.barrier()  # Synchronize all processes now that the file operation is complete.
        if self.rank != 0:
            loaded = np.load(folds_file)
            folds = [loaded[f"fold_{i}"] for i in range(num_folds)]
        
        torch.cuda.synchronize(self.device)
        dist.barrier()  # Ensure all ranks are synchronized before proceeding
        logger.info(f"Rank {self.rank} passed barrier (folds loaded)")
        local_results = []
        # Process folds assigned to this rank (using round-robin assignment)
        for fold_idx, fold_indices in enumerate(folds):
            if fold_idx % self.world_size != self.rank:
                continue  # Skip folds not assigned to this rank.
            # Define a file to store the fold’s selection result.
            fold_result_file = os.path.join(folds_dir, f"fold_{fold_idx}_selection.npz")
            if os.path.exists(fold_result_file):
                # Resume from stored fold result.
                loaded = np.load(fold_result_file, allow_pickle=True)
                fold_result = loaded['fold_result'].item()  # stored as a dict
                logger.info(f"Rank {self.rank} loaded precomputed selection for fold {fold_idx}")
            else:
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
                        # For absolute numbers, scale by fold size relative to total.
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
                # Save the computed fold result for resuming.
                np.savez(fold_result_file, fold_result=fold_result)
                logger.info(f"Rank {self.rank} saved fold {fold_idx} selection to {fold_result_file}")
                # Cleanup for the fold.
                del sim_mat, fold_embeddings
                gc.collect()
                torch.cuda.empty_cache()
            local_results.append((fold_idx, fold_result))
        # Synchronize all ranks before gathering results.
        logger.info(f"Rank {self.rank} waiting at barrier (post-subset selection)")
        torch.cuda.synchronize(self.device)
        dist.barrier()
        logger.info(f"Rank {self.rank} passed barrier")
        # Gather results from all ranks.
        gathered_results = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_results, local_results)
        logger.info(f"Rank {self.rank} gathered results from all ranks")
        
        # Flatten gathered_results (each rank returns a list of fold results).
        all_results = []
        for rank_result in gathered_results:
            if rank_result is not None:
                all_results.extend(rank_result)

        # Combine results from all folds for each subset size.
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

    @retry_on_exception
    def generate_embeddings_from_files(self, input_files: list, output_dir: str):
        if self.config.combine_files:
            logger.info("Combining datasets...")
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
            # Generate embeddings (with resume functionality).
            emb_dir = os.path.join(output_dir, dataset_name, "embeddings")
            emb_file = self.generate_embeddings(dataset, emb_dir)
        else:
            from datasets import load_dataset
            for infile in input_files:
                file_extension = infile.split('.')[-1]
                if file_extension == 'jsonl':
                    file_extension = 'json'
                dataset = load_dataset(file_extension, data_files=infile, split='train')
                dataset_name = os.path.splitext(os.path.basename(infile))[0]
                
                # Generate embeddings (with resume functionality)
                emb_dir = os.path.join(output_dir, dataset_name, "embeddings")
                emb_file = self.generate_embeddings(dataset, emb_dir)
        return None
    
    def select_subsets_from_files(self, input_files: list, output_dir: str):
        try:
            if self.config.combine_files:
                logger.info("Combining datasets...")
                from datasets import load_dataset, concatenate_datasets
                datasets = []
                for infile in input_files:
                    ext = infile.split('.')[-1]
                    if ext == 'jsonl':
                        ext = 'json'
                    ds = load_dataset(ext, data_files=infile, split='train')
                    datasets.append(ds)
                dataset = concatenate_datasets(datasets)
                dataset_name = "combined_dataset"

                # Load embeddings.
                emb_dir = os.path.join(output_dir, dataset_name, "embeddings")
                emb_file = os.path.join(emb_dir, 'embeddings.h5')
                if not os.path.exists(emb_file):
                    logger.error(f"Embeddings file {emb_file} not found. Please generate embeddings before subset selection.")
                    return
                with h5py.File(emb_file, 'r') as f:
                    embeddings_data = f['embeddings'][:]
                    if embeddings_data.size == 0:
                        logger.warning(f"No embeddings generated for dataset {dataset_name}, skipping subset selection")
                        return
                    embeddings = torch.tensor(embeddings_data, dtype=torch.float32)
                
                # Perform distributed subset selection.
                subsets = self.select_subsets_ddp(dataset_name, embeddings)
                logger.info(f"Rank {self.rank} waiting at saving barrier for getting ready to save")
                torch.cuda.synchronize(self.device)
                dist.barrier()  # Ensure all ranks are ready for saving phase
                logger.info(f"Rank {self.rank} passed pre-saving barrier")
                logger.info("Subset selection complete.")
                
                # Save subsets (only rank 0)
                if self.rank == 0:
                    for size_spec, indices in subsets.items():
                        subset_data = dataset.select(indices)
                        ext = input_files[0].split('.')[-1]
                        subset_name = f"{dataset_name}_subset_{size_spec}"
                        output_file = os.path.join(output_dir, dataset_name, f"{subset_name}.{ext}")
                        self._save_subset(subset_data, output_file, ext)
                        logger.info(f"Saved subset with {len(indices)} samples to {output_file}")
                
                logger.info(f"Rank {self.rank} waiting for saving to finish")
                torch.cuda.synchronize(self.device)
                dist.barrier()  # Ensure all ranks are ready for saving phase
                logger.info(f"Rank {self.rank} passed post-saving barrier")
                # Cleanup between files
                gc.collect()
                torch.cuda.empty_cache()
            else:
                from datasets import load_dataset
                for infile in input_files:
                    file_extension = infile.split('.')[-1]
                    if file_extension == 'jsonl':
                        file_extension = 'json'
                    dataset = load_dataset(file_extension, data_files=infile, split='train')
                    dataset_name = os.path.splitext(os.path.basename(infile))[0]
                    # Load embeddings.
                    emb_dir = os.path.join(output_dir, dataset_name, "embeddings")
                    emb_file = os.path.join(emb_dir, 'embeddings.h5')
                    if not os.path.exists(emb_file):
                        logger.error(f"Embeddings file {emb_file} not found for dataset {dataset_name}. Please generate embeddings before subset selection.")
                        return
                    with h5py.File(emb_file, 'r') as f:
                        embeddings_data = f['embeddings'][:]
                        if embeddings_data.size == 0:
                            logger.warning(f"No embeddings generated for dataset {dataset_name}, skipping subset selection")
                            return
                        embeddings = torch.tensor(embeddings_data, dtype=torch.float32)
                    
                    # Perform distributed subset selection.
                    subsets = self.select_subsets_ddp(dataset_name, embeddings)
                    logger.info(f"Rank {self.rank} waiting at saving barrier for getting ready to save")
                    torch.cuda.synchronize(self.device)
                    dist.barrier()  # Ensure all ranks are ready for saving phase
                    logger.info(f"Rank {self.rank} passed pre-saving barrier")
                    logger.info("Subset selection complete.")
                    
                    # Save subsets (only rank 0)
                    if self.rank == 0:
                        for size_spec, indices in subsets.items():
                            subset_data = dataset.select(indices)
                            ext = infile.split('.')[-1]
                            subset_name = f"{dataset_name}_subset_{size_spec}"
                            output_file = os.path.join(output_dir, dataset_name, f"{subset_name}.{ext}")
                            self._save_subset(subset_data, output_file, ext)
                            logger.info(f"Saved subset with {len(indices)} samples to {output_file}")
                    logger.info(f"Rank {self.rank} waiting for saving to finish")
                    torch.cuda.synchronize(self.device)
                    dist.barrier()  # Ensure all ranks are ready for saving phase
                    logger.info(f"Rank {self.rank} passed saving barrier")
                    # Cleanup between files
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
    # Set up subparsers for two commands: embedding generation and subset selection.
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Choose a command to run: 'embed' for embedding generation, 'subset' for subset selection.")
    
    # Subparser for embedding generation.
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings from input files.")
    embed_parser.add_argument('--input_files', nargs='+', required=True)
    embed_parser.add_argument('--output_dir', required=True)
    embed_parser.add_argument('--config', required=True)
    
    # Subparser for subset selection.
    subset_parser = subparsers.add_parser("subset", help="Select subsets from existing embeddings.")
    subset_parser.add_argument('--input_files', nargs='+', required=True)
    subset_parser.add_argument('--output_dir', required=True)
    subset_parser.add_argument('--config', required=True)
    
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    with open(args.config) as f:
        config_dict = json.load(f)

    # Process subset sizes from the config.
    subset_sizes = []
    for size in config_dict.get('subset_sizes', []):
        if isinstance(size, str) and size.endswith('%'):
            subset_sizes.append(float(size[:-1]))
        else:
            subset_sizes.append(int(size))
    config_dict['subset_sizes'] = subset_sizes

    config = ProcessingConfig(**config_dict)
    config.num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    config.output_dir = args.output_dir

    # Set local_rank in main using environment variable.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if args.command == "embed":
        # Phase 1: Embedding Generation with NCCL
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=36000)
        )
        try:
            if config.encoder_type == "bge":
                from src.encoders.bge_unified_encoder import UnifiedBGEEncoder
                processor = DataProcessor(config, UnifiedBGEEncoder)
            elif config.encoder_type == "openai":
                from src.encoders.openai_encoder import OpenAIEncoder
                processor = DataProcessor(config, OpenAIEncoder)
            elif config.encoder_type == "sfr_mistral":
                from src.encoders.sfr_mistral_encoder import SFRMistralEncoder
                processor = DataProcessor(config, SFRMistralEncoder)
            elif config.encoder_type == "nvembed":
                from src.encoders.nvembed_encoder import NVEmbedEncoder
                processor = DataProcessor(config, NVEmbedEncoder)
            elif config.encoder_type == "arctic":
                from src.encoders.arctic_encoder import ArcticEmbedEncoder
                processor = DataProcessor(config, ArcticEmbedEncoder)
            elif config.encoder_type == "qwen2":
                from src.encoders.gte_qwen2_instruct_encoder import Qwen2EmbedEncoder
                processor = DataProcessor(config, Qwen2EmbedEncoder)
            else:
                raise ValueError(f"Unknown encoder type: {config.encoder_type}")

            processor.generate_embeddings_from_files(args.input_files, args.output_dir)
            torch.cuda.synchronize(local_rank)
            dist.barrier()
        finally:
            dist.destroy_process_group()
            del processor
            gc.collect()
            torch.cuda.empty_cache()
    
    elif args.command == "subset":
        # Phase 2: Subset Selection with Gloo
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            timeout=datetime.timedelta(seconds=36000)
        )
        try:
            # Check that embeddings exist.
            if config.combine_files:
                dataset_name = "combined_dataset"
                emb_dir = os.path.join(args.output_dir, dataset_name, "embeddings")
                emb_file = os.path.join(emb_dir, "embeddings.h5")
                if not os.path.exists(emb_file):
                    logger.error(f"Embeddings file {emb_file} not found. Please generate embeddings before subset selection.")
                    return
            else:
                for infile in args.input_files:
                    dataset_name = os.path.splitext(os.path.basename(infile))[0]
                    emb_dir = os.path.join(args.output_dir, dataset_name, "embeddings")
                    emb_file = os.path.join(emb_dir, "embeddings.h5")
                    if not os.path.exists(emb_file):
                        logger.error(f"Embeddings file {emb_file} not found for dataset {dataset_name}. Please generate embeddings before subset selection.")
                        return
            
            # Dummy encoder for subset selection phase.
            class DummyEncoder:
                def __init__(self, model_name, max_length, use_fp16, device):
                    self.model = torch.nn.Linear(1, 1)
                def encode(self, texts, instruction):
                    raise NotImplementedError("Dummy encoder not used for encoding")
            
            processor = DataProcessor(config, DummyEncoder)
            processor.select_subsets_from_files(args.input_files, args.output_dir)
            torch.cuda.synchronize(local_rank)
            dist.barrier()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    set_start_method('spawn', force=True)
    main()
