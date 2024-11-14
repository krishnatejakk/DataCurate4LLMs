import torch
import faiss
from torch.nn import functional as F

__DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_pairwise_dense(tensor1, tensor2=None, batch_size=10000, 
                                metric='cosine', device=__DEVICE, scaling=None, kw=0.1):
    """
    Compute pairwise metric in batches between two sets of vectors.

    Args:
        tensor1 (Tensor): Data points for the first set (n_samples1, n_features).
        tensor2 (Tensor, optional): Data points for the second set (n_samples2, n_features). 
                                    Defaults to None, which means tensor1 will be used for self-comparison.
        batch_size (int): Size of each batch for computation.
        metric (str): Metric to compute. Options are 'cosine', 'dot', and 'euclidean'.
        device (str): Device to perform computation ('cuda' or 'cpu').
        scaling (str, optional): Scaling method to apply on the results. Options are 'min-max' or 'additive'.
        kw (float, optional): Kernel width for rbf metric.
    Returns:
        Tensor: Pairwise computed metric as a tensor.

    Note:
    The function performs computations in batches to manage GPU memory usage efficiently.
    similarity measure returned is in cpu memory to save GPU memory.
    If 'tensor2' is None, the function computes the metric for 'tensor1' against itself.
    """
    
    assert batch_size > 0, "Batch size must be positive."

    if tensor2 is None:
        tensor2 = tensor1

    tensor1, tensor2 = tensor1.to(device), tensor2.to(device)

    n_samples1, n_samples2 = tensor1.size(0), tensor2.size(0)

    # Initialize a results matrix in the CPU memory to save GPU memory
    results = torch.zeros(n_samples1, n_samples2, device='cpu')

    # Normalizing tensors if metric is cosine for cosine similarity computation
    if metric == 'cosine':
        tensor1, tensor2 = F.normalize(tensor1, p=2, dim=1), F.normalize(tensor2, p=2, dim=1)

    # Function to calculate the metric
    def calculate_metric(a, b, metric, kw):
        if metric in ['cosine', 'dot']:
            return torch.mm(a, b.T)
        elif metric == 'euclidean':
            distances = torch.cdist(a, b, p=2)
            similarities = 1 / (1 + distances**2)
            return similarities
        elif metric == 'rbf':
            distance = torch.cdist(a, b)
            squared_distance = distance ** 2
            avg_dist = torch.mean(squared_distance)
            torch.div(squared_distance, kw*avg_dist, out=squared_distance)
            torch.exp(-squared_distance, out=squared_distance)
            return squared_distance
        else:
            raise ValueError(f"Unknown metric: {metric}")

   # Process in batches
    for i in range(0, n_samples1, batch_size):
        end_i = min(i + batch_size, n_samples1)
        rows = tensor1[i:end_i]

        for j in range(0, n_samples2, batch_size):
            end_j = min(j + batch_size, n_samples2)
            cols = tensor2[j:end_j]

            batch_results = calculate_metric(rows, cols, metric, kw).cpu()
            results[i:end_i, j:end_j] = batch_results

    # Apply scaling if specified
    if scaling == 'min-max':
        min_val, max_val = results.min(), results.max()
        if max_val != min_val:
            results = (results - min_val) / (max_val - min_val)
    elif scaling == 'additive':
        results = (results + 1) / 2

    return results

def compute_pairwise_sparse(tensor1, tensor2, num_neighbors, batch_size, metric='cosine', scaling=None, 
                            kw=0.1, n_list=100, use_inverse_index=False, device=__DEVICE):
    """
    Compute pairwise similarities between rows of two tensors using a sparse representation.
    """
    if tensor2 is None:
        tensor2 = tensor1

    # Ensure tensors are compatible with FAISS (float32 NumPy arrays)
    tensor1 = tensor1.cpu().float()
    tensor2 = tensor2.cpu().float()

    if metric == 'cosine':
        # Normalize vectors for cosine similarity
        tensor1 = F.normalize(tensor1, p=2, dim=1)
        tensor2 = F.normalize(tensor2, p=2, dim=1)

    # Convert tensors to NumPy arrays
    tensor1_np = tensor1.numpy()
    tensor2_np = tensor2.numpy()

    # Create FAISS index based on the metric
    knn_index = create_faiss_index(tensor2_np, device, metric, use_inverse_index, n_list)

    # Lists to collect indices and values for the sparse tensor
    indices_list = []
    values_list = []

    for i in range(0, tensor1_np.shape[0], batch_size):
        chunk = tensor1_np[i:i + batch_size]
        idx, sim = compute_similarity_chunk(chunk, knn_index, num_neighbors, metric, kw, scaling)

        # idx and sim are torch tensors
        batch_size_actual = chunk.shape[0]

        # Compute row indices for the current chunk
        row_indices = torch.arange(i, i + batch_size_actual).unsqueeze(1).repeat(1, num_neighbors).reshape(-1)
        col_indices = idx.reshape(-1)

        # Stack row and column indices
        indices = torch.stack([row_indices, col_indices], dim=0)
        indices_list.append(indices)
        values_list.append(sim.reshape(-1))

    # Concatenate all indices and values
    indices = torch.cat(indices_list, dim=1)
    values = torch.cat(values_list, dim=0)

    size = (tensor1.shape[0], tensor2.shape[0])

    # Create the sparse tensor
    similarity = torch.sparse_coo_tensor(indices, values, size)
    return similarity

def create_faiss_index(tensor, device, metric, use_inverse_index, n_list):
    """
    Create a FAISS index for nearest neighbor search based on the specified metric.
    """
    if metric == 'cosine' or metric == 'dot':
        index_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == 'euclidean' or metric == 'rbf':
        index_metric = faiss.METRIC_L2
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if use_inverse_index:
        # Create an IVF index
        quantizer = faiss.IndexFlat(tensor.shape[1], index_metric)
        index = faiss.IndexIVFFlat(quantizer, tensor.shape[1], n_list, index_metric)
    else:
        # Create a flat index
        index = faiss.IndexFlat(tensor.shape[1], index_metric)

    # Move index to GPU if needed
    if device != 'cpu':
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    if use_inverse_index:
        index.train(tensor)

    index.add(tensor)
    return index

def compute_similarity_chunk(chunk, knn, num_neighbors, metric, kw, scaling):
    """
    Compute similarities for a chunk of data using the FAISS index.
    """
    # Perform the search
    distances, indices = knn.search(chunk, num_neighbors)

    # Convert distances and indices to torch tensors
    distances = torch.tensor(distances, dtype=torch.float32)
    indices = torch.tensor(indices, dtype=torch.long)

    if metric in ['cosine', 'dot']:
        # For cosine similarity, the distances are inner products
        similarities = distances
    elif metric == 'euclidean':
        # For Euclidean, convert distances to negative distances as similarities
        similarities = 1 / (1 + distances)
    elif metric == 'rbf':
        # For RBF, compute exponential of negative distances
        similarities = torch.exp(-distances / kw)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Apply scaling if specified
    if scaling == 'min-max':
        min_val, max_val = similarities.min(), similarities.max()
        if max_val != min_val:
            similarities = (similarities - min_val) / (max_val - min_val)
    elif scaling == 'additive':
        similarities = (similarities + 1) / 2

    return indices, similarities
