# Subset Selection for Language Models

This repository contains code for selecting representative subsets from large datasets, which can be used as replay buffers for fine-tuning language models efficiently, analyzing data redundancy, and more. The goal is to enable efficient training by reducing dataset size without significant loss of information, leveraging advanced embedding techniques and submodular optimization.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Upcoming Features](#upcoming-features)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Configuration File](#configuration-file)
- [Code Overview](#code-overview)
  - [Data Processing Pipeline](#data-processing-pipeline)
  - [Encoders](#encoders)
  - [Memory-Efficient Similarity Computation](#memory-efficient-similarity-computation)
- [Examples](#examples)
  - [Subset Selection for New Use Cases](#subset-selection-for-new-use-cases)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In large-scale language model training, it's often beneficial to select a subset of the data that is most representative or informative. This subset can serve as a replay buffer for continual learning, reduce computational costs, and help in analyzing data redundancy.

This repository provides tools to:

- Generate embeddings for large datasets using different encoders.
- Compute pairwise similarities efficiently, even for large datasets.
- Select subsets based on submodular optimization techniques.
- Support various data formats and customizable processing configurations.

## Features

- **Multiple Encoders**: Easily switch between different embedding models, such as OpenAI embeddings, BGE embeddings, Sentence Transformers, NVIDIA's NV-Embed-v2, and Alibaba's GTE-Qwen2-7B-Instruct.
- **Memory-Efficient Similarity Computation**: Compute pairwise similarities using optimized algorithms to handle large datasets without exhausting memory resources.
- **Submodular Optimization for Subset Selection**: Use submodular functions like Facility Location to select representative subsets.
- **Robustness and Fault Tolerance**: Includes retry mechanisms and logging to handle transient errors and monitor progress.
- **Parallel Processing**: Leverage multiple GPUs and multiprocessing to speed up computations.
- **Compression-Based Distances**: Compression-based distance metrics to capture semantic similarities between very large documents.
- **Evaluation Tools**: Includes scripts for evaluation tasks like in-context learning (ICL) and inference.

## Upcoming Features
- **Instruction Tuning**: Tools for fine-tuning models on instruction-following data (`src/train/instruction_tuner.py`).
- **Additional Encoders**: Integration with more encoders, including those optimized for specific domains or languages.
- **Enhanced Evaluation Framework**: Expanded evaluation scripts for benchmarking subsets in various downstream tasks.

## Installation

### Prerequisites

- Python 3.7 or higher
- [PyTorch](https://pytorch.org/get-started/locally/) (ensure compatibility with your CUDA version if using GPUs)

### Clone the Repository

```bash
git clone https://github.ibm.com/conversational-ai/subset_selection_and_analysis.git
cd replay-buffer-selection
```

### Install Dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

Install Submodlib from the source (recommended):
```bash
pip install git+https://github.com/decile-team/submodlib.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**

```
h5py
torch
numpy
datasets
submodlib
tqdm
jinja2
faiss-gpu
langchain
python-dotenv
tenacity
sentence-transformers
```

**Note:** If you don't have access to a GPU or prefer not to use one, replace `faiss-gpu` with `faiss-cpu` in the `requirements.txt` and install accordingly.

## Usage

The main script is designed to process input data files, generate embeddings, and select subsets based on the configuration provided.

### Command-Line Arguments

```bash
python replay_buffer_selection.py --input_files <file1> <file2> ... --output_dir <output_directory> --config <config.json> [--num_gpus <n>] [--max_retries <n>] [--retry_delay <seconds>]
```

- `--input_files`: List of input data files to process.
- `--output_dir`: Directory to save output files.
- `--config`: Path to the JSON configuration file.
- `--num_gpus`: (Optional) Number of GPUs to use (default is 8).
- `--max_retries`: (Optional) Maximum number of retries for failed operations (default is 3).
- `--retry_delay`: (Optional) Delay between retries in seconds (default is 30).

### Configuration File

Create a JSON configuration file (e.g., `config.json`) to specify processing parameters. You can find a example configuration file here: **configs/example_config.json**

**Example `config.json`:**

```json
{
  "instruction": "Generate embeddings that capture the core meaning of user-assistant conversations across multiple domains, ensuring generalization and suitability for clustering based on semantic similarity.",
  "query_description": "Conversation",
  "templates": {
    "default": "{{ text }}",
    "conversation": "{% for conv in conversations %}{{ conv.from }}: {{ conv.value }}\n{% endfor %}",
    "qa": "Question: {{ question }}\nAnswer: {{ answer }}"
  },
  "batch_size": 100000,
  "num_folds": 10,
  "subset_sizes": [50, 25, 10, 5, 1],
  "seed": 42,
  "num_gpus": 4,
  "max_retries": 3,
  "retry_delay": 30,
  "output_dir": "output"
}
```

In this configuration:

- **Instruction**: Custom instruction guiding the encoder to generate embeddings suitable for clustering conversations across multiple domains.
- **Templates**: Multiple templates to format different types of data, such as conversations and QA pairs.
- **Subset Sizes**: Specifies larger subset sizes to accommodate various levels of data reduction.

### Selecting an Encoder

You can specify which encoder to use by modifying the main script or configuration:

**Example using NVEmbedEncoder:**

```python
from src.encoders.nvembed_encoder import NVEmbedEncoder

processor = DataProcessor(config, NVEmbedEncoder)
```

Ensure that the selected encoder is properly configured and any required models are downloaded or accessible.

## Code Overview

### Data Processing Pipeline

The data processing pipeline involves the following steps:

1. **Data Loading**: Load the dataset from the specified input files using `datasets.load_dataset`.
2. **Text Formatting**: Use Jinja2 templates to format the text fields as required by the encoder.
3. **Embedding Generation**: Generate embeddings for the dataset using the specified encoder, processing data in batches.
4. **Embedding Merging**: Merge batch embeddings into a single file for efficient storage and access.
5. **Subset Selection**: Partition data into folds and select subsets using submodular optimization.
6. **Saving Subsets**: Save the selected subsets and their metadata for further use.

### Encoders

The repository includes support for different encoders:

- **UnifiedBGEEncoder**: A custom encoder for BGE models.
- **OpenAIEncoder**: Uses OpenAI's embedding models via the OpenAI API.
- **SentenceEncoder**: Leverages models from `sentence-transformers`.
- **NVEmbedEncoder**: Custom encoder for NVIDIA's `nvidia/NV-Embed-v2` model.
- **GTEQwen2InstructEncoder**: Custom encoder for Alibaba's `gte-Qwen2-7B-instruct` model.
- **SFRMistralEncoder**: Encoder for SFR Mistral models.

Each encoder class inherits from a `BaseEncoder` and implements the `encode` method, ensuring a consistent interface.

### Memory-Efficient Similarity Computation

Computing pairwise similarities between large sets of embeddings can be memory-intensive. To address this, the code includes:

- **Batch Processing**: Computes similarities in batches to manage memory usage.
- **Sparse Representations**: Uses sparse matrices when appropriate to reduce memory footprint.
- **Optimized Algorithms**: Employs efficient libraries like FAISS for similarity search.
- **Scaling Techniques**: Applies scaling methods to normalize similarity scores.
- **Compression-Based Distances**: Implements compression-based distance metrics for alternative similarity computations.

**Key Modules:**

- `compute_pairwise_similarity.py`: Contains functions for computing pairwise similarities.
- `compression_distance.py`: Calculates distances based on data compression techniques.
- `similarity_kernel_numpy.py` and `similarity_kernel_torch.py`: Provide additional methods for similarity computations using NumPy and PyTorch.

## Examples

### Processing a Single File

```bash
python replay_buffer_selection.py --input_files data/conversations.jsonl --output_dir output --config config.json
```

### Processing Multiple Files

```bash
python replay_buffer_selection.py --input_files data/dataset1.jsonl data/dataset2.jsonl --output_dir output --config config.json --num_gpus 2
```

### Subset Selection for New Use Cases

Suppose you have a dataset of multi-turn conversations and want to select subsets that represent the diversity of dialogues.

**Configuration Example:**

```json
{
  "instruction": "Generate embeddings that encapsulate the nuances of multi-turn conversations for effective clustering.",
  "query_description": "Multi-turn Conversation",
  "templates": {
    "conversation": "{% for turn in conversation %}{{ turn.speaker }}: {{ turn.text }}\n{% endfor %}"
  },
  "batch_size": 50000,
  "num_folds": 5,
  "subset_sizes": [50, 25, 10],
  "seed": 42
}
```

**Usage:**

```bash
python replay_buffer_selection.py --input_files data/multi_turn_conversations.jsonl --output_dir output --config conversation_config.json
```

### Using Compression-Based Similarity

To use compression-based distances in subset selection:

1. Modify the similarity computation in `compute_pairwise_similarity.py` to use `compression_distance.py`.
2. Adjust the encoder or preprocessing to ensure data is appropriately formatted for compression.

**Note:** This feature is under development and will be available in upcoming releases.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and test thoroughly.
4. Submit a pull request with a detailed description of your changes.


**Additional Information:**

- **Environment Variables:** If using the `OpenAIEncoder`, make sure to set the `OPENAI_API_KEY` in your environment or in a `.env` file.
- **Logging and Monitoring:** The code uses Python's `logging` module to provide detailed logs. Adjust the logging level as needed.
- **Error Handling:** The code includes retry mechanisms for robustness against transient errors, especially when dealing with API calls or large computations.

## Acknowledgments

- **Submodlib**: For providing submodular optimization functions.
- **FAISS**: For efficient similarity search algorithms.
- **Hugging Face**: For the `datasets` and `transformers` libraries.
- **OpenAI**: For the embedding models and API support.
- **NVIDIA**: For the NV-Embed-v2 model.
- **Alibaba**: For the GTE-Qwen2-7B-Instruct model.
