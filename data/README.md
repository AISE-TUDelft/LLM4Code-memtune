# Data Directory Documentation

This repository contains scripts and tools for dataset filtering and sample creation, organized into two main directories.

## Directory Structure

```
data/
├── filtering/
└── samples-creation/
```

## Filtering Directory

Contains the script for dataset filtering and sampling to create the fine-tuning dataset.

### Datasets
- **Fine-tuning Dataset**: Available on HuggingFace at [AISE-TUDelft/memtune-tuning_data](https://huggingface.co/datasets/AISE-TUDelft/memtune-tuning_data)
- **Source Dataset**: The original dataset that underwent filtering is available at [AISE-TUDelft/the-heap](https://huggingface.co/datasets/AISE-TUDelft/the-heap)

## Sample Creation Directory

Contains scripts for generating data extraction benchmarks used in data extraction attacks.

### Scripts
- `sample_identification_mem.py`: Generates benchmark dataset for the **fine-tuning code attack**
- `sample_identification_forget.py`: Generates benchmark dataset for the **pre-training code attack**
  - Note: Requires downloading the [TheStackV2 Java subset](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup/viewer/Java)

### Benchmarks
The generated data extraction benchmarks are available at [AISE-TUDelft/memtune-data_attack](https://huggingface.co/datasets/AISE-TUDelft/memtune-data_attack)