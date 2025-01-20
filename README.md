# LLM4Code-memtune

Replication package for the paper: "**How Much Do Code Language Models Remember? An Investigation on Data Extraction Attacks before and after Fine-tuning**"

For questions:
- Repository content: Please use the issues board
- Paper inquiries: Contact the first author via email (info DOT fabiosalern AT gmail DOT COM)

## Repository Structure

```
LLM4Code-memtune/
├── data/               # Dataset filtering and sample creation tools
├── training/          # StarCoder2 fine-tuning scripts and training stats
└── evaluation/        # Data extraction experiment code and results
```

## Requirements

### Hardware Requirements
- GPU: Nvidia A100 (80GB VRAM)
- RAM: 32GB
- CPU: 16 cores

GPU requirements by model:
- StarCoder2-3B: 2 GPUs
- StarCoder2-7B: 4 GPUs
- StarCoder2-15B: 6 GPUs

Note: Data extraction experiments can run on a single GPU.

### Software Requirements
- Python 3.8
- Additional dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Directories

### Data
Contains scripts and tools for dataset filtering and sample creation, organized into two main directories.

### Training
Contains:
- Fine-tuning scripts for StarCoder2
- Training statistics and metrics

### Evaluation
Contains code, data, and results for data extraction experiments.

For detailed documentation of each directory, please refer to their respective README files.

## Ethical use
Please use the code and concepts shared here responsibly and ethically. The authors have provided this code to enhance the security and safety of large language models (LLMs). Avoid using this code for any malicious purposes. When disclosing data leakage, take care not to compromise individuals' privacy unnecessarily.

