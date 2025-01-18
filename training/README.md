# Fine-tuning

- This folder contains the training scripts used for fine-tuning StarCoder2. Additionally we disclose the training stats.

- The fine-tuning dataset can be retrieved at this link: https://huggingface.co/datasets/AISE-TUDelft/memtune-tuning_data

- The fine-tuned models can be retrieved at this link: https://huggingface.co/collections/AISE-TUDelft/llm4code-memtune-678a2838766dd16037a8bfe0

## Fine-tuning Setup

### Hardware Configuration
- 32 CPU cores
- 32GB RAM
- Multiple NVIDIA A100 GPUs (80GB memory each)
  - StarCoder2-3B: 2 GPUs
  - StarCoder2-7B: 4 GPUs
  - StarCoder2-15B: 6 GPUs

### Software Stack
- NVIDIA Driver: 555.42.02
- CUDA Version: 12.5
- Transformer Version: 4.41.1
- Torch Version: 2.3.0+cu121

### Training Configuration
- Context Window: 1024 tokens
- Learning Rate: 3e-5
- Optimizer: Adafactor with linear scheduler
- Batch Sizes (effective, including gradient accumulation):
  - 3B model: 24
  - 7B model: 24
  - 15B model: 25

### Training Duration
Approximate training times per model:
- StarCoder2-3B: 25 hours
- StarCoder2-7B: 55 hours
- StarCoder2-15B: 110 hours

## Training Process
- Training duration: 3 epochs
- Checkpoints saved after each epoch
- GPU memory and training time were key factors in determining:
  - Optimizer selection
  - Training file configuration
  - Batch size parameters

Training was conducted using resources provided by the [Delft High-Performance Computing Centre](https://doc.dhpc.tudelft.nl/delftblue/).

## Training stats

### StarCoder2-3B

**Evaluation loss**:
![](/training/train-stats/StarCoder2-3B/eval-loss.png)

**Training loss**:
![](/training/train-stats/StarCoder2-3B/train-loss.png)

**Learning rate**:
![](/training/train-stats/StarCoder2-3B/train-learning_rate.png)

### StarCoder2-7B
**Evaluation loss**:
![](/training/train-stats/StarCoder2-7B/eval-loss.png)

**Training loss**:
![](/training/train-stats/StarCoder2-7B/train-loss.png)

**Learning rate**:
![](/training/train-stats/StarCoder2-7B/train-learning_rate.png)

### StarCoder2-15B

**Evaluation loss**:
![](/training/train-stats/StarCoder2-15B/eval-loss.png)

**Training loss**:
![](/training/train-stats/StarCoder2-15B/train-loss.png)

**Learning rate**:
![](/training/train-stats/StarCoder2-15B/train-learning_rate.png)