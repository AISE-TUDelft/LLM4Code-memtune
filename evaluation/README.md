# Evaluation Directory Documentation

This directory contains code, data, and results for data extraction experiments. The evaluation is organized into three main areas: forgetting attacks, memorization attacks, and data inspection.

## Directory Structure

```
evaluation/
├── forgetting/
│   ├── experiments/
│   └── results/
├── memorization/
│   ├── experiments/
│   └── results/
└── data-inspection/
    └── data/
```

## Forgetting Analysis
Pre-training code attack experiments on base and fine-tuned models (RQ1-2).

- **experiments/**: Contains data extraction experiment code and extracted code datasets
- **results/**: Contains generated tables and plots of experimental results

## Memorization Analysis
Fine-tuning code attack experiments on fine-tuned models (RQ3).

- **experiments/**: Contains data extraction experiment code and extracted code datasets  
- **results/**: Contains generated tables and plots of experimental results

## Data Inspection
Code data inspection for memorized code samples (RQ4).

### Files
- **data/**: Contains tagged dataset files
- **manual_tag.ipynb**: Jupyter notebook for:
  - Dataset preparation for manual labeling
  - Inter-rater agreement statistics
- **rq4.ipynb**: Jupyter notebook containing tables and analysis for Research Question 4

The evaluation directory provides a comprehensive framework for analyzing model behavior through data extraction attacks and detailed code inspection.