# Ask a Strong LLM Judge when Your Reward Model is Uncertain

**Zhenghao Xu, Qin Lu, Qingru Zhang, Liang Qiu, Ilgee Hong, Changlong Yu, Wenlin Yao, Yao Liu, Haoming Jiang, Lihong Li, Hyokun Yun, Tuo Zhao**

**NeurIPS 2025**

Implementation of uncertainty-based reward routing for Reinforcement Learning from Human Feedback (RLHF). The system uses uncertainty quantification to dynamically route preference judgments between a local reward model and external generative judges.

## Overview

This work introduces a router that estimates uncertainty in reward model predictions and routes uncertain cases to more capable generative judges (e.g., LLM-as-a-judge). This combines the efficiency of learned reward models with the quality of large language models.

### Key Features

- **Uncertainty Quantification Methods**
  - Spectral Normalization
  - Gaussian Process output layer
  - SNGP (Spectral Normalization + GP)
  - Monte Carlo Dropout

- **Uncertainty-Based Routing**
  - Routes high-uncertainty predictions to generative judges
  - Configurable uncertainty threshold
  - Random routing baseline for comparison

- **Built on OpenRLHF**
  - Compatible with PPO and RLOO training
  - Preference model training with uncertainty

## Installation

```bash
git clone https://github.com/zhenghaoxu-gatech/uncertainty-router.git
cd uncertainty-router
pip install -r requirements.txt
```

Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

## Project Structure

```
uncertainty-router/
├── ppo/openrlhf/
│   ├── models/
│   │   ├── spec_norm.py        # Spectral Normalization
│   │   └── model.py            # Preference models with uncertainty
│   ├── cli/
│   │   └── serve_pm_router.py  # Router server implementation
│   └── trainer/                # Training code
├── ppo/src/eval_pm_router/     # Evaluation scripts
└── scripts/                     # Training configurations
```

## Core Components

### 1. Uncertainty Estimation

- **Spectral Normalization** (`spec_norm.py`): Constrains weight matrix spectral norms
- **Gaussian Process Layer** (`model.py`): Random Fourier features for GP
- **MC Dropout** (`model.py`): Dropout at inference for uncertainty
- **SNGP**: Combines spectral normalization + GP layer

### 2. Router (`ppo/openrlhf/cli/serve_pm_router.py`)

The `PreferenceModelProxy` implements:
- Pairwise preference prediction with uncertainty
- Threshold-based routing to generative judge (AWS Bedrock)
- Multi-GPU inference via DDP

Key parameters:
- `--use_sn`: Enable spectral normalization
- `--sn_range`: Spectral norm bound
- `--use_gp`: Enable Gaussian process layer
- `--gp_amplitude`: GP output scale
- `--use_mcd`: Enable MC Dropout
- `--mcd_p`: Dropout rate
- `--threshold`: Uncertainty threshold for routing
- `--router`: Routing strategy (`uncertainty` or `random`)

### 3. Training

Train reward model with uncertainty:
```bash
python -m openrlhf.cli.train_rm \
    --pretrain meta-llama/Llama-3.1-8B \
    --dataset Anthropic/hh-rlhf \
    --use_sn \
    --sn_range 10.0 \
    --use_gp \
    --gp_amplitude 0.1 \
    ...
```

### 4. Evaluation

```bash
# RewardBench evaluation
python ppo/src/eval_pm_router/eval_rewardbench.py \
    --model_path /path/to/model \
    --use_sn --use_gp

# RMBench evaluation
python ppo/src/eval_pm_router/eval_rmbench.py \
    --model_path /path/to/model \
    --use_sn --use_gp
```

## Citation

```bibtex
@inproceedings{xu2025ask,
  title={Ask a Strong LLM Judge when Your Reward Model is Uncertain},
  author={Xu, Zhenghao and Lu, Qin and Zhang, Qingru and Qiu, Liang and Hong, Ilgee and Yu, Changlong and Yao, Wenlin and Liu, Yao and Jiang, Haoming and Li, Lihong and Yun, Hyokun and Zhao, Tuo},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## Acknowledgements

This work builds upon and evaluates against several important projects:

- **[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)**: Our implementation is built on the OpenRLHF framework for efficient RLHF training.
- **[RewardBench](https://github.com/allenai/reward-bench)**: We use RewardBench for evaluating reward model performance across diverse scenarios.
- **[RM-Bench](https://github.com/THU-KEG/RM-Bench)**: We evaluate on RM-Bench for comprehensive reward model assessment.

