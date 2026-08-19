<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-dark.svg">
  <img alt="Aftab Header" src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-light.svg">
</picture>

<p align="center">
  <img src="https://img.shields.io/pypi/v/aftab" />
  <img src="https://img.shields.io/github/stars/tahashieenavaz/aftab?style=social" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" />
  <img src="https://img.shields.io/badge/license-CC--BY--NC--4.0-lightgrey" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/backend-PyTorch-ee4c2c" />
  <img src="https://img.shields.io/badge/citable-yes-success" />
  <a href="https://arxiv.org/abs/2608.07335">
    <img src="https://img.shields.io/badge/arXiv-2608.07335-b31b1b" />
  </a>
</p>

<br />

<div align="center">
  <a href="https://underdash.pro">Taha Shieenavaz</a> | <a href="https://shbnmzr.github.io">Shabnam Zareshahraki</a> | <a href="https://scholar.google.com/citations?user=5NSGzcQAAAAJ&hl=en">Loris Nanni</a>
</div>

<br />

<div align="center">
  🇪🇸🇲🇽🇨🇺 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/spanish.md">Español</a> |
  🇮🇷🇦🇫🇹🇯 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/farsi.md">فارسی</a> |
  🇮🇹🇨🇭 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/italian.md">Italiano</a> |
  🇫🇷🇧🇪🇨🇭 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/french.md">Français</a> |
  🇩🇪🇦🇹🇨🇭 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/german.md">Deutsch</a> |
  🇳🇱🇧🇪🇸🇷 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/dutch.md">Nederlands</a> |
  🇵🇹🇧🇷🇦🇴 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/portuguese.md">Português</a> |
  🇸🇦🇱🇧🇮🇶 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/arabic.md">العربية</a> |
  🇷🇺🇧🇾🇰🇿 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/russian.md">Русский</a> |
  🇨🇳🇸🇬🇹🇼 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/chinese.md">中文</a> |
  🇯🇵 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/japanese.md">日本語</a> |
  🇰🇷 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/korean.md">한국어</a> |
  🇮🇳 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/hindi.md">हिन्दी</a> |
  🇮🇩 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/indonesian.md">Bahasa Indonesia</a> |
  🇧🇩🇮🇳 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/bengali.md">বাংলা</a> |
  🇻🇳 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/vietnamese.md">Tiếng Việt</a> |
  🇹🇷 <a href="https://github.com/tahashieenavaz/aftab/blob/main/i18n/turkish.md">Türkçe</a>
</div>

## Overview

**Aftab** (<a href="https://en.wikipedia.org/wiki/Aftab">Persian</a>: آفتاب, meaning "sun" or "sun rays") is a benchmarking framework for evaluating CNN-based encoders in PQN across <a href="https://en.wikipedia.org/wiki/Atari_Games">Atari games</a>. It provides standardized training, evaluation, and reproducibility tools for deep reinforcement learning research.

See how the Aftab architecture compares to standard PQN baselines in these [video demonstrations](https://github.com/tahashieenavaz/aftab/blob/main/videos.md).

This research was done without receiving any funds; therefore, if you found our work useful, please consider [sponsoring on GitHub](https://github.com/sponsors/tahashieenavaz) 💛. 

### Encoder Experiments

<div align="center">
  <table>
    <tr>
      <th>IQM HNS</th>
    </tr>
    <tr>
      <td>
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/encoder_experiments/global_dark.png" />
          <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/encoder_experiments/global_light.png" />
        </picture>    
      </td>
    </tr>
    <tr>
      <th>IQM HNS (Last 50M Frames)</th>
    </tr>
    <tr>
      <td>
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/encoder_experiments/global_zoomed_dark.png" />
          <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/encoder_experiments/global_zoomed_light.png" />
        </picture>
      </td>
    </tr>
  </table>
</div>

### Hadamax Experiments

<div align="center">
  <table>
    <tr>
      <th>IQM HNS</th>
    </tr>
    <tr>
      <td>
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/hadamax_experiments/global_dark.png" />
          <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/hadamax_experiments/global_light.png" />
        </picture>    
      </td>
    </tr>
    <tr>
      <th>IQM HNS (Last 50M Frames)</th>
    </tr>
    <tr>
      <td>
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/hadamax_experiments/global_zoomed_dark.png" />
          <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/hadamax_experiments/global_zoomed_light.png" />
        </picture>
      </td>
    </tr>
  </table>
</div>

References:
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### Q-Values Experiments

<div align="center">
  <table>
    <tr>
      <th>IQM HNS</th>
    </tr>
    <tr>
      <td>
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/qvalue_experiments/global_dark.png" />
          <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/qvalue_experiments/global_light.png" />
        </picture>    
      </td>
    </tr>
    <tr>
      <th>IQM HNS (Last 50M Frames)</th>
    </tr>
    <tr>
      <td>
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/qvalue_experiments/global_zoomed_dark.png" />
          <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/qvalue_experiments/global_zoomed_light.png" />
        </picture>
      </td>
    </tr>
  </table>
</div>

References:
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

### Procgen (Overfitting Prevention) Experiments

Since there are no public benchmarks comparing human-normalized-scores of Procgen environments, we created PNS (Procgen Normalized Score) that is a minimal min-max normalization of scores across seeds. 

<div align="center">
  <table>
    <tr>
      <th>IQM PNS</th>
    </tr>
    <tr>
      <td>
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/procgen_experiments/global_dark.png" />
          <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/procgen_experiments/global_light.png" />
        </picture>    
      </td>
    </tr>
    <tr>
      <th>IQM PNS (Last 50M Frames)</th>
    </tr>
    <tr>
      <td>
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/procgen_experiments/global_zoomed_dark.png" />
          <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/procgen_experiments/global_zoomed_light.png" />
        </picture>
      </td>
    </tr>
  </table>
</div>

## Installation

Install via pip:

```bash
pip install aftab
```

Alternatively, you can clone the repository and install in `editable` mode. 

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

We highly recommend using [Micromamba](https://github.com/mamba-org/micromamba-releases) for creating virtual environments with instructions detailed [here](https://github.com/tahashieenavaz/aftab/blob/main/scripts/README.md).

## Training Agents

**Currently JAX API is under development** and is planned to be finished by the end of 2026. Contributions are highly encouraged.

```python
from aftab import Aftab
from aftab import aftab_environments

seeds = [1, 2, 3, 4]

for environment in aftab_environments:
    agent = Aftab(encoder="gamma", frames="pilot")
    for seed in seeds:
        agent.train(environment=environment, seed=seed)
        agent.log()
```


## Custom Encoder Injection

You can define your own encoder as a PyTorch module and pass it to the agent:

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## Results

All experimental results are organized by experiment category. Each section contains:
- **Tables**: numerical results (HNS/PHS and raw scores)
- **Charts**: IQM normalized scores and training curves

### Encoder Experiments

**Tables**
- [Human Normalized Scores](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/human_normalized_scores.md)
- [Scores](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/scores.md)

**Charts**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/human_normalized_score)
- [Loss Evolution](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/loss)

---

### Hadamax Experiments

**Tables**
- [Human Normalized Scores](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/human_normalized_scores.md)
- [Scores](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/scores.md)

**Charts**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/human_normalized_score)
- [Loss Evolution](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/loss)

---

### Q-Value Experiments

**Tables**
- [Human Normalized Scores](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/human_normalized_scores.md)
- [Scores](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/scores.md)

**Charts**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/human_normalized_score)
- [Loss Evolution](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/loss)

---

### Procgen Experiments

**Tables**
- [Procgen Normalized Scores](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/procgen_normalized_scores.md)
- [Scores](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/scores.md)
- [PNS AUC by Seed](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_seed.md)
- [PNS AUC by Game](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_game.md)


**Charts**
- [IQM PNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/procgen_experiments/procgen_normalized_score)

## Model Complexity

### Base Variants

| Variant | Encoder Parameters | Regression Head Parameters | Total Parameters | Encoder FLOPs | Regression Head FLOPs | Total FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PQN** | 78,304 | 1,686,500 | 1,764,804 | 7.734 | 1.610 | 9.347 |
| **Alpha** | 174,752 | 1,782,948 | 1,957,700 | 27.541 | 1.610 | 29.151 |
| **Beta** | 89,008 | 1,782,948 | 1,871,956 | 61.515 | 1.610 | 63.126 |
| **Gamma** | 117,168 | 1,725,364 | 1,842,532 | 22.901 | 1.610 | 24.512 |
| **Delta** | 78,552 | 1,850,588 | 1,929,140 | 6.143 | 1.774 | 7.917 |
| **Epsilon** | 80,112 | 2,179,828 | 2,259,940 | 13.252 | 2.101 | 15.354 |
| **Zeta** | 77,232 | 2,537,396 | 2,614,628 | 25.362 | 2.462 | 27.824 |
| **Eta** | 78,400 | 23,739,460 | 23,817,860 | 28.422 | 23.663 | 52.085 |
| **Theta** | 76,288 | 1,127,428 | 1,203,716 | 9.065 | 1.053 | 10.118 |

> **Note:** The Eta variant has significantly more parameters than other variants, primarily due to the encoder producing a large number of features.

---

### Hadamax Variants

| Variant | Encoder Parameters | Regression Head Parameters | Total Parameters | Encoder FLOPs | Regression Head FLOPs | Total FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## Hyperparameters

The following tables reflect the defaults defined by `Aftab`. The
`experiment_name` argument is required and has no default.

### Training and Environment

| Hyperparameter (`Aftab` argument) | Default |
| :--- | :--- |
| Encoder (`encoder`) | Gamma-Hadamax-Valid  |
| Network (`network`) | Distributional Bootstrapped (Ensemble) Dueling |
| Total frames (`frames`) | 200,000,000 |
| Frame skip (`frame_skip`) | 4 |
| Frame stack (`frame_stack`) | 4 |
| No-op maximum (`noop`) | 30 |
| Learning rate (`lr`) | $2.5 \times 10^{-4}$ |
| Training environments (`train_environments`) | 128 |
| Test environments (`test_environments`) | 8 |
| Steps per update (`steps_per_update`) | 32 |
| Batch size (derived) | 4,096 |
| Mini-batches (`mini_batches`) | 32 |
| Mini-batch size (derived) | 128 |
| Discount factor ($\gamma$) | 0.99 |
| Return $\lambda$ (`return_lambda`) | 0.65 |
| Epochs (`epochs`) | 2 |
| Gradient norm (`gradient_norm`) | 10.0 |
| Embedding dimension (`embedding_dimension`) | 512 |
| Training episodic life (`train_episodic_life`) | `True` |
| Test episodic life (`test_episodic_life`) | `False` |
| Training reward clipping (`train_reward_clip`) | `True` |
| Test reward clipping (`test_reward_clip`) | `True` |
| Epsilon Schedule | Linear |
| Epsilon Annealing Ratio | 10% |

### Optimizer

| Hyperparameter (`Aftab` argument) | Default |
| :--- | :--- |
| Optimizer (`optimizer`) | [Rectified Adam](https://arxiv.org/abs/1908.03265) (`"radam"`) |
| Epsilon (`optimizer_epsilon`) | $1 \times 10^{-5}$ |
| Weight decay (`optimizer_weight_decay`) | 0.0 |
| $\beta_1$ (`optimizer_first_beta`) | 0.9 |
| $\beta_2$ (`optimizer_second_beta`) | 0.999 |

### Distributional and Bootstrapped (Ensemble) Q-Values

| Hyperparameter (`Aftab` argument) | Default |
| :--- | :--- |
| Distributional bins (`distributional_bins`) | 51 |
| Distributional minimum (`distributional_min_value`) | -10.0 |
| Distributional maximum (`distributional_max_value`) | 10.0 |
| Distributional sigma (`distributional_sigma`) | `None` (derived from the sigma ratio) |
| Distributional sigma ratio (`distributional_sigma_ratio`) | 0.75 |
| Distributional value clip (`distributional_value_clip`) | 0.0 |
| Bootstrap heads (`bootstrap_heads`) | 10 |
| Bootstrap probability (`bootstrap_probability`) | 1.0 |

### Procgen Overrides

| Hyperparameter        | Default | Procgen                           |
| :-------------------- | :------ | :-------------------------------- |
| Training environments | 128     | 64 (`procgen_train_environments`) |
| Steps per update      | 32      | 256 (`procgen_steps_per_update`)  |
| Batch size            | 4,096   | 16,384                            |
| Mini-batch size       | 128     | 512                               |

<em>For Procgen environments, Aftab automatically applies the two overrides above; other defaults remain unchanged.</em>

## Statistical Significance

### Encoder Experiments

<table>
  <tr>
    <th align="center">Wilcoxon Signed Rank Test</th>
    <th align="center">Wilcoxon Signed Rank Test (Corrected)</th>
  </tr>
  <tr>
    <td align="center">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/encoder_experiments/wilcoxon_p_matrix_dark.png" />
        <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/encoder_experiments/wilcoxon_p_matrix_light.png" />
      </picture>
    </td>
    <td align="center">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/encoder_experiments/wilcoxon_bonferroni_matrix_dark.png" />
        <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/encoder_experiments/wilcoxon_bonferroni_matrix_light.png" />
      </picture>
    </td>
  </tr>
  <tr>
    <th colspan="2" align="center">Probability of Improvement</th>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/encoder_experiments/probability_of_improvement_dark.png" />
        <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/encoder_experiments/probability_of_improvement_light.png" />
      </picture>
    </td>
  </tr>
</table>

### Hadamax Experiments

<table>
  <tr>
    <th align="center">Wilcoxon Signed Rank Test</th>
    <th align="center">Wilcoxon Signed Rank Test (Corrected)</th>
  </tr>
  <tr>
    <td align="center">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/hadamax_experiments/wilcoxon_p_matrix_dark.png" />
        <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/hadamax_experiments/wilcoxon_p_matrix_light.png" />
      </picture>
    </td>
    <td align="center">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/hadamax_experiments/wilcoxon_bonferroni_matrix_dark.png" />
        <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/hadamax_experiments/wilcoxon_bonferroni_matrix_light.png" />
      </picture>
    </td>
  </tr>
  <tr>
    <th colspan="2" align="center">Probability of Improvement</th>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/hadamax_experiments/probability_of_improvement_dark.png" />
        <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/hadamax_experiments/probability_of_improvement_light.png" />
      </picture>
    </td>
  </tr>
</table>

### Q-Value Experiments

<table>
  <tr>
    <th align="center">Wilcoxon Signed Rank Test</th>
    <th align="center">Wilcoxon Signed Rank Test (Corrected)</th>
  </tr>
  <tr>
    <td align="center">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/qvalue_experiments/wilcoxon_p_matrix_dark.png" />
        <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/qvalue_experiments/wilcoxon_p_matrix_light.png" />
      </picture>
    </td>
    <td align="center">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/qvalue_experiments/wilcoxon_bonferroni_matrix_dark.png" />
        <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/qvalue_experiments/wilcoxon_bonferroni_matrix_light.png" />
      </picture>
    </td>
  </tr>
  <tr>
    <th colspan="2" align="center">Probability of Improvement</th>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/qvalue_experiments/probability_of_improvement_dark.png" />
        <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/qvalue_experiments/probability_of_improvement_light.png" />
      </picture>
    </td>
  </tr>
</table>

## Reproducibility

Due to the stochastic nature of deep reinforcement learning, exact reproducibility via fixed datasets is not feasible.  
Instead, we provide a set of random seeds used in our experiments.

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

Full experiment replication:

```python
from aftab import Aftab
from aftab import aftab_environments
from aftab import aftab_seeds

for environment in aftab_environments:
    agent = Aftab()
    for seed in aftab_seeds:
        agent.train(environment=environment, seed=seed)
        agent.log()
```

A comprehensive set of Atari environments is available via EnvPool:  
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

Procgen environments use their native RGB observations with shape `(3, 64, 64)`.
Aftab reads each task's EnvPool configuration and only applies supported options.
Atari-only options such as `noop`, `frame_skip`, `frame_stack`,
`train_episodic_life`, and EnvPool reward clipping are therefore not passed to
Procgen.

A comprehensive set of Procgen environments is available via EnvPool:

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## Hardware

[Nvidia A40](https://www.nvidia.com/en-us/data-center/a40) GPUs were used to run all the experiments in this experiment.

| Specification | Details |
|--------------|----------|
| GPU Memory | 48 GB GDDR6 with error-correcting code (ECC) |
| GPU Memory Bandwidth | 696 GB/s |
| Interconnect | NVIDIA NVLink 112.5 GB/s (bidirectional); PCIe Gen4: 64 GB/s |
| NVLink | 2-way low profile (2-slot) |
| Display Ports | 3x DisplayPort 1.4* |
| Max Power Consumption | 300 W |
| Form Factor | 4.4" (H) x 10.5" (L), Dual Slot |
| Thermal | Passive |
| vGPU Software Support | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| vGPU Profiles Supported | See the Virtual GPU Licensing Guide |
| NVENC / NVDEC | 1x / 2x (includes AV1 decode) |
| Secure Boot | Secure and Measured Boot with Hardware Root of Trust (optional) |
| NEBS Ready | Level 3 |
| Power Connector | 8-pin CPU |

## Citation

Repository:

```bibtex
@software{aftab2026,
  author = {Taha Shieenavaz},
  title = {Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/tahashieenavaz/aftab}},
}
```

Preprint: 

```bibtex
@misc{2608.07335,
  Author = {Taha Shieenavaz and Shabnam Zareshahraki and Loris Nanni},
  Title = {Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  Year = {2026},
  Eprint = {arXiv:2608.07335},
}
```

### Related Works

```bibtex
@misc{2407.04811,
  Title = {Simplifying Deep Temporal Difference Learning},
  Author = {Matteo Gallici and Mattie Fellows and Benjamin Ellis and Bartomeu Pou and Ivan Masmitja and Jakob Nicolaus Foerster and Mario Martin},
  Year = {2024},
  Eprint = {arXiv:2407.04811},
}
```

```bibtex
@misc{2403.03950,
  Title = {Stop Regressing: Training Value Functions via Classification for Scalable Deep RL},
  Author = {Jesse Farebrother and Jordi Orbay and Quan Vuong and Adrien Ali Taïga and Yevgen Chebotar and Ted Xiao and Alex Irpan and Sergey Levine and Pablo Samuel Castro and Aleksandra Faust and Aviral Kumar and Rishabh Agarwal},
  Year = {2024},
  Eprint = {arXiv:2403.03950},
}
```

```bibtex
@misc{1511.06581,
  Title = {Dueling Network Architectures for Deep Reinforcement Learning},
  Author = {Ziyu Wang and Tom Schaul and Matteo Hessel and Hado van Hasselt and Marc Lanctot and Nando de Freitas},
  Year = {2015},
  Eprint = {arXiv:1511.06581},
}
```

```bibtex
@misc{1806.04613,
  Title = {Improving Regression Performance with Distributional Losses},
  Author = {Ehsan Imani and Martha White},
  Year = {2018},
  Eprint = {arXiv:1806.04613},
}
```

```bibtex
@misc{1602.04621,
  Title = {Deep Exploration via Bootstrapped DQN},
  Author = {Ian Osband and Charles Blundell and Alexander Pritzel and Benjamin Van Roy},
  Year = {2016},
  Eprint = {arXiv:1602.04621},
}
```

## Useful Links

- [Wikipedia: Reinforcement Learning (RL)](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [Wikipedia: Deep Reinforcement Learning (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Wikipedia: Q-Learning](https://en.wikipedia.org/wiki/Q-learning)
- [Wikipedia: PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [Wikipedia: Statistical Hypothesis Test](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [Wikipedia: Wilcoxon Signed-Rank Test](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## Font

The Vazirmatn font is used for both Persian and English text throughout the GitHub repository header and the project's landing page. 

[GitHub](https://github.com/rastikerdar/vazirmatn) | [Google Fonts](https://fonts.google.com/specimen/Vazirmatn)

## Translations

Translations were generated by AI using [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol) @ Wednesday August 19th, ~14:00.

## License

© 2025 Taha Shieenavaz.  
Licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
