<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-dark.svg">
  <img alt="Aftab 页眉" src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-light.svg">
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

<br />

## 概述

**Aftab**（<a href="https://en.wikipedia.org/wiki/Aftab">波斯语</a>：آفتاب，意为“太阳”或“阳光”）是一个基准测试框架，用于评估 PQN 在多款 <a href="https://zh.wikipedia.org/wiki/雅达利游戏">Atari 游戏</a>中采用的 CNN 编码器。它为深度强化学习研究提供标准化的训练、评估与复现工具。

通过这些[视频演示](https://github.com/tahashieenavaz/aftab/blob/main/videos.md)，了解 Aftab 架构与标准 PQN 基线的对比。

本研究未获得任何资金支持；如果我们的工作对你有帮助，请考虑[在 GitHub 上赞助](https://github.com/sponsors/tahashieenavaz) 💛。

### 编码器实验

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
      <th>IQM HNS（最后 5000 万帧）</th>
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

### Hadamax 实验

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
      <th>IQM HNS（最后 5000 万帧）</th>
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

参考文献：
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### Q 值实验

<div align="center">

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
      <th>IQM HNS（最后 5000 万帧）</th>
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

</div>

参考文献：
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

### Procgen（防止过拟合）实验

由于没有公开基准对 Procgen 环境的人类归一化分数进行比较，我们创建了 PNS（Procgen Normalized Score），即对不同随机种子的分数进行简单的最小—最大归一化。

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
      <th>IQM PNS (最后 5000 万帧)</th>
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

## 安装

使用 pip 安装：

```bash
pip install aftab
```

也可以克隆仓库，并以 `editable` 模式安装。

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

我们强烈建议使用 [Micromamba](https://github.com/mamba-org/micromamba-releases) 创建虚拟环境，详细说明见[此处](https://github.com/tahashieenavaz/aftab/blob/main/scripts/README.md)。

## 训练智能体

**JAX API 目前仍在开发中**，计划于 2026 年底前完成。非常欢迎社区贡献。

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


## 注入自定义编码器

你可以将自定义编码器定义为 PyTorch 模块，并传递给智能体：

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## 实验结果

所有实验结果均按实验类别整理。每一节包含：
- **表格**: 数值结果（HNS/PHS 和原始得分）
- **图表**: IQM 归一化得分与训练曲线

### 编码器实验

**表格**
- [人类归一化得分](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/human_normalized_scores.md)
- [原始得分](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/scores.md)

**图表**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/human_normalized_score)
- [损失变化曲线](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/loss)

---

### Hadamax 实验

**表格**
- [人类归一化得分](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/human_normalized_scores.md)
- [原始得分](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/scores.md)

**图表**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/human_normalized_score)
- [损失变化曲线](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/loss)

---

### Q 值实验

**表格**
- [人类归一化得分](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/human_normalized_scores.md)
- [原始得分](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/scores.md)

**图表**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/human_normalized_score)
- [损失变化曲线](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/loss)

---

### Procgen 实验

**表格**
- [Procgen 归一化得分](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/procgen_normalized_scores.md)
- [原始得分](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/scores.md)
- [按随机种子统计的 PNS AUC](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_seed.md)
- [按游戏统计的 PNS AUC](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_game.md)

**图表**
- [IQM PNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/procgen_experiments/procgen_normalized_score)

## 模型复杂度

### 基础变体

| 变体 | 编码器参数量 | 回归头参数量 | 总参数量 | 编码器 FLOPs | 回归头 FLOPs | 总 FLOPs |
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

> **注意：** Eta 变体的参数量显著高于其他变体，主要原因是其编码器会生成大量特征。

---

### Hadamax 变体

| 变体 | 编码器参数量 | 回归头参数量 | 总参数量 | 编码器 FLOPs | 回归头 FLOPs | 总 FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## 超参数

下表列出了 `Aftab` 定义的默认值。`experiment_name` 参数为必填项，没有默认值。

### 训练与环境

| 超参数（`Aftab` 参数） | 默认值 |
| :--- | :--- |
| 编码器（`encoder`） | Gamma-Hadamax-Valid |
| 网络（`network`） | 分布式自助法（集成）Dueling 网络 |
| 总帧数（`frames`） | 200,000,000 |
| 跳帧（`frame_skip`） | 4 |
| 帧堆叠（`frame_stack`） | 4 |
| 最大无操作次数（`noop`） | 30 |
| 学习率（`lr`） | $2.5 \times 10^{-4}$ |
| 训练环境数（`train_environments`） | 128 |
| 测试环境数（`test_environments`） | 8 |
| 每次更新步数（`steps_per_update`） | 32 |
| 批大小（推导值） | 4,096 |
| 小批次数（`mini_batches`） | 32 |
| 小批大小（推导值） | 128 |
| 折扣因子（$\gamma$） | 0.99 |
| 回报 $\lambda$（`return_lambda`） | 0.65 |
| 训练轮数（`epochs`） | 2 |
| 梯度范数（`gradient_norm`） | 10.0 |
| 嵌入维度（`embedding_dimension`） | 512 |
| 训练时回合生命（`train_episodic_life`） | `True` |
| 测试时回合生命（`test_episodic_life`） | `False` |
| 训练奖励裁剪（`train_reward_clip`） | `True` |
| 测试奖励裁剪（`test_reward_clip`） | `True` |
| Epsilon 调度 | 线性 |
| Epsilon 退火比例 | 10% |

### 优化器

| 超参数（`Aftab` 参数） | 默认值 |
| :--- | :--- |
| 优化器（`optimizer`） | [Rectified Adam](https://arxiv.org/abs/1908.03265)（`"radam"`） |
| Epsilon（`optimizer_epsilon`） | $1 \times 10^{-5}$ |
| 权重衰减（`optimizer_weight_decay`） | 0.0 |
| $\beta_1$（`optimizer_first_beta`） | 0.9 |
| $\beta_2$（`optimizer_second_beta`） | 0.999 |

### 分布式与自助法（集成）Q 值

| 超参数（`Aftab` 参数） | 默认值 |
| :--- | :--- |
| 分布区间数（`distributional_bins`） | 51 |
| 分布最小值（`distributional_min_value`） | -10.0 |
| 分布最大值（`distributional_max_value`） | 10.0 |
| 分布 Sigma（`distributional_sigma`） | `None`（由 Sigma 比例推导） |
| 分布 Sigma 比例（`distributional_sigma_ratio`） | 0.75 |
| 分布值裁剪（`distributional_value_clip`） | 0.0 |
| Bootstrap 头数（`bootstrap_heads`） | 10 |
| Bootstrap 概率（`bootstrap_probability`） | 1.0 |

### Procgen 覆盖值

| 超参数 | 默认值 | Procgen |
| :--- | :--- | :--- |
| 训练环境数 | 128 | 64（`procgen_train_environments`） |
| 每次更新步数 | 32 | 256（`procgen_steps_per_update`） |
| 批大小 | 4,096 | 16,384 |
| 小批大小 | 128 | 512 |

<em>对于 Procgen 环境，Aftab 会自动应用上述两项覆盖值；其他默认值保持不变。</em>

## 统计显著性

### 编码器实验

<table>
  <tr>
    <th align="center">Wilcoxon 符号秩检验</th>
    <th align="center">Wilcoxon 符号秩检验（校正后）</th>
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
    <th colspan="2" align="center">改进概率</th>
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

### Hadamax 实验

<table>
  <tr>
    <th align="center">Wilcoxon 符号秩检验</th>
    <th align="center">Wilcoxon 符号秩检验（校正后）</th>
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
    <th colspan="2" align="center">改进概率</th>
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

### Q 值实验

<table>
  <tr>
    <th align="center">Wilcoxon 符号秩检验</th>
    <th align="center">Wilcoxon 符号秩检验（校正后）</th>
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
    <th colspan="2" align="center">改进概率</th>
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

## 可复现性

由于深度强化学习具有随机性，仅依靠固定数据集无法做到完全复现。
因此，我们提供实验中使用的一组随机种子。

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

完整复现实验：

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

EnvPool 提供了完整的 Atari 环境集合：
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

Procgen 环境使用其原生 RGB 观测，形状为 `(3, 64, 64)`。
Aftab 会读取每个任务的 EnvPool 配置，并且只应用该任务支持的选项。
因此，`noop`、`frame_skip`、`frame_stack`、`train_episodic_life` 等仅适用于 Atari 的选项，
以及 EnvPool 的奖励裁剪，都不会传递给 Procgen。

EnvPool 提供了完整的 Procgen 环境集合：

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## 硬件

本项目的全部实验均使用 [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40) GPU 运行。

| 规格 | 详细信息 |
|--------------|----------|
| GPU 显存 | 48 GB GDDR6，支持纠错码（ECC） |
| GPU 显存带宽 | 696 GB/s |
| 互连 | NVIDIA NVLink 112.5 GB/s（双向）；PCIe Gen4：64 GB/s |
| NVLink | 双向低矮型（双槽） |
| 显示接口 | 3x DisplayPort 1.4* |
| 最大功耗 | 300 W |
| 外形尺寸 | 4.4 英寸（高）× 10.5 英寸（长），双槽 |
| 散热方式 | 被动散热 |
| 支持的 vGPU 软件 | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| 支持的 vGPU 配置文件 | 请参阅《虚拟 GPU 许可指南》 |
| NVENC / NVDEC | 1x / 2x（支持 AV1 解码） |
| 安全启动 | 基于硬件信任根的安全启动与度量启动（可选） |
| NEBS 认证 | 3 级 |
| 电源接口 | 8 针 CPU |

## 引用

代码仓库：

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

预印本：

```bibtex
@misc{2608.07335,
  Author = {Taha Shieenavaz and Shabnam Zareshahraki and Loris Nanni},
  Title = {Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  Year = {2026},
  Eprint = {arXiv:2608.07335},
}
```

### 相关工作

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

## 实用链接

- [维基百科：强化学习（RL）](https://zh.wikipedia.org/wiki/强化学习)
- [维基百科：深度强化学习（DRL）](https://zh.wikipedia.org/wiki/深度强化学习)
- [维基百科：Q 学习](https://zh.wikipedia.org/wiki/Q学习)
- [维基百科：PyTorch](https://zh.wikipedia.org/wiki/PyTorch)
- [维基百科：统计假设检验](https://zh.wikipedia.org/wiki/假說檢定)
- [维基百科：Wilcoxon 符号秩检验](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## 字体

GitHub 代码仓库页眉和项目着陆页中的波斯语与英语文本均使用 Vazirmatn 字体。

[GitHub](https://github.com/rastikerdar/vazirmatn) | [Google Fonts](https://fonts.google.com/specimen/Vazirmatn)

## 许可证

© 2025 Taha Shieenavaz.
本项目采用 CC BY-NC 4.0 许可证： https://creativecommons.org/licenses/by-nc/4.0/
