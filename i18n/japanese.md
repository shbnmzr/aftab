<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-dark.svg">
  <img alt="Aftab ヘッダー" src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-light.svg">
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

## 概要

**Aftab**（<a href="https://en.wikipedia.org/wiki/Aftab">ペルシア語</a>：آفتاب、「太陽」または「陽光」の意）は、複数の <a href="https://en.wikipedia.org/wiki/Atari_Games">Atari ゲーム</a>において PQN の CNN ベースのエンコーダを評価するためのベンチマークフレームワークです。深層強化学習研究向けに、標準化された学習・評価・再現性確保のためのツールを提供します。

これらの[動画デモ](https://github.com/tahashieenavaz/aftab/blob/main/videos.md)で、Aftab アーキテクチャと標準的な PQN ベースラインの比較をご覧ください。

本研究は資金提供を受けずに実施されました。私たちの成果が役立った場合は、[GitHub でのスポンサー](https://github.com/sponsors/tahashieenavaz)をご検討ください 💛。

### エンコーダ実験

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
      <th>IQM HNS（最後の 5,000 万フレーム）</th>
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

### Hadamax 実験

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
      <th>IQM HNS（最後の 5,000 万フレーム）</th>
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

### Q 値実験

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
      <th>IQM HNS（最後の 5,000 万フレーム）</th>
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

参考文献：
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

### Procgen 実験（過学習の抑制）

Procgen 環境の人間正規化スコアを比較する公開ベンチマークがないため、seed 間のスコアを単純に min-max 正規化する PNS（Procgen Normalized Score）を作成しました。

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
      <th>IQM PNS（最後の 5,000 万フレーム）</th>
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

## インストール

pip でインストールします：

```bash
pip install aftab
```

または、リポジトリをクローンして `editable` モードでインストールできます。

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

仮想環境の作成には [Micromamba](https://github.com/mamba-org/micromamba-releases) を強く推奨します。詳しい手順は[こちら](https://github.com/tahashieenavaz/aftab/blob/main/scripts/README.md)を参照してください。

## エージェントの学習

**JAX API は現在開発中であり**、2026 年末までの完成を予定しています。コントリビューションを歓迎します。

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


## カスタムエンコーダの組み込み

独自のエンコーダを PyTorch モジュールとして定義し、エージェントに渡すことができます：

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## 結果

すべての実験結果は実験カテゴリ別に整理されています。各セクションには次の内容が含まれます：
- **表**：数値結果（HNS/PHS および生スコア）
- **グラフ**：IQM 正規化スコアと学習曲線

### エンコーダ実験

**表**
- [人間の成績で正規化したスコア](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/human_normalized_scores.md)
- [スコア](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/scores.md)

**グラフ**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/human_normalized_score)
- [損失の推移](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/loss)

---

### Hadamax 実験

**表**
- [人間の成績で正規化したスコア](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/human_normalized_scores.md)
- [スコア](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/scores.md)

**グラフ**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/human_normalized_score)
- [損失の推移](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/loss)

---

### Q 値実験

**表**
- [人間の成績で正規化したスコア](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/human_normalized_scores.md)
- [スコア](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/scores.md)

**グラフ**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/human_normalized_score)
- [損失の推移](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/loss)

---

### Procgen 実験

**表**
- [Procgen 正規化スコア](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/procgen_normalized_scores.md)
- [スコア](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/scores.md)
- [seed 別 PNS AUC](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_seed.md)
- [ゲーム別 PNS AUC](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_game.md)

**グラフ**
- [IQM PNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/procgen_experiments/procgen_normalized_score)

## モデルの複雑度

### 基本バリアント

| バリアント | エンコーダのパラメータ数 | 回帰ヘッドのパラメータ数 | 総パラメータ数 | エンコーダの FLOPs | 回帰ヘッドの FLOPs | 総 FLOPs |
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

> **注：** Eta バリアントは、ほかのバリアントよりもパラメータ数が大幅に多くなっています。主な理由は、エンコーダが多数の特徴量を生成するためです。

---

### Hadamax バリアント

| バリアント | エンコーダのパラメータ数 | 回帰ヘッドのパラメータ数 | 総パラメータ数 | エンコーダの FLOPs | 回帰ヘッドの FLOPs | 総 FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## ハイパーパラメータ

以下の表は `Aftab` で定義されたデフォルト値を示します。`experiment_name` 引数は必須で、デフォルト値はありません。

### 学習と環境

| ハイパーパラメータ（`Aftab` 引数） | デフォルト値 |
| :--- | :--- |
| エンコーダ（`encoder`） | Gamma-Hadamax-Valid |
| ネットワーク（`network`） | Distributional Bootstrapped（Ensemble）Dueling |
| 総フレーム数（`frames`） | 200,000,000 |
| フレームスキップ（`frame_skip`） | 4 |
| フレームスタック（`frame_stack`） | 4 |
| 最大 no-op 数（`noop`） | 30 |
| 学習率（`lr`） | $2.5 \times 10^{-4}$ |
| 学習環境数（`train_environments`） | 128 |
| テスト環境数（`test_environments`） | 8 |
| 更新あたりのステップ数（`steps_per_update`） | 32 |
| バッチサイズ（導出値） | 4,096 |
| ミニバッチ数（`mini_batches`） | 32 |
| ミニバッチサイズ（導出値） | 128 |
| 割引率（$\gamma$） | 0.99 |
| リターン $\lambda$（`return_lambda`） | 0.65 |
| エポック数（`epochs`） | 2 |
| 勾配ノルム（`gradient_norm`） | 10.0 |
| 埋め込み次元（`embedding_dimension`） | 512 |
| 学習時のエピソードライフ（`train_episodic_life`） | `True` |
| テスト時のエピソードライフ（`test_episodic_life`） | `False` |
| 学習時の報酬クリッピング（`train_reward_clip`） | `True` |
| テスト時の報酬クリッピング（`test_reward_clip`） | `True` |
| Epsilon スケジュール | 線形 |
| Epsilon アニーリング比率 | 10% |

### オプティマイザ

| ハイパーパラメータ（`Aftab` 引数） | デフォルト値 |
| :--- | :--- |
| オプティマイザ（`optimizer`） | [Rectified Adam](https://arxiv.org/abs/1908.03265)（`"radam"`） |
| Epsilon（`optimizer_epsilon`） | $1 \times 10^{-5}$ |
| 重み減衰（`optimizer_weight_decay`） | 0.0 |
| $\beta_1$（`optimizer_first_beta`） | 0.9 |
| $\beta_2$（`optimizer_second_beta`） | 0.999 |

### Distributional および Bootstrapped（Ensemble）Q 値

| ハイパーパラメータ（`Aftab` 引数） | デフォルト値 |
| :--- | :--- |
| 分布ビン数（`distributional_bins`） | 51 |
| 分布最小値（`distributional_min_value`） | -10.0 |
| 分布最大値（`distributional_max_value`） | 10.0 |
| 分布 Sigma（`distributional_sigma`） | `None`（Sigma 比率から導出） |
| 分布 Sigma 比率（`distributional_sigma_ratio`） | 0.75 |
| 分布値クリッピング（`distributional_value_clip`） | 0.0 |
| Bootstrap ヘッド数（`bootstrap_heads`） | 10 |
| Bootstrap 確率（`bootstrap_probability`） | 1.0 |

### Procgen のオーバーライド

| ハイパーパラメータ | デフォルト | Procgen |
| :--- | :--- | :--- |
| 学習環境数 | 128 | 64（`procgen_train_environments`） |
| 更新あたりのステップ数 | 32 | 256（`procgen_steps_per_update`） |
| バッチサイズ | 4,096 | 16,384 |
| ミニバッチサイズ | 128 | 512 |

<em>Procgen 環境では、Aftab が上記 2 つのオーバーライドを自動的に適用し、その他のデフォルト値は変更されません。</em>

## 統計的有意性

### エンコーダ実験

<table>
  <tr>
    <th align="center">Wilcoxon 符号順位検定</th>
    <th align="center">Wilcoxon 符号順位検定（補正後）</th>
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
    <th colspan="2" align="center">改善確率</th>
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

### Hadamax 実験

<table>
  <tr>
    <th align="center">Wilcoxon 符号順位検定</th>
    <th align="center">Wilcoxon 符号順位検定（補正後）</th>
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
    <th colspan="2" align="center">改善確率</th>
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

### Q 値実験

<table>
  <tr>
    <th align="center">Wilcoxon 符号順位検定</th>
    <th align="center">Wilcoxon 符号順位検定（補正後）</th>
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
    <th colspan="2" align="center">改善確率</th>
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

## 再現性

深層強化学習には確率的な性質があるため、固定データセットだけで結果を完全に再現することはできません。
その代わり、実験で使用した乱数シード一式を提供しています。

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

実験全体の再現：

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

EnvPool では多様な Atari 環境を利用できます：
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

Procgen 環境は、形状が `(3, 64, 64)` のネイティブ RGB 観測を使用します。
Aftab は各タスクの EnvPool 設定を読み取り、サポートされているオプションだけを適用します。
したがって、`noop`、`frame_skip`、`frame_stack`、`train_episodic_life` などの
Atari 専用オプションと EnvPool の報酬クリッピングは Procgen に渡されません。

EnvPool では多様な Procgen 環境を利用できます：

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## ハードウェア

本プロジェクトのすべての実験は [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40) GPU で実行しました。

| 仕様 | 詳細 |
|--------------|----------|
| GPU メモリ | エラー訂正符号（ECC）対応 48 GB GDDR6 |
| GPU メモリ帯域幅 | 696 GB/s |
| インターコネクト | NVIDIA NVLink 112.5 GB/s（双方向）、PCIe Gen4：64 GB/s |
| NVLink | 双方向ロープロファイル（2 スロット） |
| ディスプレイポート | 3x DisplayPort 1.4* |
| 最大消費電力 | 300 W |
| 外形寸法 | 4.4"（高さ）× 10.5"（長さ）、デュアルスロット |
| 冷却方式 | パッシブ |
| 対応 vGPU ソフトウェア | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| 対応 vGPU プロファイル | Virtual GPU ライセンスガイドを参照 |
| NVENC / NVDEC | 1x / 2x（AV1 デコードを含む） |
| セキュアブート | ハードウェアの信頼の基点を用いたセキュアブートおよびメジャードブート（オプション） |
| NEBS 対応 | レベル 3 |
| 電源コネクタ | 8 ピン CPU |

## 引用

リポジトリ：

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

プレプリント：

```bibtex
@misc{2608.07335,
  Author = {Taha Shieenavaz and Shabnam Zareshahraki and Loris Nanni},
  Title = {Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  Year = {2026},
  Eprint = {arXiv:2608.07335},
}
```

### 関連研究

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

## 参考リンク

- [Wikipedia：強化学習（RL）](https://ja.wikipedia.org/wiki/強化学習)
- [Wikipedia：深層強化学習（DRL）](https://ja.wikipedia.org/wiki/深層強化学習)
- [Wikipedia：Q 学習](https://ja.wikipedia.org/wiki/Q学習)
- [Wikipedia：PyTorch](https://ja.wikipedia.org/wiki/PyTorch)
- [Wikipedia：統計的仮説検定](https://ja.wikipedia.org/wiki/仮説検定)
- [Wikipedia：Wilcoxon 符号順位検定](https://ja.wikipedia.org/wiki/ウィルコクソンの符号順位検定)
- [PyTorch](https://pytorch.org/)

## フォント

GitHub リポジトリのヘッダーとプロジェクトのランディングページでは、ペルシア語と英語の両方に Vazirmatn フォントを使用しています。

[GitHub](https://github.com/rastikerdar/vazirmatn) | [Google Fonts](https://fonts.google.com/specimen/Vazirmatn)

## ライセンス

© 2025 Taha Shieenavaz.
CC BY-NC 4.0 ライセンスに基づいて提供されています： https://creativecommons.org/licenses/by-nc/4.0/
