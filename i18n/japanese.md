<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../figures/header-light.svg">
  <img alt="Aftab 論文" src="../figures/header-light.svg">
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
  <img src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b" />
</p>

<div align="center">
  <a href="https://underdash.pro">Taha Shieenavaz</a> | <a href="https://shbnmzr.github.io">Shabnam Zareshahraki</a> | <a href="https://scholar.google.com/citations?user=5NSGzcQAAAAJ&hl=en">Loris Nanni</a>
</div>

<br />

<div align="center">
  🇪🇸🇲🇽🇨🇺 <a href="./spanish.md">Español</a> |
  🇮🇷🇦🇫🇹🇯 <a href="./farsi.md">فارسی</a> |
  🇮🇹🇨🇭 <a href="./italian.md">Italiano</a> |
  🇫🇷🇧🇪🇨🇭 <a href="./french.md">Français</a> |
  🇩🇪🇦🇹🇨🇭 <a href="./german.md">Deutsch</a> |
  🇳🇱🇧🇪🇸🇷 <a href="./dutch.md">Nederlands</a> |
  🇵🇹🇧🇷🇦🇴 <a href="./portuguese.md">Português</a> |
  🇸🇦🇱🇧🇮🇶 <a href="./arabic.md">العربية</a> |
  🇷🇺🇧🇾🇰🇿 <a href="./russian.md">Русский</a> |
  🇨🇳🇸🇬🇹🇼 <a href="./chinese.md">中文</a> |
  🇯🇵 <a href="./japanese.md">日本語</a> |
  🇰🇷 <a href="./korean.md">한국어</a> |
  🇮🇳 <a href="./hindi.md">हिन्दी</a> |
  🇮🇩 <a href="./indonesian.md">Bahasa Indonesia</a> |
  🇧🇩🇮🇳 <a href="./bengali.md">বাংলা</a> |
  🇻🇳 <a href="./vietnamese.md">Tiếng Việt</a> |
  🇹🇷 <a href="./turkish.md">Türkçe</a>
</div>

## 概要

**Aftab**（<a href="https://en.wikipedia.org/wiki/Aftab">ペルシア語</a>：آفتاب、「太陽」または「陽光」の意）は、複数の <a href="https://en.wikipedia.org/wiki/Atari_Games">Atari ゲーム</a>において PQN の CNN ベースのエンコーダを評価するためのベンチマークフレームワークです。深層強化学習研究向けに、標準化された学習・評価・再現性確保のためのツールを提供します。

PQN と Aftab のエージェントを比較する動画をいくつか用意しました。[こちら](../videos.md)からご覧いただけます。

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

仮想環境の作成には [Micromamba](https://github.com/mamba-org/micromamba-releases) を強く推奨します。詳しい手順は[こちら](../scripts/README.md)を参照してください。

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
- [人間の成績で正規化したスコア](../results/encoder_experiments/human_normalized_scores.md)
- [スコア](../results/encoder_experiments/scores.md)

**グラフ**
- [IQM HNS](../figures/encoder_experiments/human_normalized_score)
- [損失の推移](../figures/encoder_experiments/loss)

---

### Hadamax 実験

**表**
- [人間の成績で正規化したスコア](../results/hadamax_experiments/human_normalized_scores.md)
- [スコア](../results/hadamax_experiments/scores.md)

**グラフ**
- [IQM HNS](../figures/hadamax_experiments/human_normalized_score)
- [損失の推移](../figures/hadamax_experiments/loss)

---

### Q 値実験

**表**
- [人間の成績で正規化したスコア](../results/qvalue_experiments/human_normalized_scores.md)
- [スコア](../results/qvalue_experiments/scores.md)

**グラフ**
- [IQM HNS](../figures/qvalue_experiments/human_normalized_score)
- [損失の推移](../figures/qvalue_experiments/loss)

---

### Procgen 実験

**表**
- [Procgen 正規化スコア](../results/procgen_experiments/procgen_normalized_scores.md)
- [スコア](../results/procgen_experiments/scores.md)


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

<div align="center">

| ハイパーパラメータ | 値 |
| :--- | :--- |
| 学習率 | $2.5 \times 10^{-4}$ |
| 学習環境数 | 128 |
| テスト環境数 | 8 |
| オプティマイザ | [Rectified Adam](https://arxiv.org/abs/1908.03265) |
| 重み減衰 | 0 |
| $\epsilon$ | $1 \times 10^{-5}$ |
| $\beta_{1}$ | 0.9 |
| $\beta_{2}$ | 0.999 |
| 総フレーム数 | 200,000,000 |
| 損失関数 | 平均二乗誤差 |
| スケジューラ | 線形アニーリング |
| $\epsilon$-greedy 探索 | 10% of total frames |
| 割引率（$\gamma$） | 0.99 |
| GAE ($\lambda$) | 0.65 |
| エポック数 | 2 |
| バッチサイズ | 4096 |

</div>

<p align="center"><em>エンコーダ実験および Hadamax 実験で使用しています。</em></p>

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

```bibtex
@article{aftab2026drl,
  title={Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
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

- [Wikipedia：強化学習（RL）](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [Wikipedia：深層強化学習（DRL）](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Wikipedia：Q 学習](https://en.wikipedia.org/wiki/Q-learning)
- [Wikipedia：PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [Wikipedia：統計的仮説検定](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [Wikipedia：Wilcoxon 符号順位検定](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## ライセンス

© 2025 Taha Shieenavaz.
CC BY-NC 4.0 ライセンスに基づいて提供されています： https://creativecommons.org/licenses/by-nc/4.0/
