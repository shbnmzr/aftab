<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-dark.svg">
  <img alt="Aftab-Header" src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-light.svg">
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

## Überblick

**Aftab** (<a href="https://en.wikipedia.org/wiki/Aftab">Persisch</a>: آفتاب, „Sonne“ oder „Sonnenstrahlen“) ist ein Benchmarking-Framework zur Bewertung CNN-basierter Encoder in PQN über verschiedene <a href="https://de.wikipedia.org/wiki/Atari_Games">Atari-Spiele</a> hinweg. Es stellt standardisierte Werkzeuge für Training, Evaluation und Reproduzierbarkeit in der Forschung zum tiefen bestärkenden Lernen bereit.

Sehen Sie in diesen [Videodemonstrationen](https://github.com/tahashieenavaz/aftab/blob/main/videos.md), wie die Aftab-Architektur im Vergleich zu den Standard-PQN-Baselines abschneidet.

Diese Forschung wurde ohne finanzielle Förderung durchgeführt. Wenn unsere Arbeit für Sie nützlich war, können Sie uns daher gern [auf GitHub unterstützen](https://github.com/sponsors/tahashieenavaz) 💛.

### Encoder-Experimente

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
      <th>IQM HNS (letzte 50 Millionen Frames)</th>
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

### Hadamax-Experimente

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
      <th>IQM HNS (letzte 50 Millionen Frames)</th>
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

Referenzen:
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### Q-Wert-Experimente

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
      <th>IQM HNS (letzte 50 Millionen Frames)</th>
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

Referenzen:
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

### Procgen-Experimente (Vermeidung von Overfitting)

Da es keine öffentlichen Benchmarks zum Vergleich menschlich normalisierter Ergebnisse in Procgen-Umgebungen gibt, haben wir PNS (Procgen Normalized Score) entwickelt, eine einfache Min-Max-Normalisierung der Ergebnisse über verschiedene Seeds hinweg.

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
      <th>IQM PNS (letzte 50 Millionen Frames)</th>
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

Installation über pip:

```bash
pip install aftab
```

Alternativ kann das Repository geklont und im `editable`-Modus installiert werden.

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

Zum Erstellen virtueller Umgebungen empfehlen wir ausdrücklich [Micromamba](https://github.com/mamba-org/micromamba-releases). Eine ausführliche Anleitung ist [hier](https://github.com/tahashieenavaz/aftab/blob/main/scripts/README.md) verfügbar.

## Agenten trainieren

**Die JAX-API befindet sich derzeit in Entwicklung** und soll bis Ende 2026 fertiggestellt werden. Beiträge sind ausdrücklich willkommen.

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


## Benutzerdefinierten Encoder einbinden

Ein eigener Encoder kann als PyTorch-Modul definiert und an den Agenten übergeben werden:

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## Ergebnisse

Alle Versuchsergebnisse sind nach Experimentkategorie gegliedert. Jeder Abschnitt enthält:
- **Tabellen**: numerische Ergebnisse (HNS/PHS und Rohwerte)
- **Diagramme**: IQM-normalisierte Ergebnisse und Trainingskurven

### Encoder-Experimente

**Tabellen**
- [Nach menschlicher Leistung normalisierte Ergebnisse](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/human_normalized_scores.md)
- [Ergebnisse](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/scores.md)

**Diagramme**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/human_normalized_score)
- [Verlauf des Loss](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/loss)

---

### Hadamax-Experimente

**Tabellen**
- [Nach menschlicher Leistung normalisierte Ergebnisse](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/human_normalized_scores.md)
- [Ergebnisse](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/scores.md)

**Diagramme**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/human_normalized_score)
- [Verlauf des Loss](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/loss)

---

### Q-Wert-Experimente

**Tabellen**
- [Nach menschlicher Leistung normalisierte Ergebnisse](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/human_normalized_scores.md)
- [Ergebnisse](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/scores.md)

**Diagramme**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/human_normalized_score)
- [Verlauf des Loss](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/loss)

---

### Procgen-Experimente

**Tabellen**
- [Normalisierte Procgen-Ergebnisse](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/procgen_normalized_scores.md)
- [Ergebnisse](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/scores.md)
- [PNS-AUC nach Seed](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_seed.md)
- [PNS-AUC nach Spiel](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_game.md)

**Diagramme**
- [IQM PNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/procgen_experiments/procgen_normalized_score)

## Modellkomplexität

### Basisvarianten

| Variante | Encoder-Parameter | Parameter des Regressionskopfs | Gesamtparameter | Encoder-FLOPs | FLOPs des Regressionskopfs | Gesamt-FLOPs |
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

> **Hinweis:** Die Eta-Variante besitzt deutlich mehr Parameter als die übrigen Varianten. Hauptgrund dafür ist die große Anzahl an Merkmalen, die ihr Encoder erzeugt.

---

### Hadamax-Varianten

| Variante | Encoder-Parameter | Parameter des Regressionskopfs | Gesamtparameter | Encoder-FLOPs | FLOPs des Regressionskopfs | Gesamt-FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## Hyperparameter

Die folgenden Tabellen geben die von `Aftab` definierten Standardwerte wieder. Das Argument `experiment_name` ist erforderlich und hat keinen Standardwert.

### Training und Umgebung

| Hyperparameter (`Aftab`-Argument) | Standardwert |
| :--- | :--- |
| Encoder (`encoder`) | Gamma-Hadamax-Valid |
| Netzwerk (`network`) | Distributional Bootstrapped (Ensemble) Dueling |
| Frames insgesamt (`frames`) | 200,000,000 |
| Frame-Skip (`frame_skip`) | 4 |
| Frame-Stack (`frame_stack`) | 4 |
| Maximale No-op-Anzahl (`noop`) | 30 |
| Lernrate (`lr`) | $2.5 \times 10^{-4}$ |
| Trainingsumgebungen (`train_environments`) | 128 |
| Testumgebungen (`test_environments`) | 8 |
| Schritte pro Aktualisierung (`steps_per_update`) | 32 |
| Batchgröße (abgeleitet) | 4,096 |
| Mini-Batches (`mini_batches`) | 32 |
| Mini-Batch-Größe (abgeleitet) | 128 |
| Diskontierungsfaktor ($\gamma$) | 0.99 |
| Return-$\lambda$ (`return_lambda`) | 0.65 |
| Epochen (`epochs`) | 2 |
| Gradientennorm (`gradient_norm`) | 10.0 |
| Embedding-Dimension (`embedding_dimension`) | 512 |
| Episodisches Leben im Training (`train_episodic_life`) | `True` |
| Episodisches Leben im Test (`test_episodic_life`) | `False` |
| Belohnungs-Clipping im Training (`train_reward_clip`) | `True` |
| Belohnungs-Clipping im Test (`test_reward_clip`) | `True` |
| Epsilon-Zeitplan | Linear |
| Epsilon-Annealing-Verhältnis | 10% |

### Optimierer

| Hyperparameter (`Aftab`-Argument) | Standardwert |
| :--- | :--- |
| Optimierer (`optimizer`) | [Rectified Adam](https://arxiv.org/abs/1908.03265) (`"radam"`) |
| Epsilon (`optimizer_epsilon`) | $1 \times 10^{-5}$ |
| Gewichtszerfall (`optimizer_weight_decay`) | 0.0 |
| $\beta_1$ (`optimizer_first_beta`) | 0.9 |
| $\beta_2$ (`optimizer_second_beta`) | 0.999 |

### Distributionale und Bootstrapped-(Ensemble-)Q-Werte

| Hyperparameter (`Aftab`-Argument) | Standardwert |
| :--- | :--- |
| Distributionale Bins (`distributional_bins`) | 51 |
| Distributionales Minimum (`distributional_min_value`) | -10.0 |
| Distributionales Maximum (`distributional_max_value`) | 10.0 |
| Distributionales Sigma (`distributional_sigma`) | `None` (aus dem Sigma-Verhältnis abgeleitet) |
| Distributionales Sigma-Verhältnis (`distributional_sigma_ratio`) | 0.75 |
| Distributionales Werte-Clipping (`distributional_value_clip`) | 0.0 |
| Bootstrap-Köpfe (`bootstrap_heads`) | 10 |
| Bootstrap-Wahrscheinlichkeit (`bootstrap_probability`) | 1.0 |

### Procgen-Überschreibungen

| Hyperparameter | Standardwert | Procgen |
| :--- | :--- | :--- |
| Trainingsumgebungen | 128 | 64 (`procgen_train_environments`) |
| Schritte pro Aktualisierung | 32 | 256 (`procgen_steps_per_update`) |
| Batchgröße | 4,096 | 16,384 |
| Mini-Batch-Größe | 128 | 512 |

<em>Für Procgen-Umgebungen wendet Aftab die beiden obigen Überschreibungen automatisch an; alle anderen Standardwerte bleiben unverändert.</em>

## Statistische Signifikanz

### Encoder-Experimente

<table>
  <tr>
    <th align="center">Wilcoxon-Vorzeichen-Rang-Test</th>
    <th align="center">Wilcoxon-Vorzeichen-Rang-Test (korrigiert)</th>
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
    <th colspan="2" align="center">Verbesserungswahrscheinlichkeit</th>
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

### Hadamax-Experimente

<table>
  <tr>
    <th align="center">Wilcoxon-Vorzeichen-Rang-Test</th>
    <th align="center">Wilcoxon-Vorzeichen-Rang-Test (korrigiert)</th>
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
    <th colspan="2" align="center">Verbesserungswahrscheinlichkeit</th>
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

### Q-Wert-Experimente

<table>
  <tr>
    <th align="center">Wilcoxon-Vorzeichen-Rang-Test</th>
    <th align="center">Wilcoxon-Vorzeichen-Rang-Test (korrigiert)</th>
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
    <th colspan="2" align="center">Verbesserungswahrscheinlichkeit</th>
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

## Reproduzierbarkeit

Aufgrund der stochastischen Natur des tiefen bestärkenden Lernens lassen sich die Ergebnisse mit festen Datensätzen nicht exakt reproduzieren.
Stattdessen stellen wir die in unseren Experimenten verwendeten Zufalls-Seeds bereit.

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

Vollständige Reproduktion der Experimente:

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

EnvPool stellt eine umfassende Auswahl an Atari-Umgebungen bereit:
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

Procgen-Umgebungen verwenden ihre nativen RGB-Beobachtungen mit der Form `(3, 64, 64)`.
Aftab liest die EnvPool-Konfiguration jeder Aufgabe und wendet ausschließlich unterstützte Optionen an.
Atari-spezifische Optionen wie `noop`, `frame_skip`, `frame_stack` und
`train_episodic_life` sowie das Reward-Clipping von EnvPool werden daher nicht an
Procgen übergeben.

EnvPool stellt eine umfassende Auswahl an Procgen-Umgebungen bereit:

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## Hardware

Alle Experimente dieses Projekts wurden auf [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40)-GPUs ausgeführt.

| Spezifikation | Details |
|--------------|----------|
| GPU-Speicher | 48 GB GDDR6 mit Fehlerkorrekturcode (ECC) |
| GPU-Speicherbandbreite | 696 GB/s |
| Interconnect | NVIDIA NVLink 112,5 GB/s (bidirektional); PCIe Gen4: 64 GB/s |
| NVLink | Bidirektional, Low Profile (2 Slots) |
| Display-Anschlüsse | 3x DisplayPort 1.4* |
| Maximale Leistungsaufnahme | 300 W |
| Abmessungen | 4,4" (H) × 10,5" (L), Dual-Slot |
| Kühlung | Passiv |
| Unterstützte vGPU-Software | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| Unterstützte vGPU-Profile | Siehe Lizenzierungsleitfaden für virtuelle GPUs |
| NVENC / NVDEC | 1x / 2x (einschließlich AV1-Decodierung) |
| Sicherer Start | Sicherer und gemessener Start mit Hardware-Vertrauensanker (optional) |
| NEBS-konform | Stufe 3 |
| Stromanschluss | 8-poliger CPU-Anschluss |

## Zitieren

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

### Verwandte Arbeiten

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

## Nützliche Links

- [Wikipedia: Bestärkendes Lernen (RL)](https://de.wikipedia.org/wiki/Bestärkendes_Lernen)
- [Wikipedia: Tiefes bestärkendes Lernen (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Wikipedia: Q-Lernen](https://de.wikipedia.org/wiki/Q-Lernen)
- [Wikipedia: PyTorch](https://de.wikipedia.org/wiki/PyTorch)
- [Wikipedia: Statistischer Hypothesentest](https://de.wikipedia.org/wiki/Statistischer_Test)
- [Wikipedia: Wilcoxon-Vorzeichen-Rang-Test](https://de.wikipedia.org/wiki/Wilcoxon-Vorzeichen-Rang-Test)
- [PyTorch](https://pytorch.org/)

## Schriftart

Die Schriftart Vazirmatn wird für persische und englische Texte sowohl im Header des GitHub-Repositorys als auch auf der Startseite des Projekts verwendet.

[GitHub](https://github.com/rastikerdar/vazirmatn) | [Google Fonts](https://fonts.google.com/specimen/Vazirmatn)

## Lizenz

© 2025 Taha Shieenavaz.
Lizenziert unter CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
