<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../figures/header-light.svg">
  <img alt="Aftab-Publikation" src="../figures/header-light.svg">
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

## Überblick

**Aftab** (<a href="https://en.wikipedia.org/wiki/Aftab">Persisch</a>: آفتاب, „Sonne“ oder „Sonnenstrahlen“) ist ein Benchmarking-Framework zur Bewertung CNN-basierter Encoder in PQN über verschiedene <a href="https://en.wikipedia.org/wiki/Atari_Games">Atari-Spiele</a> hinweg. Es stellt standardisierte Werkzeuge für Training, Evaluation und Reproduzierbarkeit in der Forschung zum tiefen bestärkenden Lernen bereit.

Wir haben einige Videos zusammengestellt, die PQN- und Aftab-Agenten vergleichen. Sie sind [hier](../videos.md) verfügbar.

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

Zum Erstellen virtueller Umgebungen empfehlen wir ausdrücklich [Micromamba](https://github.com/mamba-org/micromamba-releases). Eine ausführliche Anleitung ist [hier](../scripts/README.md) verfügbar.

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
- [Nach menschlicher Leistung normalisierte Ergebnisse](../results/encoder_experiments/human_normalized_scores.md)
- [Ergebnisse](../results/encoder_experiments/scores.md)

**Diagramme**
- [IQM HNS](../figures/encoder_experiments/human_normalized_score)
- [Verlauf des Loss](../figures/encoder_experiments/loss)

---

### Hadamax-Experimente

**Tabellen**
- [Nach menschlicher Leistung normalisierte Ergebnisse](../results/hadamax_experiments/human_normalized_scores.md)
- [Ergebnisse](../results/hadamax_experiments/scores.md)

**Diagramme**
- [IQM HNS](../figures/hadamax_experiments/human_normalized_score)
- [Verlauf des Loss](../figures/hadamax_experiments/loss)

---

### Q-Wert-Experimente

**Tabellen**
- [Nach menschlicher Leistung normalisierte Ergebnisse](../results/qvalue_experiments/human_normalized_scores.md)
- [Ergebnisse](../results/qvalue_experiments/scores.md)

**Diagramme**
- [IQM HNS](../figures/qvalue_experiments/human_normalized_score)
- [Verlauf des Loss](../figures/qvalue_experiments/loss)

---

### Procgen-Experimente

**Tabellen**
- [Normalisierte Procgen-Ergebnisse](../results/procgen_experiments/procgen_normalized_scores.md)
- [Ergebnisse](../results/procgen_experiments/scores.md)


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

<div align="center">

| Hyperparameter | Wert |
| :--- | :--- |
| Lernrate | $2.5 \times 10^{-4}$ |
| Trainingsumgebungen | 128 |
| Testumgebungen | 8 |
| Optimierer | [Rectified Adam](https://arxiv.org/abs/1908.03265) |
| Gewichtszerfall | 0 |
| $\epsilon$ | $1 \times 10^{-5}$ |
| $\beta_{1}$ | 0.9 |
| $\beta_{2}$ | 0.999 |
| Frames insgesamt | 200,000,000 |
| Verlustfunktion | Mittlerer quadratischer Fehler |
| Scheduler | Lineares Annealing |
| $\epsilon$-greedy-Exploration | 10% of total frames |
| Diskontierungsfaktor ($\gamma$) | 0.99 |
| GAE ($\lambda$) | 0.65 |
| Epochen | 2 |
| Batchgröße | 4096 |

</div>

<p align="center"><em>Wird in den Encoder- und Hadamax-Experimenten verwendet.</em></p>

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

```bibtex
@article{aftab2026drl,
  title={Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
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

- [Wikipedia: Bestärkendes Lernen (RL)](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [Wikipedia: Tiefes bestärkendes Lernen (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Wikipedia: Q-Lernen](https://en.wikipedia.org/wiki/Q-learning)
- [Wikipedia: PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [Wikipedia: Statistischer Hypothesentest](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [Wikipedia: Wilcoxon-Vorzeichen-Rang-Test](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## Lizenz

© 2025 Taha Shieenavaz.
Lizenziert unter CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
