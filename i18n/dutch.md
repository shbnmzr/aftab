<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../figures/header-light.svg">
  <img alt="Aftab-paper" src="../figures/header-light.svg">
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

## Overzicht

**Aftab** (<a href="https://en.wikipedia.org/wiki/Aftab">Perzisch</a>: آفتاب, met de betekenis ‘zon’ of ‘zonnestralen’) is een benchmarkframework voor het evalueren van CNN-gebaseerde encoders in PQN voor verschillende <a href="https://en.wikipedia.org/wiki/Atari_Games">Atari-games</a>. Het biedt gestandaardiseerde hulpmiddelen voor training, evaluatie en reproduceerbaarheid binnen onderzoek naar deep reinforcement learning.

We hebben enkele video’s samengesteld waarin PQN- en Aftab-agents worden vergeleken. Bekijk ze [hier](../videos.md).

### Encoderexperimenten

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
      <th>IQM HNS (laatste 50 miljoen frames)</th>
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

### Hadamax-experimenten

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
      <th>IQM HNS (laatste 50 miljoen frames)</th>
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

Referenties:
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### Experimenten met Q-waarden

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
      <th>IQM HNS (laatste 50 miljoen frames)</th>
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

Referenties:
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

### Procgen-experimenten (overfitting voorkomen)

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
      <th>IQM PNS (laatste 50 miljoen frames)</th>
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

## Installatie

Installeren met pip:

```bash
pip install aftab
```

Je kunt de repository ook klonen en in `editable`-modus installeren.

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

Voor het maken van virtuele omgevingen raden we [Micromamba](https://github.com/mamba-org/micromamba-releases) sterk aan. Uitgebreide instructies staan [hier](../scripts/README.md).

## Agents trainen

**De JAX-API is momenteel in ontwikkeling** en zal naar verwachting eind 2026 gereed zijn. Bijdragen zijn van harte welkom.

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


## Een aangepaste encoder toevoegen

Je kunt je eigen encoder als PyTorch-module definiëren en aan de agent doorgeven:

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## Resultaten

Alle experimentele resultaten zijn per experimentcategorie ingedeeld. Elke sectie bevat:
- **Tabellen**: numerieke resultaten (HNS/PHS en ruwe scores)
- **Grafieken**: met IQM genormaliseerde scores en trainingscurven

### Encoderexperimenten

**Tabellen**
- [Naar menselijke prestaties genormaliseerde scores](../results/encoder_experiments/human_normalized_scores.md)
- [Scores](../results/encoder_experiments/scores.md)

**Grafieken**
- [IQM HNS](../figures/encoder_experiments/human_normalized_score)
- [Verloop van de loss](../figures/encoder_experiments/loss)

---

### Hadamax-experimenten

**Tabellen**
- [Naar menselijke prestaties genormaliseerde scores](../results/hadamax_experiments/human_normalized_scores.md)
- [Scores](../results/hadamax_experiments/scores.md)

**Grafieken**
- [IQM HNS](../figures/hadamax_experiments/human_normalized_score)
- [Verloop van de loss](../figures/hadamax_experiments/loss)

---

### Experimenten met Q-waarden

**Tabellen**
- [Naar menselijke prestaties genormaliseerde scores](../results/qvalue_experiments/human_normalized_scores.md)
- [Scores](../results/qvalue_experiments/scores.md)

**Grafieken**
- [IQM HNS](../figures/qvalue_experiments/human_normalized_score)
- [Verloop van de loss](../figures/qvalue_experiments/loss)

---

### Procgen-experimenten

**Tabellen**
- [Genormaliseerde Procgen-scores](../results/procgen_experiments/procgen_normalized_scores.md)
- [Scores](../results/procgen_experiments/scores.md)


## Modelcomplexiteit

### Basisvarianten

| Variant | Encoderparameters | Parameters van de regressiekop | Totaal aantal parameters | Encoder-FLOPs | FLOPs van de regressiekop | Totale FLOPs |
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

> **Opmerking:** de Eta-variant heeft aanzienlijk meer parameters dan de andere varianten, vooral doordat de encoder een groot aantal kenmerken produceert.

---

### Hadamax-varianten

| Variant | Encoderparameters | Parameters van de regressiekop | Totaal aantal parameters | Encoder-FLOPs | FLOPs van de regressiekop | Totale FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## Hyperparameters

<div align="center">

| Hyperparameter | Waarde |
| :--- | :--- |
| Leersnelheid | $2.5 \times 10^{-4}$ |
| Trainingsomgevingen | 128 |
| Testomgevingen | 8 |
| Optimalisator | [Rectified Adam](https://arxiv.org/abs/1908.03265) |
| Gewichtsverval | 0 |
| $\epsilon$ | $1 \times 10^{-5}$ |
| $\beta_{1}$ | 0.9 |
| $\beta_{2}$ | 0.999 |
| Totaal aantal frames | 200,000,000 |
| Lossfunctie | Gemiddelde kwadratische fout |
| Scheduler | Lineaire annealing |
| $\epsilon$-greedy-exploratie | 10% of total frames |
| Kortingsfactor ($\gamma$) | 0.99 |
| GAE ($\lambda$) | 0.65 |
| Epochs | 2 |
| Batchgrootte | 4096 |

</div>

<p align="center"><em>Gebruikt in de encoder- en Hadamax-experimenten.</em></p>

## Statistische significantie

### Encoderexperimenten

<table>
  <tr>
    <th align="center">Wilcoxon-rangtekentoets</th>
    <th align="center">Wilcoxon-rangtekentoets (gecorrigeerd)</th>
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
    <th colspan="2" align="center">Kans op verbetering</th>
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

### Hadamax-experimenten

<table>
  <tr>
    <th align="center">Wilcoxon-rangtekentoets</th>
    <th align="center">Wilcoxon-rangtekentoets (gecorrigeerd)</th>
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
    <th colspan="2" align="center">Kans op verbetering</th>
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

### Experimenten met Q-waarden

<table>
  <tr>
    <th align="center">Wilcoxon-rangtekentoets</th>
    <th align="center">Wilcoxon-rangtekentoets (gecorrigeerd)</th>
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
    <th colspan="2" align="center">Kans op verbetering</th>
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

## Reproduceerbaarheid

Door het stochastische karakter van deep reinforcement learning zijn resultaten niet exact reproduceerbaar met vaste datasets.
Daarom verstrekken we de verzameling willekeurige seeds die in onze experimenten is gebruikt.

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

Volledige reproductie van de experimenten:

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

Via EnvPool is een uitgebreide verzameling Atari-omgevingen beschikbaar:
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

Procgen-omgevingen gebruiken hun eigen RGB-observaties met vorm `(3, 64, 64)`.
Aftab leest voor elke taak de EnvPool-configuratie en past alleen ondersteunde opties toe.
Opties die uitsluitend voor Atari gelden, zoals `noop`, `frame_skip`, `frame_stack` en
`train_episodic_life`, en de reward clipping van EnvPool worden daarom niet aan
Procgen doorgegeven.

Via EnvPool is een uitgebreide verzameling Procgen-omgevingen beschikbaar:

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## Hardware

Alle experimenten in dit project zijn uitgevoerd op [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40)-GPU’s.

| Specificatie | Details |
|--------------|----------|
| GPU-geheugen | 48 GB GDDR6 met foutcorrectiecode (ECC) |
| Bandbreedte van het GPU-geheugen | 696 GB/s |
| Interconnect | NVIDIA NVLink 112,5 GB/s (bidirectioneel); PCIe Gen4: 64 GB/s |
| NVLink | Bidirectioneel, low-profile (2 slots) |
| Beeldaansluitingen | 3x DisplayPort 1.4* |
| Maximaal stroomverbruik | 300 W |
| Afmetingen | 4,4" (H) × 10,5" (L), twee slots |
| Koeling | Passief |
| Ondersteunde vGPU-software | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| Ondersteunde vGPU-profielen | Zie de licentiehandleiding voor Virtual GPU |
| NVENC / NVDEC | 1x / 2x (inclusief AV1-decodering) |
| Veilig opstarten | Veilig en gemeten opstarten met een hardwarematige root of trust (optioneel) |
| NEBS-gereed | Niveau 3 |
| Voedingsaansluiting | 8-pins CPU |

## Citeren

```bibtex
@article{aftab2026drl,
  title={Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
}
```

### Gerelateerd werk

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

## Nuttige links

- [Wikipedia: Reinforcement learning (RL)](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [Wikipedia: Deep reinforcement learning (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Wikipedia: Q-learning](https://en.wikipedia.org/wiki/Q-learning)
- [Wikipedia: PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [Wikipedia: Statistische hypothesetoets](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [Wikipedia: Wilcoxon-rangtekentoets](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## Licentie

© 2025 Taha Shieenavaz.
Uitgegeven onder de CC BY-NC 4.0-licentie: https://creativecommons.org/licenses/by-nc/4.0/
