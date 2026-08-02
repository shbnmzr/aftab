<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../figures/header-light.svg">
  <img alt="Articolo Aftab" src="../figures/header-light.svg">
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

## Panoramica

**Aftab** (dal <a href="https://en.wikipedia.org/wiki/Aftab">persiano</a> آفتاب, “sole” o “raggi del sole”) è un framework di benchmarking per valutare gli encoder basati su CNN impiegati da PQN in diversi <a href="https://en.wikipedia.org/wiki/Atari_Games">giochi Atari</a>. Offre strumenti standardizzati per l’addestramento, la valutazione e la riproducibilità della ricerca sull’apprendimento per rinforzo profondo.

Abbiamo raccolto alcuni video che confrontano gli agenti PQN e Aftab. Puoi guardarli [qui](../videos.md).

### Esperimenti sugli encoder

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
      <th>IQM HNS (ultimi 50 milioni di frame)</th>
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

### Esperimenti Hadamax

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
      <th>IQM HNS (ultimi 50 milioni di frame)</th>
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

Riferimenti:
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### Esperimenti sui valori Q

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
      <th>IQM HNS (ultimi 50 milioni di frame)</th>
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

Riferimenti:
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

## Installazione

Installazione con pip:

```bash
pip install aftab
```

In alternativa, puoi clonare il repository e installarlo in modalità `editable`.

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

Consigliamo vivamente di usare [Micromamba](https://github.com/mamba-org/micromamba-releases) per creare gli ambienti virtuali. Le istruzioni dettagliate sono disponibili [qui](../scripts/README.md).

## Addestramento degli agenti

**L’API JAX è attualmente in fase di sviluppo** e dovrebbe essere completata entro la fine del 2026. I contributi sono particolarmente graditi.

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


## Inserimento di un encoder personalizzato

Puoi definire un encoder personalizzato come modulo PyTorch e passarlo all’agente:

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## Risultati

**Esperimenti sugli encoder**:

- Tabelle:
  - [HNS](../results/encoder_experiments/human_normalized_scores.md)
  - [Punteggi](../results/encoder_experiments/scores.md)
- Grafici:
  - [IQM HNS](https://github.com/tahashieenavaz/aftab/tree/main/figures/encoder_experiments/human_normalized_score)
  - [Andamento della loss](https://github.com/tahashieenavaz/aftab/tree/main/figures/encoder_experiments/loss)

**Esperimenti Hadamax**:

- Tabelle:
  - [HNS](../results/hadamax_experiments/human_normalized_scores.md)
  - [Punteggi](../results/hadamax_experiments/scores.md)
- Grafici:
  - [IQM HNS](https://github.com/tahashieenavaz/aftab/tree/main/figures/hadamax_experiments/human_normalized_score)
  - [Andamento della loss](https://github.com/tahashieenavaz/aftab/tree/main/figures/hadamax_experiments/loss)

**Esperimenti sui valori Q**:
- Tabelle:
  - [HNS](../results/qvalue_experiments/human_normalized_scores.md)
  - [Punteggi](../results/qvalue_experiments/scores.md)
- Grafici:
  - [IQM HNS](https://github.com/tahashieenavaz/aftab/tree/main/figures/qvalue_experiments/human_normalized_score)
  - [Andamento della loss](https://github.com/tahashieenavaz/aftab/tree/main/figures/qvalue_experiments/loss)

**Esperimenti Procgen**:
- Tabelle:
  - [PHS](../results/procgen_experiments/procgen_normalized_scores.md)
  - [Punteggi](../results/procgen_experiments/scores.md)


## Complessità dei modelli

### Varianti di base

| Variante | Parametri dell’encoder | Parametri della testa di regressione | Parametri totali | FLOPs dell’encoder | FLOPs della testa di regressione | FLOPs totali |
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

> **Nota:** la variante Eta ha molti più parametri delle altre, soprattutto perché il suo encoder produce un numero elevato di feature.

---

### Varianti Hadamax

| Variante | Parametri dell’encoder | Parametri della testa di regressione | Parametri totali | FLOPs dell’encoder | FLOPs della testa di regressione | FLOPs totali |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## Iperparametri

<div align="center">

| Iperparametro | Valore |
| :--- | :--- |
| Tasso di apprendimento | $2.5 \times 10^{-4}$ |
| Ambienti di addestramento | 128 |
| Ambienti di test | 8 |
| Ottimizzatore | [Rectified Adam](https://arxiv.org/abs/1908.03265) |
| Decadimento dei pesi | 0 |
| $\epsilon$ | $1 \times 10^{-5}$ |
| $\beta_{1}$ | 0.9 |
| $\beta_{2}$ | 0.999 |
| Frame totali | 200,000,000 |
| Funzione di loss | Errore quadratico medio |
| Scheduler | Decadimento lineare |
| Esplorazione $\epsilon$-greedy | 10% of total frames |
| Fattore di sconto ($\gamma$) | 0.99 |
| GAE ($\lambda$) | 0.65 |
| Epoche | 2 |
| Dimensione del batch | 4096 |

</div>

<p align="center"><em>Utilizzati negli esperimenti sugli encoder e Hadamax.</em></p>

## Significatività statistica

### Esperimenti sugli encoder

<table>
  <tr>
    <th align="center">Test dei ranghi con segno di Wilcoxon</th>
    <th align="center">Test dei ranghi con segno di Wilcoxon (corretto)</th>
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
    <th colspan="2" align="center">Probabilità di miglioramento</th>
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

### Esperimenti Hadamax

<table>
  <tr>
    <th align="center">Test dei ranghi con segno di Wilcoxon</th>
    <th align="center">Test dei ranghi con segno di Wilcoxon (corretto)</th>
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
    <th colspan="2" align="center">Probabilità di miglioramento</th>
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

### Esperimenti sui valori Q

<table>
  <tr>
    <th align="center">Test dei ranghi con segno di Wilcoxon</th>
    <th align="center">Test dei ranghi con segno di Wilcoxon (corretto)</th>
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
    <th colspan="2" align="center">Probabilità di miglioramento</th>
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

## Riproducibilità

A causa della natura stocastica dell’apprendimento per rinforzo profondo, non è possibile ottenere una riproduzione esatta usando dataset fissi.
Forniamo quindi l’insieme dei seed casuali utilizzati nei nostri esperimenti.

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

Riproduzione completa degli esperimenti:

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

EnvPool mette a disposizione un’ampia raccolta di ambienti Atari:
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

Gli ambienti Procgen usano le proprie osservazioni RGB native con forma `(3, 64, 64)`.
Aftab legge la configurazione EnvPool di ogni task e applica soltanto le opzioni supportate.
Le opzioni specifiche per Atari, come `noop`, `frame_skip`, `frame_stack` e
`train_episodic_life`, e il clipping delle ricompense di EnvPool non vengono quindi passati a
Procgen.

EnvPool mette a disposizione un’ampia raccolta di ambienti Procgen:

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## Hardware

Tutti gli esperimenti del progetto sono stati eseguiti su GPU [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40).

| Specifica | Dettagli |
|--------------|----------|
| Memoria GPU | 48 GB GDDR6 con codice di correzione degli errori (ECC) |
| Larghezza di banda della memoria GPU | 696 GB/s |
| Interconnessione | NVIDIA NVLink 112,5 GB/s (bidirezionale); PCIe Gen4: 64 GB/s |
| NVLink | Bidirezionale, a basso profilo (2 slot) |
| Porte video | 3x DisplayPort 1.4* |
| Consumo massimo | 300 W |
| Formato | 4,4" (A) × 10,5" (L), doppio slot |
| Raffreddamento | Passivo |
| Software vGPU supportato | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| Profili vGPU supportati | Consultare la guida alle licenze Virtual GPU |
| NVENC / NVDEC | 1x / 2x (decodifica AV1 inclusa) |
| Avvio sicuro | Avvio sicuro e misurato con radice hardware di attendibilità (opzionale) |
| Conformità NEBS | Livello 3 |
| Connettore di alimentazione | CPU a 8 pin |

## Citazione

```bibtex
@article{aftab2026drl,
  title={Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
}
```

### Lavori correlati

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

## Link utili

- [Wikipedia: apprendimento per rinforzo (RL)](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [Wikipedia: apprendimento per rinforzo profondo (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Wikipedia: Q-learning](https://en.wikipedia.org/wiki/Q-learning)
- [Wikipedia: PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [Wikipedia: test d’ipotesi statistica](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [Wikipedia: test dei ranghi con segno di Wilcoxon](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## Licenza

© 2025 Taha Shieenavaz.
Distribuito con licenza CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
