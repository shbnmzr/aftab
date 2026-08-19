<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-dark.svg">
  <img alt="Intestazione Aftab" src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-light.svg">
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

## Panoramica

**Aftab** (dal <a href="https://en.wikipedia.org/wiki/Aftab">persiano</a> آفتاب, “sole” o “raggi del sole”) è un framework di benchmarking per valutare gli encoder basati su CNN impiegati da PQN in diversi <a href="https://it.wikipedia.org/wiki/Atari_Games">giochi Atari</a>. Offre strumenti standardizzati per l’addestramento, la valutazione e la riproducibilità della ricerca sull’apprendimento per rinforzo profondo.

Scopri come l’architettura Aftab si confronta con le baseline PQN standard in queste [dimostrazioni video](https://github.com/tahashieenavaz/aftab/blob/main/videos.md).

Questa ricerca è stata svolta senza finanziamenti; pertanto, se hai trovato utile il nostro lavoro, considera la possibilità di [sostenerci su GitHub](https://github.com/sponsors/tahashieenavaz) 💛.

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

### Esperimenti Procgen (prevenzione dell’overfitting)

Poiché non esistono benchmark pubblici che confrontino i punteggi normalizzati rispetto alle prestazioni umane negli ambienti Procgen, abbiamo creato PNS (Procgen Normalized Score), una semplice normalizzazione min-max dei punteggi tra i diversi seed.

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
      <th>IQM PNS (ultimi 50 milioni di frame)</th>
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

Consigliamo vivamente di usare [Micromamba](https://github.com/mamba-org/micromamba-releases) per creare gli ambienti virtuali. Le istruzioni dettagliate sono disponibili [qui](https://github.com/tahashieenavaz/aftab/blob/main/scripts/README.md).

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

Tutti i risultati sono organizzati per categoria di esperimento. Ogni sezione contiene:
- **Tabelle**: risultati numerici (HNS/PHS e punteggi grezzi)
- **Grafici**: punteggi normalizzati IQM e curve di addestramento

### Esperimenti sugli encoder

**Tabelle**
- [Punteggi normalizzati rispetto alle prestazioni umane](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/human_normalized_scores.md)
- [Punteggi](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/scores.md)

**Grafici**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/human_normalized_score)
- [Andamento della loss](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/loss)

---

### Esperimenti Hadamax

**Tabelle**
- [Punteggi normalizzati rispetto alle prestazioni umane](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/human_normalized_scores.md)
- [Punteggi](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/scores.md)

**Grafici**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/human_normalized_score)
- [Andamento della loss](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/loss)

---

### Esperimenti sui valori Q

**Tabelle**
- [Punteggi normalizzati rispetto alle prestazioni umane](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/human_normalized_scores.md)
- [Punteggi](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/scores.md)

**Grafici**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/human_normalized_score)
- [Andamento della loss](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/loss)

---

### Esperimenti Procgen

**Tabelle**
- [Punteggi Procgen normalizzati](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/procgen_normalized_scores.md)
- [Punteggi](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/scores.md)
- [AUC PNS per seed](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_seed.md)
- [AUC PNS per gioco](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_game.md)

**Grafici**
- [IQM PNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/procgen_experiments/procgen_normalized_score)

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

Le tabelle seguenti riportano i valori predefiniti definiti da `Aftab`. L’argomento `experiment_name` è obbligatorio e non ha un valore predefinito.

### Addestramento e ambiente

| Iperparametro (argomento di `Aftab`) | Valore predefinito |
| :--- | :--- |
| Encoder (`encoder`) | Gamma-Hadamax-Valid |
| Rete (`network`) | Dueling distribuzionale e bootstrapped (ensemble) |
| Frame totali (`frames`) | 200,000,000 |
| Salto dei frame (`frame_skip`) | 4 |
| Impilamento dei frame (`frame_stack`) | 4 |
| Massimo no-op (`noop`) | 30 |
| Tasso di apprendimento (`lr`) | $2.5 \times 10^{-4}$ |
| Ambienti di addestramento (`train_environments`) | 128 |
| Ambienti di test (`test_environments`) | 8 |
| Passi per aggiornamento (`steps_per_update`) | 32 |
| Dimensione del batch (derivata) | 4,096 |
| Mini-batch (`mini_batches`) | 32 |
| Dimensione del mini-batch (derivata) | 128 |
| Fattore di sconto ($\gamma$) | 0.99 |
| $\lambda$ del ritorno (`return_lambda`) | 0.65 |
| Epoche (`epochs`) | 2 |
| Norma del gradiente (`gradient_norm`) | 10.0 |
| Dimensione dell’embedding (`embedding_dimension`) | 512 |
| Vita episodica in addestramento (`train_episodic_life`) | `True` |
| Vita episodica in test (`test_episodic_life`) | `False` |
| Clipping delle ricompense in addestramento (`train_reward_clip`) | `True` |
| Clipping delle ricompense in test (`test_reward_clip`) | `True` |
| Pianificazione di epsilon | Lineare |
| Rapporto di annealing di epsilon | 10% |

### Ottimizzatore

| Iperparametro (argomento di `Aftab`) | Valore predefinito |
| :--- | :--- |
| Ottimizzatore (`optimizer`) | [Rectified Adam](https://arxiv.org/abs/1908.03265) (`"radam"`) |
| Epsilon (`optimizer_epsilon`) | $1 \times 10^{-5}$ |
| Decadimento dei pesi (`optimizer_weight_decay`) | 0.0 |
| $\beta_1$ (`optimizer_first_beta`) | 0.9 |
| $\beta_2$ (`optimizer_second_beta`) | 0.999 |

### Valori Q distribuzionali e bootstrapped (ensemble)

| Iperparametro (argomento di `Aftab`) | Valore predefinito |
| :--- | :--- |
| Bin distribuzionali (`distributional_bins`) | 51 |
| Minimo distribuzionale (`distributional_min_value`) | -10.0 |
| Massimo distribuzionale (`distributional_max_value`) | 10.0 |
| Sigma distribuzionale (`distributional_sigma`) | `None` (derivato dal rapporto sigma) |
| Rapporto sigma distribuzionale (`distributional_sigma_ratio`) | 0.75 |
| Clipping dei valori distribuzionali (`distributional_value_clip`) | 0.0 |
| Teste bootstrap (`bootstrap_heads`) | 10 |
| Probabilità bootstrap (`bootstrap_probability`) | 1.0 |

### Override Procgen

| Iperparametro | Valore predefinito | Procgen |
| :--- | :--- | :--- |
| Ambienti di addestramento | 128 | 64 (`procgen_train_environments`) |
| Passi per aggiornamento | 32 | 256 (`procgen_steps_per_update`) |
| Dimensione del batch | 4,096 | 16,384 |
| Dimensione del mini-batch | 128 | 512 |

<em>Per gli ambienti Procgen, Aftab applica automaticamente i due override indicati sopra; gli altri valori predefiniti rimangono invariati.</em>

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

- [Wikipedia: apprendimento per rinforzo (RL)](https://it.wikipedia.org/wiki/Apprendimento_per_rinforzo)
- [Wikipedia: apprendimento per rinforzo profondo (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Wikipedia: Q-learning](https://it.wikipedia.org/wiki/Q-learning)
- [Wikipedia: PyTorch](https://it.wikipedia.org/wiki/PyTorch)
- [Wikipedia: test d’ipotesi statistica](https://it.wikipedia.org/wiki/Test_di_verifica_d%27ipotesi)
- [Wikipedia: test dei ranghi con segno di Wilcoxon](https://it.wikipedia.org/wiki/Test_dei_ranghi_con_segno_di_Wilcoxon)
- [PyTorch](https://pytorch.org/)

## Carattere tipografico

Il carattere Vazirmatn è utilizzato per i testi in persiano e inglese sia nell’intestazione del repository GitHub sia nella pagina iniziale del progetto.

[GitHub](https://github.com/rastikerdar/vazirmatn) | [Google Fonts](https://fonts.google.com/specimen/Vazirmatn)

## Licenza

© 2025 Taha Shieenavaz.
Distribuito con licenza CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
