<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-dark.svg">
  <img alt="En-tête Aftab" src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-light.svg">
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

## Présentation

**Aftab** (<a href="https://en.wikipedia.org/wiki/Aftab">persan</a> : آفتاب, « soleil » ou « rayons du soleil ») est un cadre d’évaluation comparative des encodeurs basés sur des CNN employés par PQN dans différents <a href="https://fr.wikipedia.org/wiki/Atari_Games">jeux Atari</a>. Il fournit des outils standardisés pour l’entraînement, l’évaluation et la reproductibilité des travaux de recherche en apprentissage par renforcement profond.

Découvrez comment l’architecture Aftab se compare aux références PQN standard dans ces [démonstrations vidéo](https://github.com/tahashieenavaz/aftab/blob/main/videos.md).

Cette recherche a été menée sans aucun financement ; si notre travail vous a été utile, vous pouvez [nous soutenir sur GitHub](https://github.com/sponsors/tahashieenavaz) 💛.

### Expériences sur les encodeurs

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
      <th>IQM HNS (50 derniers millions de trames)</th>
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

### Expériences Hadamax

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
      <th>IQM HNS (50 derniers millions de trames)</th>
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

Références :
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### Expériences sur les valeurs Q

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
      <th>IQM HNS (50 derniers millions de trames)</th>
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

Références :
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

### Expériences Procgen (prévention du surapprentissage)

Comme il n’existe aucun benchmark public comparant les scores normalisés par rapport aux performances humaines dans les environnements Procgen, nous avons créé le PNS (Procgen Normalized Score), une simple normalisation min-max des scores entre les graines aléatoires.

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
      <th>IQM PNS (50 derniers millions de trames)</th>
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

Installation avec pip :

```bash
pip install aftab
```

Vous pouvez également cloner le dépôt et l’installer en mode `editable`.

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

Nous recommandons vivement [Micromamba](https://github.com/mamba-org/micromamba-releases) pour créer les environnements virtuels. Les instructions détaillées sont disponibles [ici](https://github.com/tahashieenavaz/aftab/blob/main/scripts/README.md).

## Entraînement des agents

**L’API JAX est actuellement en cours de développement** et devrait être achevée d’ici fin 2026. Les contributions sont vivement encouragées.

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


## Ajout d’un encodeur personnalisé

Vous pouvez définir votre propre encodeur sous forme de module PyTorch, puis le transmettre à l’agent :

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## Résultats

Tous les résultats sont classés par catégorie d’expérience. Chaque section contient :
- **Tableaux**: les résultats numériques (HNS/PHS et scores bruts)
- **Graphiques**: les scores normalisés IQM et les courbes d’entraînement

### Expériences sur les encodeurs

**Tableaux**
- [Scores normalisés par rapport aux performances humaines](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/human_normalized_scores.md)
- [Scores bruts](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/scores.md)

**Graphiques**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/human_normalized_score)
- [Évolution de la perte](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/loss)

---

### Expériences Hadamax

**Tableaux**
- [Scores normalisés par rapport aux performances humaines](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/human_normalized_scores.md)
- [Scores bruts](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/scores.md)

**Graphiques**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/human_normalized_score)
- [Évolution de la perte](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/loss)

---

### Expériences sur les valeurs Q

**Tableaux**
- [Scores normalisés par rapport aux performances humaines](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/human_normalized_scores.md)
- [Scores bruts](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/scores.md)

**Graphiques**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/human_normalized_score)
- [Évolution de la perte](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/loss)

---

### Expériences Procgen

**Tableaux**
- [Scores Procgen normalisés](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/procgen_normalized_scores.md)
- [Scores bruts](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/scores.md)
- [AUC du PNS par graine](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_seed.md)
- [AUC du PNS par jeu](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_game.md)

**Graphiques**
- [IQM PNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/procgen_experiments/procgen_normalized_score)

## Complexité des modèles

### Variantes de base

| Variante | Paramètres de l’encodeur | Paramètres de la tête de régression | Nombre total de paramètres | FLOPs de l’encodeur | FLOPs de la tête de régression | FLOPs totales |
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

> **Remarque :** la variante Eta comporte nettement plus de paramètres que les autres, principalement parce que son encodeur produit un grand nombre de caractéristiques.

---

### Variantes Hadamax

| Variante | Paramètres de l’encodeur | Paramètres de la tête de régression | Nombre total de paramètres | FLOPs de l’encodeur | FLOPs de la tête de régression | FLOPs totales |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## Hyperparamètres

Les tableaux suivants présentent les valeurs par défaut définies par `Aftab`. L’argument `experiment_name` est obligatoire et n’a pas de valeur par défaut.

### Entraînement et environnement

| Hyperparamètre (argument d’`Aftab`) | Valeur par défaut |
| :--- | :--- |
| Encodeur (`encoder`) | Gamma-Hadamax-Valid |
| Réseau (`network`) | Dueling distributionnel et bootstrapped (ensemble) |
| Nombre total de trames (`frames`) | 200,000,000 |
| Saut de trames (`frame_skip`) | 4 |
| Empilement de trames (`frame_stack`) | 4 |
| Nombre maximal de no-op (`noop`) | 30 |
| Taux d’apprentissage (`lr`) | $2.5 \times 10^{-4}$ |
| Environnements d’entraînement (`train_environments`) | 128 |
| Environnements de test (`test_environments`) | 8 |
| Pas par mise à jour (`steps_per_update`) | 32 |
| Taille du lot (dérivée) | 4,096 |
| Mini-lots (`mini_batches`) | 32 |
| Taille du mini-lot (dérivée) | 128 |
| Facteur d’actualisation ($\gamma$) | 0.99 |
| $\lambda$ de retour (`return_lambda`) | 0.65 |
| Époques (`epochs`) | 2 |
| Norme du gradient (`gradient_norm`) | 10.0 |
| Dimension d’embedding (`embedding_dimension`) | 512 |
| Vie épisodique à l’entraînement (`train_episodic_life`) | `True` |
| Vie épisodique au test (`test_episodic_life`) | `False` |
| Écrêtage des récompenses à l’entraînement (`train_reward_clip`) | `True` |
| Écrêtage des récompenses au test (`test_reward_clip`) | `True` |
| Planification d’epsilon | Linéaire |
| Ratio d’annealing d’epsilon | 10% |

### Optimiseur

| Hyperparamètre (argument d’`Aftab`) | Valeur par défaut |
| :--- | :--- |
| Optimiseur (`optimizer`) | [Rectified Adam](https://arxiv.org/abs/1908.03265) (`"radam"`) |
| Epsilon (`optimizer_epsilon`) | $1 \times 10^{-5}$ |
| Décroissance des poids (`optimizer_weight_decay`) | 0.0 |
| $\beta_1$ (`optimizer_first_beta`) | 0.9 |
| $\beta_2$ (`optimizer_second_beta`) | 0.999 |

### Valeurs Q distributionnelles et bootstrapped (ensemble)

| Hyperparamètre (argument d’`Aftab`) | Valeur par défaut |
| :--- | :--- |
| Intervalles distributionnels (`distributional_bins`) | 51 |
| Minimum distributionnel (`distributional_min_value`) | -10.0 |
| Maximum distributionnel (`distributional_max_value`) | 10.0 |
| Sigma distributionnel (`distributional_sigma`) | `None` (dérivé du ratio sigma) |
| Ratio sigma distributionnel (`distributional_sigma_ratio`) | 0.75 |
| Écrêtage des valeurs distributionnelles (`distributional_value_clip`) | 0.0 |
| Têtes bootstrap (`bootstrap_heads`) | 10 |
| Probabilité bootstrap (`bootstrap_probability`) | 1.0 |

### Substitutions Procgen

| Hyperparamètre | Valeur par défaut | Procgen |
| :--- | :--- | :--- |
| Environnements d’entraînement | 128 | 64 (`procgen_train_environments`) |
| Pas par mise à jour | 32 | 256 (`procgen_steps_per_update`) |
| Taille du lot | 4,096 | 16,384 |
| Taille du mini-lot | 128 | 512 |

<em>Pour les environnements Procgen, Aftab applique automatiquement les deux substitutions ci-dessus ; les autres valeurs par défaut restent inchangées.</em>

## Significativité statistique

### Expériences sur les encodeurs

<table>
  <tr>
    <th align="center">Test des rangs signés de Wilcoxon</th>
    <th align="center">Test des rangs signés de Wilcoxon (corrigé)</th>
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
    <th colspan="2" align="center">Probabilité d’amélioration</th>
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

### Expériences Hadamax

<table>
  <tr>
    <th align="center">Test des rangs signés de Wilcoxon</th>
    <th align="center">Test des rangs signés de Wilcoxon (corrigé)</th>
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
    <th colspan="2" align="center">Probabilité d’amélioration</th>
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

### Expériences sur les valeurs Q

<table>
  <tr>
    <th align="center">Test des rangs signés de Wilcoxon</th>
    <th align="center">Test des rangs signés de Wilcoxon (corrigé)</th>
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
    <th colspan="2" align="center">Probabilité d’amélioration</th>
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

## Reproductibilité

En raison de la nature stochastique de l’apprentissage par renforcement profond, des jeux de données fixes ne permettent pas une reproduction parfaitement identique.
Nous fournissons donc l’ensemble des graines aléatoires utilisées dans nos expériences.

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

Reproduction complète des expériences :

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

Un vaste ensemble d’environnements Atari est disponible dans EnvPool :
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

Les environnements Procgen utilisent leurs observations RGB natives de forme `(3, 64, 64)`.
Aftab lit la configuration EnvPool de chaque tâche et n’applique que les options prises en charge.
Les options propres à Atari, comme `noop`, `frame_skip`, `frame_stack` et
`train_episodic_life`, ainsi que l’écrêtage des récompenses d’EnvPool, ne sont donc pas transmises à
Procgen.

Un vaste ensemble d’environnements Procgen est disponible dans EnvPool :

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## Matériel

Toutes les expériences de ce projet ont été exécutées sur des GPU [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40).

| Caractéristique | Détails |
|--------------|----------|
| Mémoire GPU | 48 Go de GDDR6 avec code correcteur d’erreurs (ECC) |
| Bande passante de la mémoire GPU | 696 GB/s |
| Interconnexion | NVIDIA NVLink 112,5 Go/s (bidirectionnel) ; PCIe Gen4 : 64 Go/s |
| NVLink | Bidirectionnel, profil bas (2 emplacements) |
| Ports d’affichage | 3x DisplayPort 1.4* |
| Consommation maximale | 300 W |
| Format | 4,4 po (H) × 10,5 po (L), double emplacement |
| Refroidissement | Passif |
| Logiciels vGPU pris en charge | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| Profils vGPU pris en charge | Voir le guide des licences Virtual GPU |
| NVENC / NVDEC | 1x / 2x (décodage AV1 inclus) |
| Démarrage sécurisé | Démarrage sécurisé et mesuré avec racine matérielle de confiance (facultatif) |
| Conformité NEBS | Niveau 3 |
| Connecteur d’alimentation | CPU à 8 broches |

## Citation

Référentiel :

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

Prépublication :

```bibtex
@misc{2608.07335,
  Author = {Taha Shieenavaz and Shabnam Zareshahraki and Loris Nanni},
  Title = {Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  Year = {2026},
  Eprint = {arXiv:2608.07335},
}
```

### Travaux connexes

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

## Liens utiles

- [Wikipédia : apprentissage par renforcement (RL)](https://fr.wikipedia.org/wiki/Apprentissage_par_renforcement)
- [Wikipédia : apprentissage par renforcement profond (DRL)](https://fr.wikipedia.org/wiki/Apprentissage_par_renforcement_profond)
- [Wikipédia : Q-learning](https://fr.wikipedia.org/wiki/Q-learning)
- [Wikipédia : PyTorch](https://fr.wikipedia.org/wiki/PyTorch)
- [Wikipédia : test d’hypothèse statistique](https://fr.wikipedia.org/wiki/Test_statistique)
- [Wikipédia : test des rangs signés de Wilcoxon](https://fr.wikipedia.org/wiki/Test_des_rangs_signés_de_Wilcoxon)
- [PyTorch](https://pytorch.org/)

## Police

La police Vazirmatn est utilisée pour les textes persan et anglais dans l’en-tête du dépôt GitHub et sur la page d’accueil du projet.

[GitHub](https://github.com/rastikerdar/vazirmatn) | [Google Fonts](https://fonts.google.com/specimen/Vazirmatn)

## Licence

© 2025 Taha Shieenavaz.
Distribué sous licence CC BY-NC 4.0 : https://creativecommons.org/licenses/by-nc/4.0/
