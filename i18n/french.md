<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../figures/header-light.svg">
  <img alt="Article Aftab" src="../figures/header-light.svg">
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

<br />

## Présentation

**Aftab** (<a href="https://en.wikipedia.org/wiki/Aftab">persan</a> : آفتاب, « soleil » ou « rayons du soleil ») est un cadre d’évaluation comparative des encodeurs basés sur des CNN employés par PQN dans différents <a href="https://en.wikipedia.org/wiki/Atari_Games">jeux Atari</a>. Il fournit des outils standardisés pour l’entraînement, l’évaluation et la reproductibilité des travaux de recherche en apprentissage par renforcement profond.

Nous avons réuni quelques vidéos comparant les agents PQN et Aftab. Vous pouvez les regarder [ici](../videos.md).

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

Nous recommandons vivement [Micromamba](https://github.com/mamba-org/micromamba-releases) pour créer les environnements virtuels. Les instructions détaillées sont disponibles [ici](../scripts/README.md).

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
- [Scores normalisés par rapport aux performances humaines](../results/encoder_experiments/human_normalized_scores.md)
- [Scores bruts](../results/encoder_experiments/scores.md)

**Graphiques**
- [IQM HNS](../figures/encoder_experiments/human_normalized_score)
- [Évolution de la perte](../figures/encoder_experiments/loss)

---

### Expériences Hadamax

**Tableaux**
- [Scores normalisés par rapport aux performances humaines](../results/hadamax_experiments/human_normalized_scores.md)
- [Scores bruts](../results/hadamax_experiments/scores.md)

**Graphiques**
- [IQM HNS](../figures/hadamax_experiments/human_normalized_score)
- [Évolution de la perte](../figures/hadamax_experiments/loss)

---

### Expériences sur les valeurs Q

**Tableaux**
- [Scores normalisés par rapport aux performances humaines](../results/qvalue_experiments/human_normalized_scores.md)
- [Scores bruts](../results/qvalue_experiments/scores.md)

**Graphiques**
- [IQM HNS](../figures/qvalue_experiments/human_normalized_score)
- [Évolution de la perte](../figures/qvalue_experiments/loss)

---

### Expériences Procgen

**Tableaux**
- [Scores Procgen normalisés](../results/procgen_experiments/procgen_normalized_scores.md)
- [Scores bruts](../results/procgen_experiments/scores.md)


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

<div align="center">

| Hyperparamètre | Valeur |
| :--- | :--- |
| Taux d’apprentissage | $2.5 \times 10^{-4}$ |
| Environnements d’entraînement | 128 |
| Environnements de test | 8 |
| Optimiseur | [Rectified Adam](https://arxiv.org/abs/1908.03265) |
| Décroissance des poids | 0 |
| $\epsilon$ | $1 \times 10^{-5}$ |
| $\beta_{1}$ | 0.9 |
| $\beta_{2}$ | 0.999 |
| Nombre total de trames | 200,000,000 |
| Fonction de perte | Erreur quadratique moyenne |
| Ordonnanceur | Décroissance linéaire |
| Exploration $\epsilon$-gloutonne | 10% of total frames |
| Facteur d’actualisation ($\gamma$) | 0.99 |
| GAE ($\lambda$) | 0.65 |
| Époques | 2 |
| Taille de lot | 4096 |

</div>

<p align="center"><em>Utilisés dans les expériences sur les encodeurs et Hadamax.</em></p>

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

```bibtex
@article{aftab2026drl,
  title={Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
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

- [Wikipédia : apprentissage par renforcement (RL)](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [Wikipédia : apprentissage par renforcement profond (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Wikipédia : Q-learning](https://en.wikipedia.org/wiki/Q-learning)
- [Wikipédia : PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [Wikipédia : test d’hypothèse statistique](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [Wikipédia : test des rangs signés de Wilcoxon](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## Licence

© 2025 Taha Shieenavaz.
Distribué sous licence CC BY-NC 4.0 : https://creativecommons.org/licenses/by-nc/4.0/
