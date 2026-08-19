<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-dark.svg">
  <img alt="Cabeçalho do Aftab" src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-light.svg">
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

## Visão geral

**Aftab** (do <a href="https://en.wikipedia.org/wiki/Aftab">persa</a> آفتاب, que significa “sol” ou “raios solares”) é um framework de benchmark para avaliar codificadores baseados em CNN usados pelo PQN em diversos <a href="https://pt.wikipedia.org/wiki/Atari_Games">jogos Atari</a>. Ele oferece ferramentas padronizadas de treinamento, avaliação e reprodutibilidade para pesquisas em aprendizado por reforço profundo.

Veja como a arquitetura Aftab se compara às referências PQN padrão nestas [demonstrações em vídeo](https://github.com/tahashieenavaz/aftab/blob/main/videos.md).

Esta pesquisa foi realizada sem financiamento; portanto, se nosso trabalho foi útil para você, considere [patrociná-lo no GitHub](https://github.com/sponsors/tahashieenavaz) 💛.

### Experimentos com codificadores

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
      <th>IQM HNS (últimos 50 milhões de frames)</th>
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

### Experimentos com Hadamax

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
      <th>IQM HNS (últimos 50 milhões de frames)</th>
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

Referências:
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### Experimentos com valores Q

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
      <th>IQM HNS (últimos 50 milhões de frames)</th>
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

Referências:
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

### Experimentos com Procgen (prevenção de sobreajuste)

Como não existem benchmarks públicos que comparem pontuações normalizadas em relação ao desempenho humano nos ambientes Procgen, criamos o PNS (Procgen Normalized Score), uma normalização min-max simples das pontuações entre as sementes.

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
      <th>IQM PNS (últimos 50 milhões de frames)</th>
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

## Instalação

Instalação via pip:

```bash
pip install aftab
```

Como alternativa, você pode clonar o repositório e instalá-lo no modo `editable`.

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

Recomendamos fortemente o uso do [Micromamba](https://github.com/mamba-org/micromamba-releases) para criar ambientes virtuais. As instruções detalhadas estão disponíveis [aqui](https://github.com/tahashieenavaz/aftab/blob/main/scripts/README.md).

## Treinamento de agentes

**A API JAX está atualmente em desenvolvimento** e deverá ser concluída até o final de 2026. Contribuições são muito bem-vindas.

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


## Integração de um codificador personalizado

Você pode definir seu próprio codificador como um módulo PyTorch e passá-lo ao agente:

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## Resultados

Todos os resultados experimentais estão organizados por categoria de experimento. Cada seção contém:
- **Tabelas**: resultados numéricos (HNS/PHS e pontuações brutas)
- **Gráficos**: pontuações normalizadas por IQM e curvas de treinamento

### Experimentos com codificadores

**Tabelas**
- [Pontuações normalizadas em relação ao desempenho humano](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/human_normalized_scores.md)
- [Pontuações](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/scores.md)

**Gráficos**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/human_normalized_score)
- [Evolução da perda](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/loss)

---

### Experimentos com Hadamax

**Tabelas**
- [Pontuações normalizadas em relação ao desempenho humano](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/human_normalized_scores.md)
- [Pontuações](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/scores.md)

**Gráficos**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/human_normalized_score)
- [Evolução da perda](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/loss)

---

### Experimentos com valores Q

**Tabelas**
- [Pontuações normalizadas em relação ao desempenho humano](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/human_normalized_scores.md)
- [Pontuações](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/scores.md)

**Gráficos**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/human_normalized_score)
- [Evolução da perda](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/loss)

---

### Experimentos com Procgen

**Tabelas**
- [Pontuações normalizadas do Procgen](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/procgen_normalized_scores.md)
- [Pontuações](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/scores.md)
- [AUC do PNS por semente](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_seed.md)
- [AUC do PNS por jogo](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_game.md)

**Gráficos**
- [IQM PNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/procgen_experiments/procgen_normalized_score)

## Complexidade dos modelos

### Variantes básicas

| Variante | Parâmetros do codificador | Parâmetros da cabeça de regressão | Total de parâmetros | FLOPs do codificador | FLOPs da cabeça de regressão | FLOPs totais |
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

> **Observação:** a variante Eta tem significativamente mais parâmetros do que as demais, principalmente porque seu codificador produz um grande número de características.

---

### Variantes Hadamax

| Variante | Parâmetros do codificador | Parâmetros da cabeça de regressão | Total de parâmetros | FLOPs do codificador | FLOPs da cabeça de regressão | FLOPs totais |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## Hiperparâmetros

As tabelas a seguir refletem os valores padrão definidos por `Aftab`. O argumento `experiment_name` é obrigatório e não possui valor padrão.

### Treinamento e ambiente

| Hiperparâmetro (argumento de `Aftab`) | Padrão |
| :--- | :--- |
| Codificador (`encoder`) | Gamma-Hadamax-Valid |
| Rede (`network`) | Dueling distribucional e bootstrapped (ensemble) |
| Total de frames (`frames`) | 200,000,000 |
| Salto de frames (`frame_skip`) | 4 |
| Empilhamento de frames (`frame_stack`) | 4 |
| Máximo de no-op (`noop`) | 30 |
| Taxa de aprendizado (`lr`) | $2.5 \times 10^{-4}$ |
| Ambientes de treinamento (`train_environments`) | 128 |
| Ambientes de teste (`test_environments`) | 8 |
| Passos por atualização (`steps_per_update`) | 32 |
| Tamanho do lote (derivado) | 4,096 |
| Minilotes (`mini_batches`) | 32 |
| Tamanho do minilote (derivado) | 128 |
| Fator de desconto ($\gamma$) | 0.99 |
| $\lambda$ do retorno (`return_lambda`) | 0.65 |
| Épocas (`epochs`) | 2 |
| Norma do gradiente (`gradient_norm`) | 10.0 |
| Dimensão do embedding (`embedding_dimension`) | 512 |
| Vida episódica no treinamento (`train_episodic_life`) | `True` |
| Vida episódica no teste (`test_episodic_life`) | `False` |
| Clipping de recompensa no treinamento (`train_reward_clip`) | `True` |
| Clipping de recompensa no teste (`test_reward_clip`) | `True` |
| Agendamento de épsilon | Linear |
| Razão de annealing de épsilon | 10% |

### Otimizador

| Hiperparâmetro (argumento de `Aftab`) | Padrão |
| :--- | :--- |
| Otimizador (`optimizer`) | [Rectified Adam](https://arxiv.org/abs/1908.03265) (`"radam"`) |
| Épsilon (`optimizer_epsilon`) | $1 \times 10^{-5}$ |
| Decaimento de pesos (`optimizer_weight_decay`) | 0.0 |
| $\beta_1$ (`optimizer_first_beta`) | 0.9 |
| $\beta_2$ (`optimizer_second_beta`) | 0.999 |

### Valores Q distribucionais e bootstrapped (ensemble)

| Hiperparâmetro (argumento de `Aftab`) | Padrão |
| :--- | :--- |
| Bins distribucionais (`distributional_bins`) | 51 |
| Mínimo distribucional (`distributional_min_value`) | -10.0 |
| Máximo distribucional (`distributional_max_value`) | 10.0 |
| Sigma distribucional (`distributional_sigma`) | `None` (derivado da razão sigma) |
| Razão sigma distribucional (`distributional_sigma_ratio`) | 0.75 |
| Clipping de valor distribucional (`distributional_value_clip`) | 0.0 |
| Cabeças bootstrap (`bootstrap_heads`) | 10 |
| Probabilidade bootstrap (`bootstrap_probability`) | 1.0 |

### Substituições do Procgen

| Hiperparâmetro | Padrão | Procgen |
| :--- | :--- | :--- |
| Ambientes de treinamento | 128 | 64 (`procgen_train_environments`) |
| Passos por atualização | 32 | 256 (`procgen_steps_per_update`) |
| Tamanho do lote | 4,096 | 16,384 |
| Tamanho do minilote | 128 | 512 |

<em>Nos ambientes Procgen, o Aftab aplica automaticamente as duas substituições acima; os demais padrões permanecem inalterados.</em>

## Significância estatística

### Experimentos com codificadores

<table>
  <tr>
    <th align="center">Teste dos postos sinalizados de Wilcoxon</th>
    <th align="center">Teste dos postos sinalizados de Wilcoxon (corrigido)</th>
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
    <th colspan="2" align="center">Probabilidade de melhoria</th>
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

### Experimentos com Hadamax

<table>
  <tr>
    <th align="center">Teste dos postos sinalizados de Wilcoxon</th>
    <th align="center">Teste dos postos sinalizados de Wilcoxon (corrigido)</th>
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
    <th colspan="2" align="center">Probabilidade de melhoria</th>
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

### Experimentos com valores Q

<table>
  <tr>
    <th align="center">Teste dos postos sinalizados de Wilcoxon</th>
    <th align="center">Teste dos postos sinalizados de Wilcoxon (corrigido)</th>
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
    <th colspan="2" align="center">Probabilidade de melhoria</th>
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

## Reprodutibilidade

Devido à natureza estocástica do aprendizado por reforço profundo, não é possível reproduzir exatamente os resultados usando conjuntos de dados fixos.
Por isso, fornecemos o conjunto de sementes aleatórias usado em nossos experimentos.

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

Reprodução completa dos experimentos:

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

O EnvPool oferece um conjunto abrangente de ambientes Atari:
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

Os ambientes Procgen usam suas observações RGB nativas com formato `(3, 64, 64)`.
O Aftab lê a configuração do EnvPool de cada tarefa e aplica somente as opções compatíveis.
Portanto, opções exclusivas do Atari, como `noop`, `frame_skip`, `frame_stack` e
`train_episodic_life`, assim como o recorte de recompensas do EnvPool, não são repassadas ao
Procgen.

O EnvPool oferece um conjunto abrangente de ambientes Procgen:

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## Hardware

Todos os experimentos deste projeto foram executados em GPUs [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40).

| Especificação | Detalhes |
|--------------|----------|
| Memória da GPU | 48 GB GDDR6 com código de correção de erros (ECC) |
| Largura de banda da memória da GPU | 696 GB/s |
| Interconexão | NVIDIA NVLink 112,5 GB/s (bidirecional); PCIe Gen4: 64 GB/s |
| NVLink | Bidirecional, perfil baixo (2 slots) |
| Portas de vídeo | 3x DisplayPort 1.4* |
| Consumo máximo de energia | 300 W |
| Dimensões | 4,4" (A) × 10,5" (C), dois slots |
| Refrigeração | Passiva |
| Software vGPU compatível | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| Perfis vGPU compatíveis | Consulte o Guia de Licenciamento de Virtual GPU |
| NVENC / NVDEC | 1x / 2x (inclui decodificação AV1) |
| Inicialização segura | Inicialização segura e medida com raiz de confiança de hardware (opcional) |
| Conformidade com NEBS | Nível 3 |
| Conector de alimentação | CPU de 8 pinos |

## Citação

Repositório:

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

### Trabalhos relacionados

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

## Links úteis

- [Wikipédia: Aprendizado por reforço (RL)](https://pt.wikipedia.org/wiki/Aprendizagem_por_reforço)
- [Wikipédia: Aprendizado por reforço profundo (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Wikipédia: Q-learning](https://pt.wikipedia.org/wiki/Q-learning)
- [Wikipédia: PyTorch](https://pt.wikipedia.org/wiki/PyTorch)
- [Wikipédia: Teste de hipótese estatística](https://pt.wikipedia.org/wiki/Testes_de_hipóteses)
- [Wikipédia: Teste dos postos sinalizados de Wilcoxon](https://pt.wikipedia.org/wiki/Teste_de_Wilcoxon)
- [PyTorch](https://pytorch.org/)

## Fonte

A fonte Vazirmatn é usada nos textos em persa e inglês tanto no cabeçalho do repositório do GitHub quanto na página inicial do projeto.

[GitHub](https://github.com/rastikerdar/vazirmatn) | [Google Fonts](https://fonts.google.com/specimen/Vazirmatn)

## Licença

© 2025 Taha Shieenavaz.
Licenciado sob a licença CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
