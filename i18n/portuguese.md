<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../figures/header-light.svg">
  <img alt="Artigo do Aftab" src="../figures/header-light.svg">
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

## Visão geral

**Aftab** (do <a href="https://en.wikipedia.org/wiki/Aftab">persa</a> آفتاب, que significa “sol” ou “raios solares”) é um framework de benchmark para avaliar codificadores baseados em CNN usados pelo PQN em diversos <a href="https://en.wikipedia.org/wiki/Atari_Games">jogos Atari</a>. Ele oferece ferramentas padronizadas de treinamento, avaliação e reprodutibilidade para pesquisas em aprendizado por reforço profundo.

Reunimos alguns vídeos que comparam os agentes PQN e Aftab. Assista [aqui](../videos.md).

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

Recomendamos fortemente o uso do [Micromamba](https://github.com/mamba-org/micromamba-releases) para criar ambientes virtuais. As instruções detalhadas estão disponíveis [aqui](../scripts/README.md).

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
- [Pontuações normalizadas em relação ao desempenho humano](../results/encoder_experiments/human_normalized_scores.md)
- [Pontuações](../results/encoder_experiments/scores.md)

**Gráficos**
- [IQM HNS](../figures/encoder_experiments/human_normalized_score)
- [Evolução da perda](../figures/encoder_experiments/loss)

---

### Experimentos com Hadamax

**Tabelas**
- [Pontuações normalizadas em relação ao desempenho humano](../results/hadamax_experiments/human_normalized_scores.md)
- [Pontuações](../results/hadamax_experiments/scores.md)

**Gráficos**
- [IQM HNS](../figures/hadamax_experiments/human_normalized_score)
- [Evolução da perda](../figures/hadamax_experiments/loss)

---

### Experimentos com valores Q

**Tabelas**
- [Pontuações normalizadas em relação ao desempenho humano](../results/qvalue_experiments/human_normalized_scores.md)
- [Pontuações](../results/qvalue_experiments/scores.md)

**Gráficos**
- [IQM HNS](../figures/qvalue_experiments/human_normalized_score)
- [Evolução da perda](../figures/qvalue_experiments/loss)

---

### Experimentos com Procgen

**Tabelas**
- [Pontuações normalizadas do Procgen](../results/procgen_experiments/procgen_normalized_scores.md)
- [Pontuações](../results/procgen_experiments/scores.md)


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

<div align="center">

| Hiperparâmetro | Valor |
| :--- | :--- |
| Taxa de aprendizado | $2.5 \times 10^{-4}$ |
| Ambientes de treinamento | 128 |
| Ambientes de teste | 8 |
| Otimizador | [Rectified Adam](https://arxiv.org/abs/1908.03265) |
| Decaimento de pesos | 0 |
| $\epsilon$ | $1 \times 10^{-5}$ |
| $\beta_{1}$ | 0.9 |
| $\beta_{2}$ | 0.999 |
| Total de frames | 200,000,000 |
| Função de perda | Erro quadrático médio |
| Agendador | Recozimento linear |
| Exploração $\epsilon$-greedy | 10% of total frames |
| Fator de desconto ($\gamma$) | 0.99 |
| GAE ($\lambda$) | 0.65 |
| Épocas | 2 |
| Tamanho do lote | 4096 |

</div>

<p align="center"><em>Usados nos experimentos com codificadores e Hadamax.</em></p>

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

```bibtex
@article{aftab2026drl,
  title={Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
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

- [Wikipédia: Aprendizado por reforço (RL)](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [Wikipédia: Aprendizado por reforço profundo (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Wikipédia: Q-learning](https://en.wikipedia.org/wiki/Q-learning)
- [Wikipédia: PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [Wikipédia: Teste de hipótese estatística](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [Wikipédia: Teste dos postos sinalizados de Wilcoxon](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## Licença

© 2025 Taha Shieenavaz.
Licenciado sob a licença CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
