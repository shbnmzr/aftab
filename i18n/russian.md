<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../figures/header-light.svg">
  <img alt="Статья Aftab" src="../figures/header-light.svg">
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

## Обзор

**Aftab** (от <a href="https://en.wikipedia.org/wiki/Aftab">персидского</a> آفتاب — «солнце» или «солнечные лучи») — это фреймворк для сравнительного тестирования CNN-энкодеров в PQN на различных <a href="https://en.wikipedia.org/wiki/Atari_Games">играх Atari</a>. Он предоставляет стандартизированные инструменты для обучения, оценки и воспроизводимости исследований в области глубокого обучения с подкреплением.

Мы подготовили несколько видео со сравнением агентов PQN и Aftab. Посмотреть их можно [здесь](../videos.md).

### Эксперименты с энкодерами

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
      <th>IQM HNS (последние 50 млн кадров)</th>
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

### Эксперименты с Hadamax

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
      <th>IQM HNS (последние 50 млн кадров)</th>
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

Ссылки:
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### Эксперименты со значениями Q

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
      <th>IQM HNS (последние 50 млн кадров)</th>
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

Ссылки:
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

### Эксперименты с Procgen (предотвращение переобучения)

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
      <th>IQM PNS (последние 50 млн кадров)</th>
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

## Установка

Установка через pip:

```bash
pip install aftab
```

Кроме того, можно клонировать репозиторий и установить пакет в режиме `editable`.

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

Для создания виртуальных окружений настоятельно рекомендуем использовать [Micromamba](https://github.com/mamba-org/micromamba-releases). Подробная инструкция доступна [здесь](../scripts/README.md).

## Обучение агентов

**API для JAX в настоящее время разрабатывается**, завершение работ запланировано до конца 2026 года. Мы будем рады вашим предложениям и вкладу в проект.

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


## Подключение собственного энкодера

Можно определить собственный энкодер как модуль PyTorch и передать его агенту:

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## Результаты

Все результаты сгруппированы по категориям экспериментов. Каждый раздел содержит:
- **Таблицы**: численные результаты (HNS/PHS и исходные очки)
- **Графики**: нормализованные оценки IQM и кривые обучения

### Эксперименты с энкодерами

**Таблицы**
- [Оценки, нормализованные относительно уровня человека](../results/encoder_experiments/human_normalized_scores.md)
- [Очки](../results/encoder_experiments/scores.md)

**Графики**
- [IQM HNS](../figures/encoder_experiments/human_normalized_score)
- [Динамика функции потерь](../figures/encoder_experiments/loss)

---

### Эксперименты с Hadamax

**Таблицы**
- [Оценки, нормализованные относительно уровня человека](../results/hadamax_experiments/human_normalized_scores.md)
- [Очки](../results/hadamax_experiments/scores.md)

**Графики**
- [IQM HNS](../figures/hadamax_experiments/human_normalized_score)
- [Динамика функции потерь](../figures/hadamax_experiments/loss)

---

### Эксперименты со значениями Q

**Таблицы**
- [Оценки, нормализованные относительно уровня человека](../results/qvalue_experiments/human_normalized_scores.md)
- [Очки](../results/qvalue_experiments/scores.md)

**Графики**
- [IQM HNS](../figures/qvalue_experiments/human_normalized_score)
- [Динамика функции потерь](../figures/qvalue_experiments/loss)

---

### Эксперименты с Procgen

**Таблицы**
- [Нормализованные оценки Procgen](../results/procgen_experiments/procgen_normalized_scores.md)
- [Очки](../results/procgen_experiments/scores.md)


## Сложность моделей

### Базовые варианты

| Вариант | Параметры энкодера | Параметры регрессионной головы | Всего параметров | FLOPs энкодера | FLOPs регрессионной головы | Всего FLOPs |
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

> **Примечание:** вариант Eta содержит значительно больше параметров, чем остальные варианты. Главная причина — большое количество признаков, формируемых его энкодером.

---

### Варианты Hadamax

| Вариант | Параметры энкодера | Параметры регрессионной головы | Всего параметров | FLOPs энкодера | FLOPs регрессионной головы | Всего FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## Гиперпараметры

<div align="center">

| Гиперпараметр | Значение |
| :--- | :--- |
| Скорость обучения | $2.5 \times 10^{-4}$ |
| Среды обучения | 128 |
| Тестовые среды | 8 |
| Оптимизатор | [Rectified Adam](https://arxiv.org/abs/1908.03265) |
| Затухание весов | 0 |
| $\epsilon$ | $1 \times 10^{-5}$ |
| $\beta_{1}$ | 0.9 |
| $\beta_{2}$ | 0.999 |
| Общее число кадров | 200,000,000 |
| Функция потерь | Среднеквадратичная ошибка |
| Планировщик | Линейное уменьшение |
| $\epsilon$-жадное исследование | 10% of total frames |
| Коэффициент дисконтирования ($\gamma$) | 0.99 |
| GAE ($\lambda$) | 0.65 |
| Эпохи | 2 |
| Размер батча | 4096 |

</div>

<p align="center"><em>Используются в экспериментах с энкодерами и Hadamax.</em></p>

## Статистическая значимость

### Эксперименты с энкодерами

<table>
  <tr>
    <th align="center">Знаково-ранговый критерий Уилкоксона</th>
    <th align="center">Знаково-ранговый критерий Уилкоксона (с поправкой)</th>
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
    <th colspan="2" align="center">Вероятность улучшения</th>
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

### Эксперименты с Hadamax

<table>
  <tr>
    <th align="center">Знаково-ранговый критерий Уилкоксона</th>
    <th align="center">Знаково-ранговый критерий Уилкоксона (с поправкой)</th>
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
    <th colspan="2" align="center">Вероятность улучшения</th>
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

### Эксперименты со значениями Q

<table>
  <tr>
    <th align="center">Знаково-ранговый критерий Уилкоксона</th>
    <th align="center">Знаково-ранговый критерий Уилкоксона (с поправкой)</th>
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
    <th colspan="2" align="center">Вероятность улучшения</th>
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

## Воспроизводимость

Из-за стохастической природы глубокого обучения с подкреплением точное воспроизведение результатов с помощью фиксированных наборов данных невозможно.
Вместо этого мы публикуем набор случайных начальных значений, использованных в экспериментах.

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

Полное воспроизведение экспериментов:

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

EnvPool предоставляет обширный набор сред Atari:
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

Среды Procgen используют собственные RGB-наблюдения формы `(3, 64, 64)`.
Aftab считывает конфигурацию EnvPool для каждой задачи и применяет только поддерживаемые параметры.
Поэтому специфичные для Atari параметры, такие как `noop`, `frame_skip`, `frame_stack` и
`train_episodic_life`, а также ограничение награды в EnvPool не передаются в
Procgen.

EnvPool предоставляет обширный набор сред Procgen:

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## Аппаратное обеспечение

Все эксперименты проекта выполнялись на графических процессорах [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40).

| Характеристика | Значение |
|--------------|----------|
| Память GPU | 48 ГБ GDDR6 с кодом коррекции ошибок (ECC) |
| Пропускная способность памяти GPU | 696 GB/s |
| Интерфейс | NVIDIA NVLink 112,5 ГБ/с (двунаправленный); PCIe Gen4: 64 ГБ/с |
| NVLink | Двунаправленный, низкопрофильный (2 слота) |
| Видеовыходы | 3x DisplayPort 1.4* |
| Максимальная потребляемая мощность | 300 W |
| Габариты | 4,4" (В) × 10,5" (Д), два слота |
| Охлаждение | Пассивное |
| Поддерживаемое ПО vGPU | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| Поддерживаемые профили vGPU | См. руководство по лицензированию Virtual GPU |
| NVENC / NVDEC | 1x / 2x (включая декодирование AV1) |
| Безопасная загрузка | Безопасная и измеряемая загрузка с аппаратным корнем доверия (опционально) |
| Соответствие NEBS | Уровень 3 |
| Разъём питания | 8-контактный CPU |

## Цитирование

```bibtex
@article{aftab2026drl,
  title={Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
}
```

### Связанные работы

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

## Полезные ссылки

- [Википедия: Обучение с подкреплением (RL)](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [Википедия: Глубокое обучение с подкреплением (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Википедия: Q-обучение](https://en.wikipedia.org/wiki/Q-learning)
- [Википедия: PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [Википедия: Проверка статистических гипотез](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [Википедия: Знаково-ранговый критерий Уилкоксона](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## Лицензия

© 2025 Taha Shieenavaz.
Распространяется по лицензии CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
