<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-dark.svg">
  <img alt="Aftab 헤더" src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-light.svg">
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

## 개요

**Aftab**(<a href="https://en.wikipedia.org/wiki/Aftab">페르시아어</a>: آفتاب, “태양” 또는 “햇살”이라는 뜻)은 여러 <a href="https://en.wikipedia.org/wiki/Atari_Games">Atari 게임</a>에서 PQN의 CNN 기반 인코더를 평가하기 위한 벤치마크 프레임워크입니다. 심층 강화학습 연구에 필요한 표준화된 학습·평가·재현성 도구를 제공합니다.

이 [영상 데모](https://github.com/tahashieenavaz/aftab/blob/main/videos.md)에서 Aftab 아키텍처와 표준 PQN 베이스라인을 비교해 보세요.

이 연구는 어떠한 자금 지원도 받지 않고 수행되었습니다. 저희 연구가 유용했다면 [GitHub에서 후원](https://github.com/sponsors/tahashieenavaz)을 고려해 주세요 💛.

### 인코더 실험

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
      <th>IQM HNS(마지막 5천만 프레임)</th>
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

### Hadamax 실험

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
      <th>IQM HNS(마지막 5천만 프레임)</th>
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

참고 문헌:
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### Q 값 실험

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
      <th>IQM HNS(마지막 5천만 프레임)</th>
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

참고 문헌:
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

### Procgen(과적합 방지) 실험

Procgen 환경의 인간 정규화 점수를 비교하는 공개 벤치마크가 없으므로, 여러 시드의 점수를 단순 최소-최대 정규화한 PNS(Procgen Normalized Score)를 만들었습니다.

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
      <th>IQM PNS (마지막 5천만 프레임)</th>
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

## 설치

pip로 설치합니다:

```bash
pip install aftab
```

또는 저장소를 복제한 뒤 `editable` 모드로 설치할 수 있습니다.

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

가상 환경을 만들 때는 [Micromamba](https://github.com/mamba-org/micromamba-releases)를 적극 권장합니다. 자세한 방법은 [여기](https://github.com/tahashieenavaz/aftab/blob/main/scripts/README.md)를 참고하세요.

## 에이전트 학습

**현재 JAX API는 개발 중이며** 2026년 말까지 완성할 예정입니다. 여러분의 기여를 환영합니다.

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


## 사용자 정의 인코더 연결

사용자 정의 인코더를 PyTorch 모듈로 정의해 에이전트에 전달할 수 있습니다:

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## 결과

모든 실험 결과는 실험 범주별로 정리되어 있습니다. 각 섹션에는 다음이 포함됩니다:
- **표**: 수치 결과(HNS/PHS 및 원점수)
- **차트**: IQM 정규화 점수 및 학습 곡선

### 인코더 실험

**표**
- [인간 정규화 점수](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/human_normalized_scores.md)
- [점수](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/scores.md)

**차트**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/human_normalized_score)
- [손실 추이](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/loss)

---

### Hadamax 실험

**표**
- [인간 정규화 점수](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/human_normalized_scores.md)
- [점수](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/scores.md)

**차트**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/human_normalized_score)
- [손실 추이](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/loss)

---

### Q 값 실험

**표**
- [인간 정규화 점수](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/human_normalized_scores.md)
- [점수](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/scores.md)

**차트**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/human_normalized_score)
- [손실 추이](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/loss)

---

### Procgen 실험

**표**
- [Procgen 정규화 점수](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/procgen_normalized_scores.md)
- [점수](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/scores.md)
- [시드별 PNS AUC](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_seed.md)
- [게임별 PNS AUC](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_game.md)

**차트**
- [IQM PNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/procgen_experiments/procgen_normalized_score)

## 모델 복잡도

### 기본 변형

| 변형 | 인코더 파라미터 | 회귀 헤드 파라미터 | 전체 파라미터 | 인코더 FLOPs | 회귀 헤드 FLOPs | 전체 FLOPs |
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

> **참고:** Eta 변형은 다른 변형보다 파라미터가 훨씬 많습니다. 주된 이유는 인코더가 많은 수의 특징을 생성하기 때문입니다.

---

### Hadamax 변형

| 변형 | 인코더 파라미터 | 회귀 헤드 파라미터 | 전체 파라미터 | 인코더 FLOPs | 회귀 헤드 FLOPs | 전체 FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## 하이퍼파라미터

다음 표는 `Aftab`에 정의된 기본값을 보여 줍니다. `experiment_name` 인수는 필수이며 기본값이 없습니다.

### 학습 및 환경

| 하이퍼파라미터(`Aftab` 인수) | 기본값 |
| :--- | :--- |
| 인코더(`encoder`) | Gamma-Hadamax-Valid |
| 네트워크(`network`) | Distributional Bootstrapped(Ensemble) Dueling |
| 전체 프레임(`frames`) | 200,000,000 |
| 프레임 건너뛰기(`frame_skip`) | 4 |
| 프레임 스택(`frame_stack`) | 4 |
| 최대 no-op(`noop`) | 30 |
| 학습률(`lr`) | $2.5 \times 10^{-4}$ |
| 학습 환경 수(`train_environments`) | 128 |
| 테스트 환경 수(`test_environments`) | 8 |
| 업데이트당 스텝(`steps_per_update`) | 32 |
| 배치 크기(파생값) | 4,096 |
| 미니배치 수(`mini_batches`) | 32 |
| 미니배치 크기(파생값) | 128 |
| 할인율($\gamma$) | 0.99 |
| 리턴 $\lambda$(`return_lambda`) | 0.65 |
| 에포크(`epochs`) | 2 |
| 그래디언트 노름(`gradient_norm`) | 10.0 |
| 임베딩 차원(`embedding_dimension`) | 512 |
| 학습 에피소드 라이프(`train_episodic_life`) | `True` |
| 테스트 에피소드 라이프(`test_episodic_life`) | `False` |
| 학습 보상 클리핑(`train_reward_clip`) | `True` |
| 테스트 보상 클리핑(`test_reward_clip`) | `True` |
| Epsilon 스케줄 | 선형 |
| Epsilon 어닐링 비율 | 10% |

### 옵티마이저

| 하이퍼파라미터(`Aftab` 인수) | 기본값 |
| :--- | :--- |
| 옵티마이저(`optimizer`) | [Rectified Adam](https://arxiv.org/abs/1908.03265) (`"radam"`) |
| Epsilon(`optimizer_epsilon`) | $1 \times 10^{-5}$ |
| 가중치 감쇠(`optimizer_weight_decay`) | 0.0 |
| $\beta_1$(`optimizer_first_beta`) | 0.9 |
| $\beta_2$(`optimizer_second_beta`) | 0.999 |

### Distributional 및 Bootstrapped(Ensemble) Q 값

| 하이퍼파라미터(`Aftab` 인수) | 기본값 |
| :--- | :--- |
| 분포 구간 수(`distributional_bins`) | 51 |
| 분포 최솟값(`distributional_min_value`) | -10.0 |
| 분포 최댓값(`distributional_max_value`) | 10.0 |
| 분포 Sigma(`distributional_sigma`) | `None`(Sigma 비율에서 파생) |
| 분포 Sigma 비율(`distributional_sigma_ratio`) | 0.75 |
| 분포 값 클리핑(`distributional_value_clip`) | 0.0 |
| Bootstrap 헤드 수(`bootstrap_heads`) | 10 |
| Bootstrap 확률(`bootstrap_probability`) | 1.0 |

### Procgen 재정의

| 하이퍼파라미터 | 기본값 | Procgen |
| :--- | :--- | :--- |
| 학습 환경 수 | 128 | 64(`procgen_train_environments`) |
| 업데이트당 스텝 | 32 | 256(`procgen_steps_per_update`) |
| 배치 크기 | 4,096 | 16,384 |
| 미니배치 크기 | 128 | 512 |

<em>Procgen 환경에서는 Aftab이 위 두 재정의를 자동으로 적용하며, 나머지 기본값은 변경되지 않습니다.</em>

## 통계적 유의성

### 인코더 실험

<table>
  <tr>
    <th align="center">Wilcoxon 부호 순위 검정</th>
    <th align="center">Wilcoxon 부호 순위 검정(보정)</th>
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
    <th colspan="2" align="center">개선 확률</th>
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

### Hadamax 실험

<table>
  <tr>
    <th align="center">Wilcoxon 부호 순위 검정</th>
    <th align="center">Wilcoxon 부호 순위 검정(보정)</th>
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
    <th colspan="2" align="center">개선 확률</th>
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

### Q 값 실험

<table>
  <tr>
    <th align="center">Wilcoxon 부호 순위 검정</th>
    <th align="center">Wilcoxon 부호 순위 검정(보정)</th>
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
    <th colspan="2" align="center">개선 확률</th>
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

## 재현성

심층 강화학습은 확률적 특성이 있으므로 고정된 데이터셋만으로 결과를 완전히 똑같이 재현하기는 어렵습니다.
대신 실험에 사용한 난수 시드 모음을 제공합니다.

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

전체 실험 재현:

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

EnvPool에서 다양한 Atari 환경을 사용할 수 있습니다:
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

Procgen 환경은 `(3, 64, 64)` 형태의 기본 RGB 관측값을 사용합니다.
Aftab은 각 태스크의 EnvPool 설정을 읽고 지원되는 옵션만 적용합니다.
따라서 `noop`, `frame_skip`, `frame_stack`, `train_episodic_life`처럼
Atari 전용인 옵션과 EnvPool의 보상 클리핑은 Procgen에 전달하지 않습니다.

EnvPool에서 다양한 Procgen 환경을 사용할 수 있습니다:

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## 하드웨어

이 프로젝트의 모든 실험은 [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40) GPU에서 실행했습니다.

| 사양 | 세부 정보 |
|--------------|----------|
| GPU 메모리 | 오류 정정 코드(ECC)를 지원하는 48 GB GDDR6 |
| GPU 메모리 대역폭 | 696 GB/s |
| 인터커넥트 | NVIDIA NVLink 112.5 GB/s(양방향), PCIe Gen4: 64 GB/s |
| NVLink | 양방향 로우 프로파일(2슬롯) |
| 디스플레이 포트 | 3x DisplayPort 1.4* |
| 최대 소비 전력 | 300 W |
| 폼 팩터 | 4.4"(높이) × 10.5"(길이), 듀얼 슬롯 |
| 냉각 방식 | 패시브 |
| 지원 vGPU 소프트웨어 | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| 지원 vGPU 프로필 | Virtual GPU 라이선스 가이드 참조 |
| NVENC / NVDEC | 1x / 2x(AV1 디코딩 포함) |
| 보안 부팅 | 하드웨어 신뢰 루트를 활용한 보안 및 측정 부팅(선택 사항) |
| NEBS 준수 | 레벨 3 |
| 전원 커넥터 | 8핀 CPU |

## 인용

저장소:

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

프리프린트:

```bibtex
@misc{2608.07335,
  Author = {Taha Shieenavaz and Shabnam Zareshahraki and Loris Nanni},
  Title = {Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  Year = {2026},
  Eprint = {arXiv:2608.07335},
}
```

### 관련 연구

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

## 유용한 링크

- [위키백과: 강화학습(RL)](https://ko.wikipedia.org/wiki/강화_학습)
- [위키백과: 심층 강화학습(DRL)](https://ko.wikipedia.org/wiki/심층_강화_학습)
- [위키백과: Q-러닝](https://ko.wikipedia.org/wiki/Q_러닝)
- [위키백과: PyTorch](https://ko.wikipedia.org/wiki/PyTorch)
- [위키백과: 통계적 가설 검정](https://ko.wikipedia.org/wiki/통계적_가설_검정)
- [위키백과: Wilcoxon 부호 순위 검정](https://ko.wikipedia.org/wiki/윌콕슨_부호-순위_검정)
- [PyTorch](https://pytorch.org/)

## 글꼴

GitHub 저장소 헤더와 프로젝트 랜딩 페이지의 페르시아어 및 영어 텍스트에는 Vazirmatn 글꼴을 사용합니다.

[GitHub](https://github.com/rastikerdar/vazirmatn) | [Google Fonts](https://fonts.google.com/specimen/Vazirmatn)

## 라이선스

© 2025 Taha Shieenavaz.
CC BY-NC 4.0 라이선스를 따릅니다: https://creativecommons.org/licenses/by-nc/4.0/
