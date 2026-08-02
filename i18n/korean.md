<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../figures/header-light.svg">
  <img alt="Aftab 논문" src="../figures/header-light.svg">
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

## 개요

**Aftab**(<a href="https://en.wikipedia.org/wiki/Aftab">페르시아어</a>: آفتاب, “태양” 또는 “햇살”이라는 뜻)은 여러 <a href="https://en.wikipedia.org/wiki/Atari_Games">Atari 게임</a>에서 PQN의 CNN 기반 인코더를 평가하기 위한 벤치마크 프레임워크입니다. 심층 강화학습 연구에 필요한 표준화된 학습·평가·재현성 도구를 제공합니다.

PQN과 Aftab 에이전트를 비교한 영상을 모았습니다. [여기](../videos.md)에서 확인할 수 있습니다.

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

가상 환경을 만들 때는 [Micromamba](https://github.com/mamba-org/micromamba-releases)를 적극 권장합니다. 자세한 방법은 [여기](../scripts/README.md)를 참고하세요.

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

**인코더 실험**:

- 표:
  - [HNS](../results/encoder_experiments/human_normalized_scores.md)
  - [점수](../results/encoder_experiments/scores.md)
- 차트:
  - [IQM HNS](https://github.com/tahashieenavaz/aftab/tree/main/figures/encoder_experiments/human_normalized_score)
  - [손실 추이](https://github.com/tahashieenavaz/aftab/tree/main/figures/encoder_experiments/loss)

**Hadamax 실험**:

- 표:
  - [HNS](../results/hadamax_experiments/human_normalized_scores.md)
  - [점수](../results/hadamax_experiments/scores.md)
- 차트:
  - [IQM HNS](https://github.com/tahashieenavaz/aftab/tree/main/figures/hadamax_experiments/human_normalized_score)
  - [손실 추이](https://github.com/tahashieenavaz/aftab/tree/main/figures/hadamax_experiments/loss)

**Q 값 실험**:
- 표:
  - [HNS](../results/qvalue_experiments/human_normalized_scores.md)
  - [점수](../results/qvalue_experiments/scores.md)
- 차트:
  - [IQM HNS](https://github.com/tahashieenavaz/aftab/tree/main/figures/qvalue_experiments/human_normalized_score)
  - [손실 추이](https://github.com/tahashieenavaz/aftab/tree/main/figures/qvalue_experiments/loss)

**Procgen 실험**:
- 표:
  - [PHS](../results/procgen_experiments/procgen_normalized_scores.md)
  - [점수](../results/procgen_experiments/scores.md)


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

<div align="center">

| 하이퍼파라미터 | 값 |
| :--- | :--- |
| 학습률 | $2.5 \times 10^{-4}$ |
| 학습 환경 수 | 128 |
| 테스트 환경 수 | 8 |
| 옵티마이저 | [Rectified Adam](https://arxiv.org/abs/1908.03265) |
| 가중치 감쇠 | 0 |
| $\epsilon$ | $1 \times 10^{-5}$ |
| $\beta_{1}$ | 0.9 |
| $\beta_{2}$ | 0.999 |
| 전체 프레임 | 200,000,000 |
| 손실 함수 | 평균 제곱 오차 |
| 스케줄러 | 선형 감쇠 |
| $\epsilon$-탐욕 탐색 | 10% of total frames |
| 할인율($\gamma$) | 0.99 |
| GAE ($\lambda$) | 0.65 |
| 에포크 | 2 |
| 배치 크기 | 4096 |

</div>

<p align="center"><em>인코더 및 Hadamax 실험에 사용했습니다.</em></p>

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

```bibtex
@article{aftab2026drl,
  title={Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
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

- [위키백과: 강화학습(RL)](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [위키백과: 심층 강화학습(DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [위키백과: Q-러닝](https://en.wikipedia.org/wiki/Q-learning)
- [위키백과: PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [위키백과: 통계적 가설 검정](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [위키백과: Wilcoxon 부호 순위 검정](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## 라이선스

© 2025 Taha Shieenavaz.
CC BY-NC 4.0 라이선스를 따릅니다: https://creativecommons.org/licenses/by-nc/4.0/
