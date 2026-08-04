<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../figures/header-light.svg">
  <img alt="Aftab গবেষণাপত্র" src="../figures/header-light.svg">
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

## সংক্ষিপ্ত পরিচিতি

**Aftab** (<a href="https://en.wikipedia.org/wiki/Aftab">ফারসি</a>: آفتاب, অর্থ “সূর্য” বা “সূর্যের রশ্মি”) হলো বিভিন্ন <a href="https://en.wikipedia.org/wiki/Atari_Games">Atari গেমে</a> PQN-এর CNN-ভিত্তিক এনকোডার মূল্যায়নের একটি বেঞ্চমার্কিং ফ্রেমওয়ার্ক। এটি গভীর রিইনফোর্সমেন্ট লার্নিং গবেষণার জন্য প্রশিক্ষণ, মূল্যায়ন ও পুনরুৎপাদনযোগ্যতার প্রমিত টুল সরবরাহ করে।

PQN ও Aftab এজেন্টের তুলনামূলক কয়েকটি ভিডিও আমরা তৈরি করেছি। সেগুলো [এখানে](../videos.md) দেখুন।

### এনকোডার পরীক্ষা

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
      <th>IQM HNS (শেষ ৫ কোটি ফ্রেম)</th>
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

### Hadamax পরীক্ষা

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
      <th>IQM HNS (শেষ ৫ কোটি ফ্রেম)</th>
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

তথ্যসূত্র:
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### Q-মান পরীক্ষা

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
      <th>IQM HNS (শেষ ৫ কোটি ফ্রেম)</th>
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

তথ্যসূত্র:
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

### Procgen পরীক্ষা (ওভারফিটিং প্রতিরোধ)

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
      <th>IQM PNS (শেষ ৫ কোটি ফ্রেম)</th>
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

## ইনস্টলেশন

pip দিয়ে ইনস্টল করুন:

```bash
pip install aftab
```

বিকল্পভাবে, রিপোজিটরি ক্লোন করে `editable` মোডে ইনস্টল করতে পারেন।

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

ভার্চুয়াল এনভায়রনমেন্ট তৈরির জন্য আমরা [Micromamba](https://github.com/mamba-org/micromamba-releases) ব্যবহারের জোরালো পরামর্শ দিই। বিস্তারিত নির্দেশনা [এখানে](../scripts/README.md) রয়েছে।

## এজেন্ট প্রশিক্ষণ

**JAX API বর্তমানে উন্নয়নাধীন** এবং ২০২৬ সালের শেষ নাগাদ এটি সম্পন্ন করার পরিকল্পনা রয়েছে। অবদানকে আন্তরিকভাবে স্বাগত জানাই।

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


## কাস্টম এনকোডার যুক্ত করা

নিজস্ব এনকোডারকে PyTorch মডিউল হিসেবে সংজ্ঞায়িত করে এজেন্টে পাঠাতে পারেন:

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## ফলাফল

সব পরীক্ষার ফলাফল পরীক্ষার ধরন অনুযায়ী সাজানো হয়েছে। প্রতিটি অংশে রয়েছে:
- **টেবিল**: সংখ্যাগত ফলাফল (HNS/PHS এবং অপরিশোধিত স্কোর)
- **চার্ট**: IQM-স্বাভাবিকীকৃত স্কোর ও প্রশিক্ষণ কার্ভ

### এনকোডার পরীক্ষা

**টেবিল**
- [মানবীয় পারফরম্যান্সের তুলনায় স্বাভাবিকীকৃত স্কোর](../results/encoder_experiments/human_normalized_scores.md)
- [স্কোর](../results/encoder_experiments/scores.md)

**চার্ট**
- [IQM HNS](../figures/encoder_experiments/human_normalized_score)
- [লসের পরিবর্তন](../figures/encoder_experiments/loss)

---

### Hadamax পরীক্ষা

**টেবিল**
- [মানবীয় পারফরম্যান্সের তুলনায় স্বাভাবিকীকৃত স্কোর](../results/hadamax_experiments/human_normalized_scores.md)
- [স্কোর](../results/hadamax_experiments/scores.md)

**চার্ট**
- [IQM HNS](../figures/hadamax_experiments/human_normalized_score)
- [লসের পরিবর্তন](../figures/hadamax_experiments/loss)

---

### Q-মান পরীক্ষা

**টেবিল**
- [মানবীয় পারফরম্যান্সের তুলনায় স্বাভাবিকীকৃত স্কোর](../results/qvalue_experiments/human_normalized_scores.md)
- [স্কোর](../results/qvalue_experiments/scores.md)

**চার্ট**
- [IQM HNS](../figures/qvalue_experiments/human_normalized_score)
- [লসের পরিবর্তন](../figures/qvalue_experiments/loss)

---

### Procgen পরীক্ষা

**টেবিল**
- [Procgen স্বাভাবিকীকৃত স্কোর](../results/procgen_experiments/procgen_normalized_scores.md)
- [স্কোর](../results/procgen_experiments/scores.md)


## মডেলের জটিলতা

### ভিত্তি ভ্যারিয়েন্ট

| ভ্যারিয়েন্ট | এনকোডার প্যারামিটার | রিগ্রেশন হেড প্যারামিটার | মোট প্যারামিটার | এনকোডার FLOPs | রিগ্রেশন হেড FLOPs | মোট FLOPs |
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

> **দ্রষ্টব্য:** Eta ভ্যারিয়েন্টে অন্যগুলোর তুলনায় উল্লেখযোগ্যভাবে বেশি প্যারামিটার রয়েছে। এর প্রধান কারণ হলো এনকোডারটি বিপুল সংখ্যক ফিচার তৈরি করে।

---

### Hadamax ভ্যারিয়েন্ট

| ভ্যারিয়েন্ট | এনকোডার প্যারামিটার | রিগ্রেশন হেড প্যারামিটার | মোট প্যারামিটার | এনকোডার FLOPs | রিগ্রেশন হেড FLOPs | মোট FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## হাইপারপ্যারামিটার

<div align="center">

| হাইপারপ্যারামিটার | মান |
| :--- | :--- |
| লার্নিং রেট | $2.5 \times 10^{-4}$ |
| প্রশিক্ষণ এনভায়রনমেন্ট | 128 |
| পরীক্ষণ এনভায়রনমেন্ট | 8 |
| অপটিমাইজার | [Rectified Adam](https://arxiv.org/abs/1908.03265) |
| ওয়েট ডিকে | 0 |
| $\epsilon$ | $1 \times 10^{-5}$ |
| $\beta_{1}$ | 0.9 |
| $\beta_{2}$ | 0.999 |
| মোট ফ্রেম | 200,000,000 |
| লস ফাংশন | গড় বর্গ ত্রুটি |
| শিডিউলার | রৈখিক অ্যানিলিং |
| $\epsilon$-greedy অনুসন্ধান | 10% of total frames |
| ডিসকাউন্ট ফ্যাক্টর ($\gamma$) | 0.99 |
| GAE ($\lambda$) | 0.65 |
| এপক | 2 |
| ব্যাচের আকার | 4096 |

</div>

<p align="center"><em>এনকোডার ও Hadamax পরীক্ষায় ব্যবহৃত হয়েছে।</em></p>

## পরিসংখ্যানগত তাৎপর্য

### এনকোডার পরীক্ষা

<table>
  <tr>
    <th align="center">Wilcoxon signed-rank পরীক্ষা</th>
    <th align="center">Wilcoxon signed-rank পরীক্ষা (সংশোধিত)</th>
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
    <th colspan="2" align="center">উন্নতির সম্ভাবনা</th>
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

### Hadamax পরীক্ষা

<table>
  <tr>
    <th align="center">Wilcoxon signed-rank পরীক্ষা</th>
    <th align="center">Wilcoxon signed-rank পরীক্ষা (সংশোধিত)</th>
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
    <th colspan="2" align="center">উন্নতির সম্ভাবনা</th>
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

### Q-মান পরীক্ষা

<table>
  <tr>
    <th align="center">Wilcoxon signed-rank পরীক্ষা</th>
    <th align="center">Wilcoxon signed-rank পরীক্ষা (সংশোধিত)</th>
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
    <th colspan="2" align="center">উন্নতির সম্ভাবনা</th>
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

## পুনরুৎপাদনযোগ্যতা

গভীর রিইনফোর্সমেন্ট লার্নিংয়ের দৈব প্রকৃতির কারণে নির্দিষ্ট ডেটাসেট ব্যবহার করে ফলাফল হুবহু পুনরুৎপাদন করা সম্ভব নয়।
তাই আমাদের পরীক্ষায় ব্যবহৃত র‍্যান্ডম সিডের সেটটি আমরা সরবরাহ করছি।

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

সম্পূর্ণ পরীক্ষা পুনরুৎপাদন:

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

EnvPool-এ Atari এনভায়রনমেন্টের একটি বিস্তৃত সংগ্রহ পাওয়া যায়:
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

Procgen এনভায়রনমেন্ট নিজস্ব `(3, 64, 64)` আকৃতির RGB অবজারভেশন ব্যবহার করে।
Aftab প্রতিটি টাস্কের EnvPool কনফিগারেশন পড়ে এবং কেবল সমর্থিত অপশন প্রয়োগ করে।
তাই `noop`, `frame_skip`, `frame_stack` ও `train_episodic_life`-এর মতো
শুধু Atari-র জন্য প্রযোজ্য অপশন এবং EnvPool-এর রিওয়ার্ড ক্লিপিং Procgen-এ পাঠানো হয় না।

EnvPool-এ Procgen এনভায়রনমেন্টের একটি বিস্তৃত সংগ্রহ পাওয়া যায়:

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## হার্ডওয়্যার

এই প্রকল্পের সব পরীক্ষা [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40) GPU-তে চালানো হয়েছে।

| স্পেসিফিকেশন | বিস্তারিত |
|--------------|----------|
| GPU মেমরি | ত্রুটি-সংশোধন কোডসহ (ECC) 48 GB GDDR6 |
| GPU মেমরি ব্যান্ডউইডথ | 696 GB/s |
| আন্তঃসংযোগ | NVIDIA NVLink 112.5 GB/s (দ্বিমুখী); PCIe Gen4: 64 GB/s |
| NVLink | দ্বিমুখী লো-প্রোফাইল (২ স্লট) |
| ডিসপ্লে পোর্ট | 3x DisplayPort 1.4* |
| সর্বোচ্চ বিদ্যুৎ খরচ | 300 W |
| আকার | 4.4" (উচ্চতা) × 10.5" (দৈর্ঘ্য), দুই স্লট |
| শীতলীকরণ | প্যাসিভ |
| সমর্থিত vGPU সফটওয়্যার | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| সমর্থিত vGPU প্রোফাইল | Virtual GPU লাইসেন্সিং গাইড দেখুন |
| NVENC / NVDEC | 1x / 2x (AV1 ডিকোডিংসহ) |
| সিকিউর বুট | হার্ডওয়্যার রুট অব ট্রাস্টসহ সিকিউর ও মেজার্ড বুট (ঐচ্ছিক) |
| NEBS প্রস্তুতি | স্তর ৩ |
| পাওয়ার কানেক্টর | ৮-পিন CPU |

## উদ্ধৃতি

```bibtex
@article{aftab2026drl,
  title={Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
}
```

### সংশ্লিষ্ট কাজ

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

## দরকারি লিংক

- [উইকিপিডিয়া: রিইনফোর্সমেন্ট লার্নিং (RL)](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [উইকিপিডিয়া: গভীর রিইনফোর্সমেন্ট লার্নিং (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [উইকিপিডিয়া: Q-লার্নিং](https://en.wikipedia.org/wiki/Q-learning)
- [উইকিপিডিয়া: PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [উইকিপিডিয়া: পরিসংখ্যানগত হাইপোথিসিস পরীক্ষা](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [উইকিপিডিয়া: Wilcoxon signed-rank পরীক্ষা](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## লাইসেন্স

© 2025 Taha Shieenavaz.
CC BY-NC 4.0 লাইসেন্সের অধীনে প্রকাশিত: https://creativecommons.org/licenses/by-nc/4.0/
