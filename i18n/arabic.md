<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../figures/header-light.svg">
  <img alt="ورقة Aftab البحثية" src="../figures/header-light.svg">
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

## نظرة عامة

**Aftab** (بالفارسية <a href="https://en.wikipedia.org/wiki/Aftab">آفتاب</a>، وتعني «الشمس» أو «أشعة الشمس») هو إطار معياري لتقييم المُرمِّزات القائمة على الشبكات العصبية الالتفافية (CNN) في PQN عبر مجموعة من <a href="https://en.wikipedia.org/wiki/Atari_Games">ألعاب Atari</a>. ويوفر أدوات موحّدة للتدريب والتقييم وقابلية إعادة الإنتاج في أبحاث التعلم المعزز العميق.

جمعنا بعض المقاطع التي تقارن بين وكلاء PQN وAftab. يمكن مشاهدتها [هنا](../videos.md).

### تجارب المُرمِّزات

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
      <th>IQM HNS (آخر 50 مليون إطار)</th>
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

### تجارب Hadamax

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
      <th>IQM HNS (آخر 50 مليون إطار)</th>
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

المراجع:
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### تجارب قيم Q

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
      <th>IQM HNS (آخر 50 مليون إطار)</th>
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

المراجع:
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

### تجارب Procgen (الحد من فرط التخصيص)

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
      <th>IQM PNS (آخر 50 مليون إطار)</th>
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

## التثبيت

التثبيت باستخدام pip:

```bash
pip install aftab
```

يمكنك بدلاً من ذلك استنساخ المستودع وتثبيته في وضع `editable`.

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

نوصي بشدة باستخدام [Micromamba](https://github.com/mamba-org/micromamba-releases) لإنشاء البيئات الافتراضية. تتوفر التعليمات المفصلة [هنا](../scripts/README.md).

## تدريب الوكلاء

**واجهة JAX البرمجية قيد التطوير حالياً**، ومن المخطط إكمالها بحلول نهاية عام 2026. نرحب كثيراً بالمساهمات.

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


## إضافة مُرمِّز مخصص

يمكنك تعريف مُرمِّزك الخاص بوصفه وحدة PyTorch وتمريره إلى الوكيل:

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## النتائج

نُظمت جميع النتائج التجريبية بحسب فئة التجربة. يتضمن كل قسم ما يلي:
- **الجداول**: النتائج العددية (HNS/PHS والدرجات الخام)
- **المخططات**: الدرجات المطبّعة باستخدام IQM ومنحنيات التدريب

### تجارب المُرمِّزات

**الجداول**
- [الدرجات المطبّعة قياساً إلى أداء الإنسان](../results/encoder_experiments/human_normalized_scores.md)
- [الدرجات](../results/encoder_experiments/scores.md)

**المخططات**
- [IQM HNS](../figures/encoder_experiments/human_normalized_score)
- [تطور دالة الخسارة](../figures/encoder_experiments/loss)

---

### تجارب Hadamax

**الجداول**
- [الدرجات المطبّعة قياساً إلى أداء الإنسان](../results/hadamax_experiments/human_normalized_scores.md)
- [الدرجات](../results/hadamax_experiments/scores.md)

**المخططات**
- [IQM HNS](../figures/hadamax_experiments/human_normalized_score)
- [تطور دالة الخسارة](../figures/hadamax_experiments/loss)

---

### تجارب قيم Q

**الجداول**
- [الدرجات المطبّعة قياساً إلى أداء الإنسان](../results/qvalue_experiments/human_normalized_scores.md)
- [الدرجات](../results/qvalue_experiments/scores.md)

**المخططات**
- [IQM HNS](../figures/qvalue_experiments/human_normalized_score)
- [تطور دالة الخسارة](../figures/qvalue_experiments/loss)

---

### تجارب Procgen

**الجداول**
- [درجات Procgen المطبّعة](../results/procgen_experiments/procgen_normalized_scores.md)
- [الدرجات](../results/procgen_experiments/scores.md)


## تعقيد النماذج

### المتغيرات الأساسية

| المتغير | معاملات المُرمِّز | معاملات رأس الانحدار | إجمالي المعاملات | FLOPs المُرمِّز | FLOPs رأس الانحدار | إجمالي FLOPs |
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

> **ملاحظة:** يحتوي متغير Eta على معاملات أكثر بكثير من المتغيرات الأخرى، ويرجع ذلك أساساً إلى أن مُرمِّزه ينتج عدداً كبيراً من السمات.

---

### متغيرات Hadamax

| المتغير | معاملات المُرمِّز | معاملات رأس الانحدار | إجمالي المعاملات | FLOPs المُرمِّز | FLOPs رأس الانحدار | إجمالي FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## المعاملات الفائقة

<div align="center">

| المعامل الفائق | القيمة |
| :--- | :--- |
| معدل التعلم | $2.5 \times 10^{-4}$ |
| بيئات التدريب | 128 |
| بيئات الاختبار | 8 |
| خوارزمية التحسين | [Rectified Adam](https://arxiv.org/abs/1908.03265) |
| اضمحلال الأوزان | 0 |
| $\epsilon$ | $1 \times 10^{-5}$ |
| $\beta_{1}$ | 0.9 |
| $\beta_{2}$ | 0.999 |
| إجمالي الإطارات | 200,000,000 |
| دالة الخسارة | متوسط مربع الخطأ |
| المجدول | خفض خطي |
| استكشاف $\epsilon$-الجشع | 10% of total frames |
| معامل الخصم ($\gamma$) | 0.99 |
| GAE ($\lambda$) | 0.65 |
| الحقب | 2 |
| حجم الدفعة | 4096 |

</div>

<p align="center"><em>مستخدمة في تجارب المُرمِّزات وHadamax.</em></p>

## الدلالة الإحصائية

### تجارب المُرمِّزات

<table>
  <tr>
    <th align="center">اختبار ويلكوكسون للرتب الموقعة</th>
    <th align="center">اختبار ويلكوكسون للرتب الموقعة (بعد التصحيح)</th>
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
    <th colspan="2" align="center">احتمال التحسن</th>
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

### تجارب Hadamax

<table>
  <tr>
    <th align="center">اختبار ويلكوكسون للرتب الموقعة</th>
    <th align="center">اختبار ويلكوكسون للرتب الموقعة (بعد التصحيح)</th>
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
    <th colspan="2" align="center">احتمال التحسن</th>
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

### تجارب قيم Q

<table>
  <tr>
    <th align="center">اختبار ويلكوكسون للرتب الموقعة</th>
    <th align="center">اختبار ويلكوكسون للرتب الموقعة (بعد التصحيح)</th>
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
    <th colspan="2" align="center">احتمال التحسن</th>
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

## قابلية إعادة الإنتاج

نظراً إلى الطبيعة العشوائية للتعلم المعزز العميق، لا يمكن إعادة إنتاج النتائج بدقة بالاعتماد على مجموعات بيانات ثابتة.
لذلك نوفر مجموعة البذور العشوائية المستخدمة في تجاربنا.

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

إعادة إنتاج التجارب بالكامل:

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

تتوفر عبر EnvPool مجموعة شاملة من بيئات Atari:
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

تستخدم بيئات Procgen ملاحظاتها الأصلية بنظام RGB وبالشكل `(3, 64, 64)`.
يقرأ Aftab إعدادات EnvPool لكل مهمة ولا يطبق إلا الخيارات المدعومة.
لذلك لا تُمرر إلى Procgen الخيارات الخاصة بـAtari، مثل `noop` و`frame_skip` و`frame_stack`
و`train_episodic_life`، ولا قص المكافآت في EnvPool.

تتوفر عبر EnvPool مجموعة شاملة من بيئات Procgen:

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## العتاد

شُغلت جميع تجارب هذا المشروع على وحدات معالجة رسومية [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40).

| المواصفة | التفاصيل |
|--------------|----------|
| ذاكرة GPU | ‏48 GB من GDDR6 مع رمز تصحيح الأخطاء (ECC) |
| عرض نطاق ذاكرة GPU | 696 GB/s |
| الربط البيني | NVIDIA NVLink بسرعة 112.5 GB/s ثنائية الاتجاه؛ PCIe Gen4 بسرعة 64 GB/s |
| NVLink | ثنائي الاتجاه، منخفض الارتفاع (فتحتان) |
| منافذ العرض | 3x DisplayPort 1.4* |
| الحد الأقصى لاستهلاك الطاقة | 300 W |
| الأبعاد | ‏4.4 بوصة (ارتفاع) × 10.5 بوصة (طول)، بفتحتين |
| التبريد | سلبي |
| برمجيات vGPU المدعومة | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| ملفات تعريف vGPU المدعومة | راجع دليل ترخيص Virtual GPU |
| NVENC / NVDEC | ‏1x / 2x (يشمل فك ترميز AV1) |
| الإقلاع الآمن | إقلاع آمن ومقاس بجذر ثقة عتادي (اختياري) |
| التوافق مع NEBS | المستوى 3 |
| موصل الطاقة | CPU بثمانية سنون |

## الاستشهاد

```bibtex
@article{aftab2026drl,
  title={Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
}
```

### أعمال ذات صلة

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

## روابط مفيدة

- [ويكيبيديا: التعلم المعزز (RL)](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [ويكيبيديا: التعلم المعزز العميق (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [ويكيبيديا: تعلم Q](https://en.wikipedia.org/wiki/Q-learning)
- [ويكيبيديا: PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [ويكيبيديا: اختبار الفرضيات الإحصائية](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [ويكيبيديا: اختبار ويلكوكسون للرتب الموقعة](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## الترخيص

© 2025 Taha Shieenavaz.
مرخّص بموجب CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
