<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../figures/header-light.svg">
  <img alt="مقالهٔ آفتاب" src="../figures/header-light.svg">
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

## معرفی

**آفتاب** (واژه‌ای <a href="https://en.wikipedia.org/wiki/Aftab">فارسی</a> به معنای «خورشید» یا «پرتو خورشید») چارچوبی برای بنچمارک‌کردن رمزگذارهای مبتنی بر CNN در PQN و در مجموعه‌ای از <a href="https://en.wikipedia.org/wiki/Atari_Games">بازی‌های آتاری</a> است. این چارچوب ابزارهای استانداردی برای آموزش، ارزیابی و بازتولیدپذیری پژوهش‌های یادگیری تقویتی عمیق فراهم می‌کند.

چند ویدئو برای مقایسهٔ عامل‌های PQN و آفتاب آماده کرده‌ایم؛ آن‌ها را [اینجا](../videos.md) ببینید.

### آزمایش‌های رمزگذار

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
      <th>IQM HNS (۵۰ میلیون فریم پایانی)</th>
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

### آزمایش‌های Hadamax

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
      <th>IQM HNS (۵۰ میلیون فریم پایانی)</th>
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

منابع:
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### آزمایش‌های مقدار Q

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
      <th>IQM HNS (۵۰ میلیون فریم پایانی)</th>
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

منابع:
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

## نصب

نصب با pip:

```bash
pip install aftab
```

همچنین می‌توانید مخزن را کلون و آن را در حالت `editable` نصب کنید.

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

برای ساخت محیط مجازی، استفاده از [Micromamba](https://github.com/mamba-org/micromamba-releases) را قویاً توصیه می‌کنیم. راهنمای کامل [اینجا](../scripts/README.md) آمده است.

## آموزش عامل‌ها

**رابط برنامه‌نویسی JAX در حال حاضر در دست توسعه است** و انتظار می‌رود تا پایان سال ۲۰۲۶ تکمیل شود. از مشارکت شما بسیار استقبال می‌کنیم.

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


## افزودن رمزگذار سفارشی

می‌توانید رمزگذار دلخواه خود را به‌صورت یک ماژول PyTorch تعریف کنید و آن را به عامل بدهید:

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## نتایج

**آزمایش‌های رمزگذار**:

- جدول‌ها:
  - [HNS](../results/encoder_experiments/human_normalized_scores.md)
  - [امتیازها](../results/encoder_experiments/scores.md)
- نمودارها:
  - [IQM HNS](https://github.com/tahashieenavaz/aftab/tree/main/figures/encoder_experiments/human_normalized_score)
  - [روند تغییر خطا](https://github.com/tahashieenavaz/aftab/tree/main/figures/encoder_experiments/loss)

**آزمایش‌های Hadamax**:

- جدول‌ها:
  - [HNS](../results/hadamax_experiments/human_normalized_scores.md)
  - [امتیازها](../results/hadamax_experiments/scores.md)
- نمودارها:
  - [IQM HNS](https://github.com/tahashieenavaz/aftab/tree/main/figures/hadamax_experiments/human_normalized_score)
  - [روند تغییر خطا](https://github.com/tahashieenavaz/aftab/tree/main/figures/hadamax_experiments/loss)

**آزمایش‌های مقدار Q**:
- جدول‌ها:
  - [HNS](../results/qvalue_experiments/human_normalized_scores.md)
  - [امتیازها](../results/qvalue_experiments/scores.md)
- نمودارها:
  - [IQM HNS](https://github.com/tahashieenavaz/aftab/tree/main/figures/qvalue_experiments/human_normalized_score)
  - [روند تغییر خطا](https://github.com/tahashieenavaz/aftab/tree/main/figures/qvalue_experiments/loss)

**آزمایش‌های Procgen**:
- جدول‌ها:
  - [PHS](../results/procgen_experiments/procgen_normalized_scores.md)
  - [امتیازها](../results/procgen_experiments/scores.md)


## پیچیدگی مدل

### گونه‌های پایه

| گونه | پارامترهای رمزگذار | پارامترهای سر رگرسیون | کل پارامترها | FLOPs رمزگذار | FLOPs سر رگرسیون | کل FLOPs |
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

> **نکته:** گونهٔ Eta به‌طور محسوسی پارامترهای بیشتری از سایر گونه‌ها دارد؛ دلیل اصلی، تعداد زیاد ویژگی‌هایی است که رمزگذار تولید می‌کند.

---

### گونه‌های Hadamax

| گونه | پارامترهای رمزگذار | پارامترهای سر رگرسیون | کل پارامترها | FLOPs رمزگذار | FLOPs سر رگرسیون | کل FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## ابرپارامترها

<div align="center">

| ابرپارامتر | مقدار |
| :--- | :--- |
| نرخ یادگیری | $2.5 \times 10^{-4}$ |
| تعداد محیط‌های آموزش | 128 |
| تعداد محیط‌های آزمون | 8 |
| بهینه‌ساز | [Rectified Adam](https://arxiv.org/abs/1908.03265) |
| زوال وزن | 0 |
| $\epsilon$ | $1 \times 10^{-5}$ |
| $\beta_{1}$ | 0.9 |
| $\beta_{2}$ | 0.999 |
| کل فریم‌ها | 200,000,000 |
| تابع خطا | میانگین مربعات خطا |
| زمان‌بند | کاهش خطی |
| اکتشاف $\epsilon$-حریصانه | 10% of total frames |
| ضریب تنزیل ($\gamma$) | 0.99 |
| GAE ($\lambda$) | 0.65 |
| دوره‌ها | 2 |
| اندازهٔ دسته | 4096 |

</div>

<p align="center"><em>در آزمایش‌های رمزگذار و Hadamax استفاده شده است.</em></p>

## معناداری آماری

### آزمایش‌های رمزگذار

<table>
  <tr>
    <th align="center">آزمون رتبه علامت‌دار ویلکاکسون</th>
    <th align="center">آزمون رتبه علامت‌دار ویلکاکسون (اصلاح‌شده)</th>
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
    <th colspan="2" align="center">احتمال بهبود</th>
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

### آزمایش‌های Hadamax

<table>
  <tr>
    <th align="center">آزمون رتبه علامت‌دار ویلکاکسون</th>
    <th align="center">آزمون رتبه علامت‌دار ویلکاکسون (اصلاح‌شده)</th>
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
    <th colspan="2" align="center">احتمال بهبود</th>
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

### آزمایش‌های مقدار Q

<table>
  <tr>
    <th align="center">آزمون رتبه علامت‌دار ویلکاکسون</th>
    <th align="center">آزمون رتبه علامت‌دار ویلکاکسون (اصلاح‌شده)</th>
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
    <th colspan="2" align="center">احتمال بهبود</th>
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

## بازتولیدپذیری

به‌دلیل ماهیت تصادفی یادگیری تقویتی عمیق، بازتولید دقیق نتایج صرفاً با مجموعه‌داده‌های ثابت امکان‌پذیر نیست.
در عوض، مجموعهٔ بذرهای تصادفی استفاده‌شده در آزمایش‌ها را ارائه می‌کنیم.

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

بازتولید کامل آزمایش:

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

مجموعهٔ کاملی از محیط‌های آتاری از طریق EnvPool در دسترس است:
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

محیط‌های Procgen از مشاهده‌های RGB بومی خود با شکل `(3, 64, 64)` استفاده می‌کنند.
آفتاب پیکربندی EnvPool هر وظیفه را می‌خواند و فقط گزینه‌های پشتیبانی‌شده را اعمال می‌کند.
بنابراین گزینه‌های ویژهٔ آتاری مانند `noop`، `frame_skip`، `frame_stack` و
`train_episodic_life`، و نیز برش پاداش EnvPool، به Procgen ارسال نمی‌شوند.

مجموعهٔ کاملی از محیط‌های Procgen از طریق EnvPool در دسترس است:

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## سخت‌افزار

تمام آزمایش‌های این پروژه با پردازنده‌های گرافیکی [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40) اجرا شده‌اند.

| مشخصه | جزئیات |
|--------------|----------|
| حافظهٔ GPU | ۴۸ گیگابایت GDDR6 با کد تصحیح خطا (ECC) |
| پهنای باند حافظهٔ GPU | 696 GB/s |
| اتصال داخلی | NVIDIA NVLink با سرعت 112.5 GB/s (دوسویه)؛ PCIe Gen4 با سرعت 64 GB/s |
| NVLink | دوسویه، کم‌ارتفاع (دو اسلات) |
| درگاه‌های نمایش | 3x DisplayPort 1.4* |
| حداکثر مصرف توان | 300 W |
| ابعاد | ۴٫۴ اینچ (ارتفاع) × ۱۰٫۵ اینچ (طول)، دو اسلات |
| خنک‌کاری | غیرفعال |
| نرم‌افزارهای vGPU پشتیبانی‌شده | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| پروفایل‌های vGPU پشتیبانی‌شده | به راهنمای مجوز Virtual GPU مراجعه کنید |
| NVENC / NVDEC | 1x / 2x (شامل رمزگشایی AV1) |
| راه‌اندازی امن | راه‌اندازی امن و اندازه‌گیری‌شده با ریشهٔ اعتماد سخت‌افزاری (اختیاری) |
| آمادگی NEBS | سطح ۳ |
| رابط برق | CPU هشت‌پین |

## شیوهٔ ارجاع

```bibtex
@article{aftab2026drl,
  title={Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
}
```

### آثار مرتبط

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

## پیوندهای مفید

- [ویکی‌پدیا: یادگیری تقویتی (RL)](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [ویکی‌پدیا: یادگیری تقویتی عمیق (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [ویکی‌پدیا: یادگیری Q](https://en.wikipedia.org/wiki/Q-learning)
- [ویکی‌پدیا: PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [ویکی‌پدیا: آزمون فرض آماری](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [ویکی‌پدیا: آزمون رتبه علامت‌دار ویلکاکسون](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## مجوز

© 2025 Taha Shieenavaz.
منتشرشده تحت مجوز CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
