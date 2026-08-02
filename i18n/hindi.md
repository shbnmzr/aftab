<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../figures/header-light.svg">
  <img alt="आफ़ताब शोधपत्र" src="../figures/header-light.svg">
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

## परिचय

**आफ़ताब** (<a href="https://en.wikipedia.org/wiki/Aftab">फ़ारसी</a>: آفتاب, जिसका अर्थ “सूर्य” या “सूर्य की किरणें” है) विभिन्न <a href="https://en.wikipedia.org/wiki/Atari_Games">Atari गेमों</a> में PQN के CNN-आधारित एनकोडरों के मूल्यांकन के लिए एक बेंचमार्किंग फ्रेमवर्क है। यह डीप रीइन्फोर्समेंट लर्निंग अनुसंधान के लिए प्रशिक्षण, मूल्यांकन और पुनरुत्पादकता के मानकीकृत टूल उपलब्ध कराता है।

हमने PQN और आफ़ताब एजेंटों की तुलना करने वाले कुछ वीडियो तैयार किए हैं। उन्हें [यहाँ](../videos.md) देखें।

### एनकोडर प्रयोग

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
      <th>IQM HNS (अंतिम 5 करोड़ फ़्रेम)</th>
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

### Hadamax प्रयोग

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
      <th>IQM HNS (अंतिम 5 करोड़ फ़्रेम)</th>
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

संदर्भ:
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### Q-वैल्यू प्रयोग

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
      <th>IQM HNS (अंतिम 5 करोड़ फ़्रेम)</th>
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

संदर्भ:
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

## इंस्टॉलेशन

pip से इंस्टॉल करें:

```bash
pip install aftab
```

इसके अलावा, आप रिपॉज़िटरी को क्लोन करके `editable` मोड में इंस्टॉल कर सकते हैं।

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

वर्चुअल एनवायरनमेंट बनाने के लिए हम [Micromamba](https://github.com/mamba-org/micromamba-releases) के उपयोग की पुरज़ोर सलाह देते हैं। विस्तृत निर्देश [यहाँ](../scripts/README.md) उपलब्ध हैं।

## एजेंटों को प्रशिक्षित करना

**JAX API अभी विकासाधीन है** और इसे 2026 के अंत तक पूरा करने की योजना है। योगदान का हार्दिक स्वागत है।

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


## कस्टम एनकोडर जोड़ना

आप अपना एनकोडर PyTorch मॉड्यूल के रूप में परिभाषित करके एजेंट को दे सकते हैं:

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## परिणाम

**एनकोडर प्रयोग**:

- तालिकाएँ:
  - [HNS](../results/encoder_experiments/human_normalized_scores.md)
  - [स्कोर](../results/encoder_experiments/scores.md)
- चार्ट:
  - [IQM HNS](https://github.com/tahashieenavaz/aftab/tree/main/figures/encoder_experiments/human_normalized_score)
  - [लॉस में बदलाव](https://github.com/tahashieenavaz/aftab/tree/main/figures/encoder_experiments/loss)

**Hadamax प्रयोग**:

- तालिकाएँ:
  - [HNS](../results/hadamax_experiments/human_normalized_scores.md)
  - [स्कोर](../results/hadamax_experiments/scores.md)
- चार्ट:
  - [IQM HNS](https://github.com/tahashieenavaz/aftab/tree/main/figures/hadamax_experiments/human_normalized_score)
  - [लॉस में बदलाव](https://github.com/tahashieenavaz/aftab/tree/main/figures/hadamax_experiments/loss)

**Q-वैल्यू प्रयोग**:
- तालिकाएँ:
  - [HNS](../results/qvalue_experiments/human_normalized_scores.md)
  - [स्कोर](../results/qvalue_experiments/scores.md)
- चार्ट:
  - [IQM HNS](https://github.com/tahashieenavaz/aftab/tree/main/figures/qvalue_experiments/human_normalized_score)
  - [लॉस में बदलाव](https://github.com/tahashieenavaz/aftab/tree/main/figures/qvalue_experiments/loss)

**Procgen प्रयोग**:
- तालिकाएँ:
  - [PHS](../results/procgen_experiments/procgen_normalized_scores.md)
  - [स्कोर](../results/procgen_experiments/scores.md)


## मॉडल की जटिलता

### आधारभूत वेरिएंट

| वेरिएंट | एनकोडर पैरामीटर | रिग्रेशन हेड पैरामीटर | कुल पैरामीटर | एनकोडर FLOPs | रिग्रेशन हेड FLOPs | कुल FLOPs |
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

> **ध्यान दें:** Eta वेरिएंट में अन्य वेरिएंटों की तुलना में काफ़ी अधिक पैरामीटर हैं। इसका मुख्य कारण एनकोडर द्वारा बड़ी संख्या में फ़ीचर तैयार करना है।

---

### Hadamax वेरिएंट

| वेरिएंट | एनकोडर पैरामीटर | रिग्रेशन हेड पैरामीटर | कुल पैरामीटर | एनकोडर FLOPs | रिग्रेशन हेड FLOPs | कुल FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## हाइपरपैरामीटर

<div align="center">

| हाइपरपैरामीटर | मान |
| :--- | :--- |
| लर्निंग रेट | $2.5 \times 10^{-4}$ |
| प्रशिक्षण एनवायरनमेंट | 128 |
| परीक्षण एनवायरनमेंट | 8 |
| ऑप्टिमाइज़र | [Rectified Adam](https://arxiv.org/abs/1908.03265) |
| वेट डिके | 0 |
| $\epsilon$ | $1 \times 10^{-5}$ |
| $\beta_{1}$ | 0.9 |
| $\beta_{2}$ | 0.999 |
| कुल फ़्रेम | 200,000,000 |
| लॉस फ़ंक्शन | माध्य वर्ग त्रुटि |
| शेड्यूलर | रैखिक एनीलिंग |
| $\epsilon$-ग्रीडी एक्सप्लोरेशन | 10% of total frames |
| डिस्काउंट फ़ैक्टर ($\gamma$) | 0.99 |
| GAE ($\lambda$) | 0.65 |
| एपोक | 2 |
| बैच आकार | 4096 |

</div>

<p align="center"><em>एनकोडर और Hadamax प्रयोगों में उपयोग किया गया।</em></p>

## सांख्यिकीय सार्थकता

### एनकोडर प्रयोग

<table>
  <tr>
    <th align="center">Wilcoxon साइन्ड-रैंक परीक्षण</th>
    <th align="center">Wilcoxon साइन्ड-रैंक परीक्षण (संशोधित)</th>
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
    <th colspan="2" align="center">सुधार की संभावना</th>
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

### Hadamax प्रयोग

<table>
  <tr>
    <th align="center">Wilcoxon साइन्ड-रैंक परीक्षण</th>
    <th align="center">Wilcoxon साइन्ड-रैंक परीक्षण (संशोधित)</th>
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
    <th colspan="2" align="center">सुधार की संभावना</th>
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

### Q-वैल्यू प्रयोग

<table>
  <tr>
    <th align="center">Wilcoxon साइन्ड-रैंक परीक्षण</th>
    <th align="center">Wilcoxon साइन्ड-रैंक परीक्षण (संशोधित)</th>
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
    <th colspan="2" align="center">सुधार की संभावना</th>
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

## पुनरुत्पादकता

डीप रीइन्फोर्समेंट लर्निंग की स्टोकैस्टिक प्रकृति के कारण केवल स्थिर डेटासेट से परिणामों को हूबहू पुनरुत्पादित करना संभव नहीं है।
इसलिए हम अपने प्रयोगों में उपयोग किए गए रैंडम सीड उपलब्ध कराते हैं।

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

पूरे प्रयोग को दोहराना:

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

EnvPool में Atari एनवायरनमेंट का एक व्यापक संग्रह उपलब्ध है:
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

Procgen एनवायरनमेंट अपने मूल RGB ऑब्ज़र्वेशन का उपयोग करते हैं, जिनका आकार `(3, 64, 64)` है।
आफ़ताब हर टास्क का EnvPool कॉन्फ़िगरेशन पढ़ता है और केवल समर्थित विकल्प लागू करता है।
इसलिए `noop`, `frame_skip`, `frame_stack` और `train_episodic_life` जैसे
केवल Atari के विकल्प तथा EnvPool की रिवॉर्ड क्लिपिंग Procgen को नहीं भेजे जाते।

EnvPool में Procgen एनवायरनमेंट का एक व्यापक संग्रह उपलब्ध है:

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## हार्डवेयर

इस प्रोजेक्ट के सभी प्रयोग [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40) GPU पर चलाए गए।

| विशिष्टता | विवरण |
|--------------|----------|
| GPU मेमोरी | त्रुटि-सुधार कोड (ECC) के साथ 48 GB GDDR6 |
| GPU मेमोरी बैंडविड्थ | 696 GB/s |
| इंटरकनेक्ट | NVIDIA NVLink 112.5 GB/s (दोनों दिशाओं में); PCIe Gen4: 64 GB/s |
| NVLink | दोतरफ़ा लो-प्रोफ़ाइल (2 स्लॉट) |
| डिस्प्ले पोर्ट | 3x DisplayPort 1.4* |
| अधिकतम बिजली खपत | 300 W |
| आकार | 4.4" (ऊँचाई) × 10.5" (लंबाई), दो स्लॉट |
| कूलिंग | पैसिव |
| समर्थित vGPU सॉफ़्टवेयर | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| समर्थित vGPU प्रोफ़ाइल | Virtual GPU लाइसेंसिंग गाइड देखें |
| NVENC / NVDEC | 1x / 2x (AV1 डिकोडिंग सहित) |
| सिक्योर बूट | हार्डवेयर रूट ऑफ़ ट्रस्ट के साथ सिक्योर और मेज़र्ड बूट (वैकल्पिक) |
| NEBS अनुपालन | स्तर 3 |
| पावर कनेक्टर | 8-पिन CPU |

## उद्धरण

```bibtex
@article{aftab2026drl,
  title={Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
}
```

### संबंधित कार्य

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

## उपयोगी लिंक

- [विकिपीडिया: रीइन्फोर्समेंट लर्निंग (RL)](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [विकिपीडिया: डीप रीइन्फोर्समेंट लर्निंग (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [विकिपीडिया: Q-लर्निंग](https://en.wikipedia.org/wiki/Q-learning)
- [विकिपीडिया: PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [विकिपीडिया: सांख्यिकीय परिकल्पना परीक्षण](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [विकिपीडिया: Wilcoxon साइन्ड-रैंक परीक्षण](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## लाइसेंस

© 2025 Taha Shieenavaz.
CC BY-NC 4.0 के अंतर्गत लाइसेंस प्राप्त: https://creativecommons.org/licenses/by-nc/4.0/
