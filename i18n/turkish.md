<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-dark.svg">
  <img alt="Aftab başlığı" src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-light.svg">
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

## Genel Bakış

**Aftab** (<a href="https://en.wikipedia.org/wiki/Aftab">Farsça</a>: آفتاب; “güneş” veya “güneş ışınları”), PQN’de kullanılan CNN tabanlı kodlayıcıları farklı <a href="https://en.wikipedia.org/wiki/Atari_Games">Atari oyunlarında</a> değerlendirmeye yönelik bir kıyaslama çerçevesidir. Derin pekiştirmeli öğrenme araştırmaları için standartlaştırılmış eğitim, değerlendirme ve yeniden üretilebilirlik araçları sunar.

Aftab mimarisinin standart PQN temel modelleriyle karşılaştırmasını bu [video gösterimlerinde](https://github.com/tahashieenavaz/aftab/blob/main/videos.md) izleyin.

Bu araştırma herhangi bir fon alınmadan gerçekleştirildi; çalışmamızı yararlı bulduysanız [GitHub üzerinden sponsor olmayı](https://github.com/sponsors/tahashieenavaz) değerlendirebilirsiniz 💛.

### Kodlayıcı Deneyleri

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
      <th>IQM HNS (Son 50 Milyon Kare)</th>
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

### Hadamax Deneyleri

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
      <th>IQM HNS (Son 50 Milyon Kare)</th>
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

Kaynaklar:
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### Q Değeri Deneyleri

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
      <th>IQM HNS (Son 50 Milyon Kare)</th>
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

Kaynaklar:
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

### Procgen (Aşırı Öğrenmeyi Önleme) Deneyleri

Procgen ortamlarının insan-normalleştirilmiş puanlarını karşılaştıran herkese açık bir kıyaslama bulunmadığından, farklı seed’lerdeki puanlara basit bir min-maks normalleştirmesi uygulayan PNS’yi (Procgen Normalized Score) oluşturduk.

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
      <th>IQM PNS (Son 50 Milyon Kare)</th>
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

## Kurulum

pip ile kurulum:

```bash
pip install aftab
```

Alternatif olarak depoyu klonlayıp `editable` modunda kurabilirsiniz.

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

Sanal ortam oluşturmak için [Micromamba](https://github.com/mamba-org/micromamba-releases) kullanmanızı özellikle öneriyoruz. Ayrıntılı yönergeleri [burada](https://github.com/tahashieenavaz/aftab/blob/main/scripts/README.md) bulabilirsiniz.

## Ajanları Eğitme

**JAX API şu anda geliştiriliyor** ve 2026 sonuna kadar tamamlanması planlanıyor. Katkılarınızı memnuniyetle karşılıyoruz.

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


## Özel Kodlayıcı Ekleme

Kendi kodlayıcınızı bir PyTorch modülü olarak tanımlayıp ajana verebilirsiniz:

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## Sonuçlar

Tüm deney sonuçları deney kategorisine göre düzenlenmiştir. Her bölüm şunları içerir:
- **Tablolar**: sayısal sonuçlar (HNS/PHS ve ham puanlar)
- **Grafikler**: IQM ile normalleştirilmiş puanlar ve eğitim eğrileri

### Kodlayıcı Deneyleri

**Tablolar**
- [İnsan Performansına Göre Normalleştirilmiş Puanlar](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/human_normalized_scores.md)
- [Puanlar](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/scores.md)

**Grafikler**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/human_normalized_score)
- [Kayıp Değişimi](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/loss)

---

### Hadamax Deneyleri

**Tablolar**
- [İnsan Performansına Göre Normalleştirilmiş Puanlar](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/human_normalized_scores.md)
- [Puanlar](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/scores.md)

**Grafikler**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/human_normalized_score)
- [Kayıp Değişimi](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/loss)

---

### Q Değeri Deneyleri

**Tablolar**
- [İnsan Performansına Göre Normalleştirilmiş Puanlar](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/human_normalized_scores.md)
- [Puanlar](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/scores.md)

**Grafikler**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/human_normalized_score)
- [Kayıp Değişimi](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/loss)

---

### Procgen Deneyleri

**Tablolar**
- [Procgen Normalleştirilmiş Puanları](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/procgen_normalized_scores.md)
- [Puanlar](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/scores.md)
- [Seed’e göre PNS AUC](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_seed.md)
- [Oyuna göre PNS AUC](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_game.md)

**Grafikler**
- [IQM PNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/procgen_experiments/procgen_normalized_score)

## Model Karmaşıklığı

### Temel Varyantlar

| Varyant | Kodlayıcı Parametreleri | Regresyon Başlığı Parametreleri | Toplam Parametre | Kodlayıcı FLOP’ları | Regresyon Başlığı FLOP’ları | Toplam FLOP |
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

> **Not:** Eta varyantı, kodlayıcısının çok sayıda öznitelik üretmesi nedeniyle diğer varyantlardan belirgin ölçüde daha fazla parametreye sahiptir.

---

### Hadamax Varyantları

| Varyant | Kodlayıcı Parametreleri | Regresyon Başlığı Parametreleri | Toplam Parametre | Kodlayıcı FLOP’ları | Regresyon Başlığı FLOP’ları | Toplam FLOP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## Hiperparametreler

Aşağıdaki tablolar `Aftab` tarafından tanımlanan varsayılan değerleri gösterir. `experiment_name` argümanı zorunludur ve varsayılan değeri yoktur.

### Eğitim ve Ortam

| Hiperparametre (`Aftab` argümanı) | Varsayılan |
| :--- | :--- |
| Kodlayıcı (`encoder`) | Gamma-Hadamax-Valid |
| Ağ (`network`) | Distributional Bootstrapped (Ensemble) Dueling |
| Toplam kare (`frames`) | 200,000,000 |
| Kare atlama (`frame_skip`) | 4 |
| Kare yığını (`frame_stack`) | 4 |
| Azami no-op (`noop`) | 30 |
| Öğrenme oranı (`lr`) | $2.5 \times 10^{-4}$ |
| Eğitim ortamları (`train_environments`) | 128 |
| Test ortamları (`test_environments`) | 8 |
| Güncelleme başına adım (`steps_per_update`) | 32 |
| Yığın boyutu (türetilmiş) | 4,096 |
| Mini yığınlar (`mini_batches`) | 32 |
| Mini yığın boyutu (türetilmiş) | 128 |
| İndirim faktörü ($\gamma$) | 0.99 |
| Dönüş $\lambda$’sı (`return_lambda`) | 0.65 |
| Dönem sayısı (`epochs`) | 2 |
| Gradyan normu (`gradient_norm`) | 10.0 |
| Gömme boyutu (`embedding_dimension`) | 512 |
| Eğitim bölüm ömrü (`train_episodic_life`) | `True` |
| Test bölüm ömrü (`test_episodic_life`) | `False` |
| Eğitim ödülü kırpma (`train_reward_clip`) | `True` |
| Test ödülü kırpma (`test_reward_clip`) | `True` |
| Epsilon zamanlaması | Doğrusal |
| Epsilon azaltma oranı | 10% |

### Optimizasyon Algoritması

| Hiperparametre (`Aftab` argümanı) | Varsayılan |
| :--- | :--- |
| Optimizasyon algoritması (`optimizer`) | [Rectified Adam](https://arxiv.org/abs/1908.03265) (`"radam"`) |
| Epsilon (`optimizer_epsilon`) | $1 \times 10^{-5}$ |
| Ağırlık azalması (`optimizer_weight_decay`) | 0.0 |
| $\beta_1$ (`optimizer_first_beta`) | 0.9 |
| $\beta_2$ (`optimizer_second_beta`) | 0.999 |

### Dağılımsal ve Bootstrapped (Ensemble) Q Değerleri

| Hiperparametre (`Aftab` argümanı) | Varsayılan |
| :--- | :--- |
| Dağılımsal kutular (`distributional_bins`) | 51 |
| Dağılımsal en küçük değer (`distributional_min_value`) | -10.0 |
| Dağılımsal en büyük değer (`distributional_max_value`) | 10.0 |
| Dağılımsal sigma (`distributional_sigma`) | `None` (sigma oranından türetilir) |
| Dağılımsal sigma oranı (`distributional_sigma_ratio`) | 0.75 |
| Dağılımsal değer kırpma (`distributional_value_clip`) | 0.0 |
| Bootstrap başlıkları (`bootstrap_heads`) | 10 |
| Bootstrap olasılığı (`bootstrap_probability`) | 1.0 |

### Procgen Geçersiz Kılmaları

| Hiperparametre | Varsayılan | Procgen |
| :--- | :--- | :--- |
| Eğitim ortamları | 128 | 64 (`procgen_train_environments`) |
| Güncelleme başına adım | 32 | 256 (`procgen_steps_per_update`) |
| Yığın boyutu | 4,096 | 16,384 |
| Mini yığın boyutu | 128 | 512 |

<em>Procgen ortamlarında Aftab yukarıdaki iki geçersiz kılmayı otomatik olarak uygular; diğer varsayılanlar değişmez.</em>

## İstatistiksel Anlamlılık

### Kodlayıcı Deneyleri

<table>
  <tr>
    <th align="center">Wilcoxon İşaretli Sıralar Testi</th>
    <th align="center">Wilcoxon İşaretli Sıralar Testi (Düzeltilmiş)</th>
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
    <th colspan="2" align="center">İyileşme Olasılığı</th>
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

### Hadamax Deneyleri

<table>
  <tr>
    <th align="center">Wilcoxon İşaretli Sıralar Testi</th>
    <th align="center">Wilcoxon İşaretli Sıralar Testi (Düzeltilmiş)</th>
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
    <th colspan="2" align="center">İyileşme Olasılığı</th>
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

### Q Değeri Deneyleri

<table>
  <tr>
    <th align="center">Wilcoxon İşaretli Sıralar Testi</th>
    <th align="center">Wilcoxon İşaretli Sıralar Testi (Düzeltilmiş)</th>
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
    <th colspan="2" align="center">İyileşme Olasılığı</th>
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

## Yeniden Üretilebilirlik

Derin pekiştirmeli öğrenmenin stokastik yapısı nedeniyle yalnızca sabit veri kümeleriyle sonuçları birebir yeniden üretmek mümkün değildir.
Bunun yerine deneylerimizde kullandığımız rastgele sayı üreteci tohumlarını sağlıyoruz.

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

Deneyleri eksiksiz yeniden üretme:

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

EnvPool üzerinden kapsamlı bir Atari ortamları koleksiyonuna ulaşabilirsiniz:
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

Procgen ortamları, `(3, 64, 64)` biçimindeki yerel RGB gözlemlerini kullanır.
Aftab her görevin EnvPool yapılandırmasını okur ve yalnızca desteklenen seçenekleri uygular.
Bu nedenle `noop`, `frame_skip`, `frame_stack` ve `train_episodic_life` gibi
Atari’ye özgü seçenekler ile EnvPool ödül kırpma ayarı Procgen’e aktarılmaz.

EnvPool üzerinden kapsamlı bir Procgen ortamları koleksiyonuna ulaşabilirsiniz:

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## Donanım

Bu projedeki tüm deneyler [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40) GPU’lar üzerinde çalıştırılmıştır.

| Özellik | Ayrıntılar |
|--------------|----------|
| GPU belleği | Hata düzeltme kodlu (ECC) 48 GB GDDR6 |
| GPU bellek bant genişliği | 696 GB/s |
| Ara bağlantı | NVIDIA NVLink 112,5 GB/s (çift yönlü); PCIe Gen4: 64 GB/s |
| NVLink | Çift yönlü, düşük profil (2 yuva) |
| Görüntü bağlantı noktaları | 3x DisplayPort 1.4* |
| En yüksek güç tüketimi | 300 W |
| Fiziksel boyut | 4,4" (Y) × 10,5" (U), çift yuva |
| Soğutma | Pasif |
| Desteklenen vGPU yazılımları | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| Desteklenen vGPU profilleri | Virtual GPU Lisanslama Kılavuzu’na bakın |
| NVENC / NVDEC | 1x / 2x (AV1 kod çözme dâhil) |
| Güvenli önyükleme | Donanım güven köküyle güvenli ve ölçümlü önyükleme (isteğe bağlı) |
| NEBS uyumluluğu | Seviye 3 |
| Güç bağlantısı | 8 pimli CPU |

## Atıf

Depo:

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

Ön baskı:

```bibtex
@misc{2608.07335,
  Author = {Taha Shieenavaz and Shabnam Zareshahraki and Loris Nanni},
  Title = {Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  Year = {2026},
  Eprint = {arXiv:2608.07335},
}
```

### İlgili Çalışmalar

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

## Yararlı Bağlantılar

- [Vikipedi: Pekiştirmeli Öğrenme (RL)](https://tr.wikipedia.org/wiki/Pekiştirmeli_öğrenme)
- [Vikipedi: Derin Pekiştirmeli Öğrenme (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Vikipedi: Q-Öğrenme](https://en.wikipedia.org/wiki/Q-learning)
- [Vikipedi: PyTorch](https://tr.wikipedia.org/wiki/PyTorch)
- [Vikipedi: İstatistiksel Hipotez Testi](https://tr.wikipedia.org/wiki/Hipotez_testi)
- [Vikipedi: Wilcoxon İşaretli Sıralar Testi](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## Yazı Tipi

Vazirmatn yazı tipi, GitHub deposunun başlığında ve projenin açılış sayfasında hem Farsça hem de İngilizce metinler için kullanılır.

[GitHub](https://github.com/rastikerdar/vazirmatn) | [Google Fonts](https://fonts.google.com/specimen/Vazirmatn)

## Lisans

© 2025 Taha Shieenavaz.
CC BY-NC 4.0 lisansı altında sunulmaktadır: https://creativecommons.org/licenses/by-nc/4.0/
