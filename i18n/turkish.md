<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../figures/header-light.svg">
  <img alt="Aftab makalesi" src="../figures/header-light.svg">
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

<br />

## Genel Bakış

**Aftab** (<a href="https://en.wikipedia.org/wiki/Aftab">Farsça</a>: آفتاب; “güneş” veya “güneş ışınları”), PQN’de kullanılan CNN tabanlı kodlayıcıları farklı <a href="https://en.wikipedia.org/wiki/Atari_Games">Atari oyunlarında</a> değerlendirmeye yönelik bir kıyaslama çerçevesidir. Derin pekiştirmeli öğrenme araştırmaları için standartlaştırılmış eğitim, değerlendirme ve yeniden üretilebilirlik araçları sunar.

PQN ve Aftab ajanlarını karşılaştıran birkaç video hazırladık. Videoları [buradan](../videos.md) izleyebilirsiniz.

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

Sanal ortam oluşturmak için [Micromamba](https://github.com/mamba-org/micromamba-releases) kullanmanızı özellikle öneriyoruz. Ayrıntılı yönergeleri [burada](../scripts/README.md) bulabilirsiniz.

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
- [İnsan Performansına Göre Normalleştirilmiş Puanlar](../results/encoder_experiments/human_normalized_scores.md)
- [Puanlar](../results/encoder_experiments/scores.md)

**Grafikler**
- [IQM HNS](../figures/encoder_experiments/human_normalized_score)
- [Kayıp Değişimi](../figures/encoder_experiments/loss)

---

### Hadamax Deneyleri

**Tablolar**
- [İnsan Performansına Göre Normalleştirilmiş Puanlar](../results/hadamax_experiments/human_normalized_scores.md)
- [Puanlar](../results/hadamax_experiments/scores.md)

**Grafikler**
- [IQM HNS](../figures/hadamax_experiments/human_normalized_score)
- [Kayıp Değişimi](../figures/hadamax_experiments/loss)

---

### Q Değeri Deneyleri

**Tablolar**
- [İnsan Performansına Göre Normalleştirilmiş Puanlar](../results/qvalue_experiments/human_normalized_scores.md)
- [Puanlar](../results/qvalue_experiments/scores.md)

**Grafikler**
- [IQM HNS](../figures/qvalue_experiments/human_normalized_score)
- [Kayıp Değişimi](../figures/qvalue_experiments/loss)

---

### Procgen Deneyleri

**Tablolar**
- [Procgen Normalleştirilmiş Puanları](../results/procgen_experiments/procgen_normalized_scores.md)
- [Puanlar](../results/procgen_experiments/scores.md)


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

<div align="center">

| Hiperparametre | Değer |
| :--- | :--- |
| Öğrenme oranı | $2.5 \times 10^{-4}$ |
| Eğitim ortamları | 128 |
| Test ortamları | 8 |
| Optimizasyon algoritması | [Rectified Adam](https://arxiv.org/abs/1908.03265) |
| Ağırlık azalması | 0 |
| $\epsilon$ | $1 \times 10^{-5}$ |
| $\beta_{1}$ | 0.9 |
| $\beta_{2}$ | 0.999 |
| Toplam kare | 200,000,000 |
| Kayıp fonksiyonu | Ortalama Kare Hatası |
| Zamanlayıcı | Doğrusal azaltma |
| $\epsilon$-açgözlü keşif | 10% of total frames |
| İndirim faktörü ($\gamma$) | 0.99 |
| GAE ($\lambda$) | 0.65 |
| Dönem sayısı | 2 |
| Yığın boyutu | 4096 |

</div>

<p align="center"><em>Kodlayıcı ve Hadamax deneylerinde kullanılmıştır.</em></p>

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

```bibtex
@article{aftab2026drl,
  title={Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
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

- [Vikipedi: Pekiştirmeli Öğrenme (RL)](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [Vikipedi: Derin Pekiştirmeli Öğrenme (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Vikipedi: Q-Öğrenme](https://en.wikipedia.org/wiki/Q-learning)
- [Vikipedi: PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [Vikipedi: İstatistiksel Hipotez Testi](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [Vikipedi: Wilcoxon İşaretli Sıralar Testi](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## Lisans

© 2025 Taha Shieenavaz.
CC BY-NC 4.0 lisansı altında sunulmaktadır: https://creativecommons.org/licenses/by-nc/4.0/
