<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-dark.svg">
  <img alt="Header Aftab" src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-light.svg">
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

<br />

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

## Gambaran umum

**Aftab** (dari <a href="https://en.wikipedia.org/wiki/Aftab">bahasa Persia</a> آفتاب, yang berarti “matahari” atau “sinar matahari”) adalah framework benchmark untuk mengevaluasi encoder berbasis CNN pada PQN di berbagai <a href="https://en.wikipedia.org/wiki/Atari_Games">gim Atari</a>. Framework ini menyediakan perangkat standar untuk pelatihan, evaluasi, dan reproduksibilitas dalam riset deep reinforcement learning.

Lihat perbandingan arsitektur Aftab dengan baseline PQN standar dalam [demonstrasi video](https://github.com/tahashieenavaz/aftab/blob/main/videos.md) ini.

Riset ini dilakukan tanpa menerima pendanaan apa pun; jika karya kami bermanfaat, pertimbangkan untuk [mensponsori melalui GitHub](https://github.com/sponsors/tahashieenavaz) 💛.

### Eksperimen encoder

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
      <th>IQM HNS (50 juta frame terakhir)</th>
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

### Eksperimen Hadamax

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
      <th>IQM HNS (50 juta frame terakhir)</th>
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

Referensi:
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### Eksperimen nilai Q

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
      <th>IQM HNS (50 juta frame terakhir)</th>
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

Referensi:
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

### Eksperimen Procgen (pencegahan overfitting)

Karena belum ada benchmark publik yang membandingkan skor ternormalisasi manusia pada lingkungan Procgen, kami membuat PNS (Procgen Normalized Score), yaitu normalisasi min-maks sederhana atas skor lintas seed.

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
      <th>IQM PNS (50 juta frame terakhir)</th>
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

## Instalasi

Instal dengan pip:

```bash
pip install aftab
```

Sebagai alternatif, Anda dapat mengkloning repositori dan menginstalnya dalam mode `editable`.

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

Kami sangat menyarankan penggunaan [Micromamba](https://github.com/mamba-org/micromamba-releases) untuk membuat lingkungan virtual. Petunjuk lengkap tersedia [di sini](https://github.com/tahashieenavaz/aftab/blob/main/scripts/README.md).

## Melatih agen

**API JAX saat ini masih dalam tahap pengembangan** dan ditargetkan selesai pada akhir 2026. Kontribusi sangat kami harapkan.

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


## Menambahkan encoder khusus

Anda dapat mendefinisikan encoder sendiri sebagai modul PyTorch lalu meneruskannya ke agen:

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## Hasil

Semua hasil eksperimen disusun berdasarkan kategori eksperimen. Setiap bagian memuat:
- **Tabel**: hasil numerik (HNS/PHS dan skor mentah)
- **Grafik**: skor yang dinormalisasi dengan IQM dan kurva pelatihan

### Eksperimen encoder

**Tabel**
- [Skor yang dinormalisasi terhadap performa manusia](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/human_normalized_scores.md)
- [Skor](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/scores.md)

**Grafik**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/human_normalized_score)
- [Perkembangan loss](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/loss)

---

### Eksperimen Hadamax

**Tabel**
- [Skor yang dinormalisasi terhadap performa manusia](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/human_normalized_scores.md)
- [Skor](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/scores.md)

**Grafik**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/human_normalized_score)
- [Perkembangan loss](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/loss)

---

### Eksperimen nilai Q

**Tabel**
- [Skor yang dinormalisasi terhadap performa manusia](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/human_normalized_scores.md)
- [Skor](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/scores.md)

**Grafik**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/human_normalized_score)
- [Perkembangan loss](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/loss)

---

### Eksperimen Procgen

**Tabel**
- [Skor Procgen yang dinormalisasi](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/procgen_normalized_scores.md)
- [Skor](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/scores.md)
- [AUC PNS per seed](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_seed.md)
- [AUC PNS per gim](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_game.md)

**Grafik**
- [IQM PNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/procgen_experiments/procgen_normalized_score)

## Kompleksitas model

### Varian dasar

| Varian | Parameter encoder | Parameter head regresi | Total parameter | FLOPs encoder | FLOPs head regresi | Total FLOPs |
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

> **Catatan:** varian Eta memiliki parameter jauh lebih banyak daripada varian lainnya, terutama karena encodernya menghasilkan fitur dalam jumlah besar.

---

### Varian Hadamax

| Varian | Parameter encoder | Parameter head regresi | Total parameter | FLOPs encoder | FLOPs head regresi | Total FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## Hiperparameter

Tabel berikut menampilkan nilai default yang ditentukan oleh `Aftab`. Argumen `experiment_name` wajib diisi dan tidak memiliki nilai default.

### Pelatihan dan lingkungan

| Hiperparameter (argumen `Aftab`) | Default |
| :--- | :--- |
| Encoder (`encoder`) | Gamma-Hadamax-Valid |
| Jaringan (`network`) | Distributional Bootstrapped (Ensemble) Dueling |
| Total frame (`frames`) | 200,000,000 |
| Lompatan frame (`frame_skip`) | 4 |
| Tumpukan frame (`frame_stack`) | 4 |
| No-op maksimum (`noop`) | 30 |
| Laju pembelajaran (`lr`) | $2.5 \times 10^{-4}$ |
| Lingkungan pelatihan (`train_environments`) | 128 |
| Lingkungan pengujian (`test_environments`) | 8 |
| Langkah per pembaruan (`steps_per_update`) | 32 |
| Ukuran batch (turunan) | 4,096 |
| Mini-batch (`mini_batches`) | 32 |
| Ukuran mini-batch (turunan) | 128 |
| Faktor diskonto ($\gamma$) | 0.99 |
| $\lambda$ return (`return_lambda`) | 0.65 |
| Epoch (`epochs`) | 2 |
| Norma gradien (`gradient_norm`) | 10.0 |
| Dimensi embedding (`embedding_dimension`) | 512 |
| Episodic life pelatihan (`train_episodic_life`) | `True` |
| Episodic life pengujian (`test_episodic_life`) | `False` |
| Kliping reward pelatihan (`train_reward_clip`) | `True` |
| Kliping reward pengujian (`test_reward_clip`) | `True` |
| Jadwal epsilon | Linear |
| Rasio annealing epsilon | 10% |

### Pengoptimal

| Hiperparameter (argumen `Aftab`) | Default |
| :--- | :--- |
| Pengoptimal (`optimizer`) | [Rectified Adam](https://arxiv.org/abs/1908.03265) (`"radam"`) |
| Epsilon (`optimizer_epsilon`) | $1 \times 10^{-5}$ |
| Peluruhan bobot (`optimizer_weight_decay`) | 0.0 |
| $\beta_1$ (`optimizer_first_beta`) | 0.9 |
| $\beta_2$ (`optimizer_second_beta`) | 0.999 |

### Nilai Q distributional dan bootstrapped (ensemble)

| Hiperparameter (argumen `Aftab`) | Default |
| :--- | :--- |
| Bin distributional (`distributional_bins`) | 51 |
| Minimum distributional (`distributional_min_value`) | -10.0 |
| Maksimum distributional (`distributional_max_value`) | 10.0 |
| Sigma distributional (`distributional_sigma`) | `None` (diturunkan dari rasio sigma) |
| Rasio sigma distributional (`distributional_sigma_ratio`) | 0.75 |
| Kliping nilai distributional (`distributional_value_clip`) | 0.0 |
| Head bootstrap (`bootstrap_heads`) | 10 |
| Probabilitas bootstrap (`bootstrap_probability`) | 1.0 |

### Penggantian Procgen

| Hiperparameter | Default | Procgen |
| :--- | :--- | :--- |
| Lingkungan pelatihan | 128 | 64 (`procgen_train_environments`) |
| Langkah per pembaruan | 32 | 256 (`procgen_steps_per_update`) |
| Ukuran batch | 4,096 | 16,384 |
| Ukuran mini-batch | 128 | 512 |

<em>Untuk lingkungan Procgen, Aftab secara otomatis menerapkan dua penggantian di atas; default lainnya tidak berubah.</em>

## Signifikansi statistik

### Eksperimen encoder

<table>
  <tr>
    <th align="center">Uji peringkat bertanda Wilcoxon</th>
    <th align="center">Uji peringkat bertanda Wilcoxon (dikoreksi)</th>
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
    <th colspan="2" align="center">Probabilitas peningkatan</th>
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

### Eksperimen Hadamax

<table>
  <tr>
    <th align="center">Uji peringkat bertanda Wilcoxon</th>
    <th align="center">Uji peringkat bertanda Wilcoxon (dikoreksi)</th>
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
    <th colspan="2" align="center">Probabilitas peningkatan</th>
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

### Eksperimen nilai Q

<table>
  <tr>
    <th align="center">Uji peringkat bertanda Wilcoxon</th>
    <th align="center">Uji peringkat bertanda Wilcoxon (dikoreksi)</th>
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
    <th colspan="2" align="center">Probabilitas peningkatan</th>
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

## Reproduksibilitas

Karena sifat stokastik deep reinforcement learning, hasil tidak dapat direproduksi secara persis hanya dengan dataset tetap.
Sebagai gantinya, kami menyediakan kumpulan seed acak yang digunakan dalam eksperimen.

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

Replikasi eksperimen secara lengkap:

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

EnvPool menyediakan koleksi lingkungan Atari yang lengkap:
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

Lingkungan Procgen menggunakan observasi RGB native dengan bentuk `(3, 64, 64)`.
Aftab membaca konfigurasi EnvPool setiap tugas dan hanya menerapkan opsi yang didukung.
Karena itu, opsi khusus Atari seperti `noop`, `frame_skip`, `frame_stack`, dan
`train_episodic_life`, serta reward clipping dari EnvPool, tidak diteruskan ke
Procgen.

EnvPool menyediakan koleksi lingkungan Procgen yang lengkap:

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## Perangkat keras

Seluruh eksperimen dalam proyek ini dijalankan pada GPU [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40).

| Spesifikasi | Detail |
|--------------|----------|
| Memori GPU | 48 GB GDDR6 dengan kode koreksi galat (ECC) |
| Bandwidth memori GPU | 696 GB/s |
| Interkoneksi | NVIDIA NVLink 112,5 GB/s (dua arah); PCIe Gen4: 64 GB/s |
| NVLink | Dua arah, low-profile (2 slot) |
| Port tampilan | 3x DisplayPort 1.4* |
| Konsumsi daya maksimum | 300 W |
| Dimensi | 4,4" (T) × 10,5" (P), dua slot |
| Pendinginan | Pasif |
| Dukungan perangkat lunak vGPU | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| Profil vGPU yang didukung | Lihat Panduan Lisensi Virtual GPU |
| NVENC / NVDEC | 1x / 2x (termasuk decoding AV1) |
| Boot aman | Boot aman dan terukur dengan root of trust berbasis perangkat keras (opsional) |
| Kepatuhan NEBS | Tingkat 3 |
| Konektor daya | CPU 8 pin |

## Sitasi

Repositori:

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

Pracetak:

```bibtex
@misc{2608.07335,
  Author = {Taha Shieenavaz and Shabnam Zareshahraki and Loris Nanni},
  Title = {Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  Year = {2026},
  Eprint = {arXiv:2608.07335},
}
```

### Karya terkait

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

## Tautan berguna

- [Wikipedia: Pembelajaran penguatan (RL)](https://id.wikipedia.org/wiki/Pemelajaran_pengukuhan)
- [Wikipedia: Pembelajaran penguatan mendalam (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Wikipedia: Pembelajaran Q](https://en.wikipedia.org/wiki/Q-learning)
- [Wikipedia: PyTorch](https://id.wikipedia.org/wiki/PyTorch)
- [Wikipedia: Uji Hipotesis Statistik](https://id.wikipedia.org/wiki/Uji_hipotesis)
- [Wikipedia: Uji Peringkat Bertanda Wilcoxon](https://id.wikipedia.org/wiki/Uji_peringkat_bertanda_Wilcoxon)
- [PyTorch](https://pytorch.org/)

## Font

Font Vazirmatn digunakan untuk teks Persia dan Inggris pada header repositori GitHub serta halaman landing proyek.

[GitHub](https://github.com/rastikerdar/vazirmatn) | [Google Fonts](https://fonts.google.com/specimen/Vazirmatn)

## Lisensi

© 2025 Taha Shieenavaz.
Dilisensikan berdasarkan CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
