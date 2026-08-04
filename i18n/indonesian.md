<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../figures/header-light.svg">
  <img alt="Makalah Aftab" src="../figures/header-light.svg">
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

## Gambaran umum

**Aftab** (dari <a href="https://en.wikipedia.org/wiki/Aftab">bahasa Persia</a> آفتاب, yang berarti “matahari” atau “sinar matahari”) adalah framework benchmark untuk mengevaluasi encoder berbasis CNN pada PQN di berbagai <a href="https://en.wikipedia.org/wiki/Atari_Games">gim Atari</a>. Framework ini menyediakan perangkat standar untuk pelatihan, evaluasi, dan reproduksibilitas dalam riset deep reinforcement learning.

Kami telah menyiapkan beberapa video yang membandingkan agen PQN dan Aftab. Tonton [di sini](../videos.md).

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

Kami sangat menyarankan penggunaan [Micromamba](https://github.com/mamba-org/micromamba-releases) untuk membuat lingkungan virtual. Petunjuk lengkap tersedia [di sini](../scripts/README.md).

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
- [Skor yang dinormalisasi terhadap performa manusia](../results/encoder_experiments/human_normalized_scores.md)
- [Skor](../results/encoder_experiments/scores.md)

**Grafik**
- [IQM HNS](../figures/encoder_experiments/human_normalized_score)
- [Perkembangan loss](../figures/encoder_experiments/loss)

---

### Eksperimen Hadamax

**Tabel**
- [Skor yang dinormalisasi terhadap performa manusia](../results/hadamax_experiments/human_normalized_scores.md)
- [Skor](../results/hadamax_experiments/scores.md)

**Grafik**
- [IQM HNS](../figures/hadamax_experiments/human_normalized_score)
- [Perkembangan loss](../figures/hadamax_experiments/loss)

---

### Eksperimen nilai Q

**Tabel**
- [Skor yang dinormalisasi terhadap performa manusia](../results/qvalue_experiments/human_normalized_scores.md)
- [Skor](../results/qvalue_experiments/scores.md)

**Grafik**
- [IQM HNS](../figures/qvalue_experiments/human_normalized_score)
- [Perkembangan loss](../figures/qvalue_experiments/loss)

---

### Eksperimen Procgen

**Tabel**
- [Skor Procgen yang dinormalisasi](../results/procgen_experiments/procgen_normalized_scores.md)
- [Skor](../results/procgen_experiments/scores.md)


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

<div align="center">

| Hiperparameter | Nilai |
| :--- | :--- |
| Laju pembelajaran | $2.5 \times 10^{-4}$ |
| Lingkungan pelatihan | 128 |
| Lingkungan pengujian | 8 |
| Pengoptimal | [Rectified Adam](https://arxiv.org/abs/1908.03265) |
| Peluruhan bobot | 0 |
| $\epsilon$ | $1 \times 10^{-5}$ |
| $\beta_{1}$ | 0.9 |
| $\beta_{2}$ | 0.999 |
| Total frame | 200,000,000 |
| Fungsi kerugian | Galat kuadrat rata-rata |
| Penjadwal | Annealing linear |
| Eksplorasi $\epsilon$-greedy | 10% of total frames |
| Faktor diskonto ($\gamma$) | 0.99 |
| GAE ($\lambda$) | 0.65 |
| Epoch | 2 |
| Ukuran batch | 4096 |

</div>

<p align="center"><em>Digunakan dalam eksperimen encoder dan Hadamax.</em></p>

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

```bibtex
@article{aftab2026drl,
  title={Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
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

- [Wikipedia: Pembelajaran penguatan (RL)](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [Wikipedia: Pembelajaran penguatan mendalam (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Wikipedia: Pembelajaran Q](https://en.wikipedia.org/wiki/Q-learning)
- [Wikipedia: PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [Wikipedia: Uji Hipotesis Statistik](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [Wikipedia: Uji Peringkat Bertanda Wilcoxon](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## Lisensi

© 2025 Taha Shieenavaz.
Dilisensikan berdasarkan CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
