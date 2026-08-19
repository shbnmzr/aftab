<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-dark.svg">
  <img alt="Phần đầu Aftab" src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header-light.svg">
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

## Tổng quan

**Aftab** (trong <a href="https://en.wikipedia.org/wiki/Aftab">tiếng Ba Tư</a>: آفتاب, nghĩa là “mặt trời” hoặc “tia nắng”) là một framework benchmark dùng để đánh giá các bộ mã hóa dựa trên CNN trong PQN trên nhiều <a href="https://en.wikipedia.org/wiki/Atari_Games">trò chơi Atari</a>. Framework cung cấp các công cụ chuẩn hóa cho việc huấn luyện, đánh giá và tái lập kết quả nghiên cứu học tăng cường sâu.

Hãy xem kiến trúc Aftab so với các đường cơ sở PQN tiêu chuẩn trong các [video minh họa](https://github.com/tahashieenavaz/aftab/blob/main/videos.md) này.

Nghiên cứu này được thực hiện mà không nhận bất kỳ nguồn tài trợ nào; vì vậy, nếu công trình của chúng tôi hữu ích, hãy cân nhắc [tài trợ trên GitHub](https://github.com/sponsors/tahashieenavaz) 💛.

### Thí nghiệm về bộ mã hóa

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
      <th>IQM HNS (50 triệu khung hình cuối)</th>
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

### Thí nghiệm Hadamax

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
      <th>IQM HNS (50 triệu khung hình cuối)</th>
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

Tài liệu tham khảo:
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### Thí nghiệm về giá trị Q

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
      <th>IQM HNS (50 triệu khung hình cuối)</th>
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

Tài liệu tham khảo:
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

### Thí nghiệm Procgen (ngăn ngừa quá khớp)

Do chưa có benchmark công khai nào so sánh điểm chuẩn hóa theo con người của các môi trường Procgen, chúng tôi đã tạo PNS (Procgen Normalized Score), một phép chuẩn hóa min-max đơn giản cho điểm số giữa các seed.

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
      <th>IQM PNS (50 triệu khung hình cuối)</th>
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

## Cài đặt

Cài đặt bằng pip:

```bash
pip install aftab
```

Ngoài ra, bạn có thể sao chép kho mã và cài đặt ở chế độ `editable`.

```bash
git clone https://github.com/tahashieenavaz/aftab.git aftab_source
pip install -e aftab_source
```

Chúng tôi đặc biệt khuyên dùng [Micromamba](https://github.com/mamba-org/micromamba-releases) để tạo môi trường ảo. Hướng dẫn chi tiết có [tại đây](https://github.com/tahashieenavaz/aftab/blob/main/scripts/README.md).

## Huấn luyện tác tử

**API JAX hiện đang được phát triển** và dự kiến hoàn thành trước cuối năm 2026. Chúng tôi rất hoan nghênh mọi đóng góp.

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


## Tích hợp bộ mã hóa tùy chỉnh

Bạn có thể định nghĩa bộ mã hóa riêng dưới dạng một mô-đun PyTorch rồi truyền nó cho tác tử:

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## Kết quả

Tất cả kết quả thí nghiệm được sắp xếp theo từng nhóm. Mỗi phần bao gồm:
- **Bảng**: kết quả dạng số (HNS/PHS và điểm thô)
- **Biểu đồ**: điểm IQM đã chuẩn hóa và các đường cong huấn luyện

### Thí nghiệm về bộ mã hóa

**Bảng**
- [Điểm chuẩn hóa theo hiệu suất con người](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/human_normalized_scores.md)
- [Điểm số](https://github.com/tahashieenavaz/aftab/blob/main/results/encoder_experiments/scores.md)

**Biểu đồ**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/human_normalized_score)
- [Diễn biến hàm mất mát](https://github.com/tahashieenavaz/aftab/blob/main/figures/encoder_experiments/loss)

---

### Thí nghiệm Hadamax

**Bảng**
- [Điểm chuẩn hóa theo hiệu suất con người](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/human_normalized_scores.md)
- [Điểm số](https://github.com/tahashieenavaz/aftab/blob/main/results/hadamax_experiments/scores.md)

**Biểu đồ**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/human_normalized_score)
- [Diễn biến hàm mất mát](https://github.com/tahashieenavaz/aftab/blob/main/figures/hadamax_experiments/loss)

---

### Thí nghiệm về giá trị Q

**Bảng**
- [Điểm chuẩn hóa theo hiệu suất con người](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/human_normalized_scores.md)
- [Điểm số](https://github.com/tahashieenavaz/aftab/blob/main/results/qvalue_experiments/scores.md)

**Biểu đồ**
- [IQM HNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/human_normalized_score)
- [Diễn biến hàm mất mát](https://github.com/tahashieenavaz/aftab/blob/main/figures/qvalue_experiments/loss)

---

### Thí nghiệm Procgen

**Bảng**
- [Điểm Procgen đã chuẩn hóa](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/procgen_normalized_scores.md)
- [Điểm số](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/scores.md)
- [PNS AUC theo seed](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_seed.md)
- [PNS AUC theo trò chơi](https://github.com/tahashieenavaz/aftab/blob/main/results/procgen_experiments/auc_game.md)

**Biểu đồ**
- [IQM PNS](https://github.com/tahashieenavaz/aftab/blob/main/figures/procgen_experiments/procgen_normalized_score)

## Độ phức tạp của mô hình

### Các biến thể cơ sở

| Biến thể | Tham số bộ mã hóa | Tham số đầu hồi quy | Tổng số tham số | FLOPs bộ mã hóa | FLOPs đầu hồi quy | Tổng FLOPs |
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

> **Lưu ý:** biến thể Eta có nhiều tham số hơn đáng kể so với các biến thể khác, chủ yếu vì bộ mã hóa tạo ra số lượng đặc trưng rất lớn.

---

### Các biến thể Hadamax

| Biến thể | Tham số bộ mã hóa | Tham số đầu hồi quy | Tổng số tham số | FLOPs bộ mã hóa | FLOPs đầu hồi quy | Tổng FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma-Hadamax-Valid** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma-Hadamax-Same** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## Siêu tham số

Các bảng sau thể hiện giá trị mặc định do `Aftab` định nghĩa. Đối số `experiment_name` là bắt buộc và không có giá trị mặc định.

### Huấn luyện và môi trường

| Siêu tham số (đối số `Aftab`) | Mặc định |
| :--- | :--- |
| Bộ mã hóa (`encoder`) | Gamma-Hadamax-Valid |
| Mạng (`network`) | Distributional Bootstrapped (Ensemble) Dueling |
| Tổng số khung hình (`frames`) | 200,000,000 |
| Bỏ qua khung hình (`frame_skip`) | 4 |
| Xếp chồng khung hình (`frame_stack`) | 4 |
| No-op tối đa (`noop`) | 30 |
| Tốc độ học (`lr`) | $2.5 \times 10^{-4}$ |
| Số môi trường huấn luyện (`train_environments`) | 128 |
| Số môi trường kiểm thử (`test_environments`) | 8 |
| Số bước mỗi lần cập nhật (`steps_per_update`) | 32 |
| Kích thước batch (suy ra) | 4,096 |
| Số mini-batch (`mini_batches`) | 32 |
| Kích thước mini-batch (suy ra) | 128 |
| Hệ số chiết khấu ($\gamma$) | 0.99 |
| $\lambda$ hoàn vốn (`return_lambda`) | 0.65 |
| Số epoch (`epochs`) | 2 |
| Chuẩn gradient (`gradient_norm`) | 10.0 |
| Kích thước embedding (`embedding_dimension`) | 512 |
| Episodic life khi huấn luyện (`train_episodic_life`) | `True` |
| Episodic life khi kiểm thử (`test_episodic_life`) | `False` |
| Cắt phần thưởng khi huấn luyện (`train_reward_clip`) | `True` |
| Cắt phần thưởng khi kiểm thử (`test_reward_clip`) | `True` |
| Lịch epsilon | Tuyến tính |
| Tỷ lệ annealing epsilon | 10% |

### Bộ tối ưu hóa

| Siêu tham số (đối số `Aftab`) | Mặc định |
| :--- | :--- |
| Bộ tối ưu hóa (`optimizer`) | [Rectified Adam](https://arxiv.org/abs/1908.03265) (`"radam"`) |
| Epsilon (`optimizer_epsilon`) | $1 \times 10^{-5}$ |
| Suy giảm trọng số (`optimizer_weight_decay`) | 0.0 |
| $\beta_1$ (`optimizer_first_beta`) | 0.9 |
| $\beta_2$ (`optimizer_second_beta`) | 0.999 |

### Giá trị Q phân phối và bootstrapped (ensemble)

| Siêu tham số (đối số `Aftab`) | Mặc định |
| :--- | :--- |
| Số bin phân phối (`distributional_bins`) | 51 |
| Giá trị phân phối tối thiểu (`distributional_min_value`) | -10.0 |
| Giá trị phân phối tối đa (`distributional_max_value`) | 10.0 |
| Sigma phân phối (`distributional_sigma`) | `None` (suy ra từ tỷ lệ sigma) |
| Tỷ lệ sigma phân phối (`distributional_sigma_ratio`) | 0.75 |
| Cắt giá trị phân phối (`distributional_value_clip`) | 0.0 |
| Số head bootstrap (`bootstrap_heads`) | 10 |
| Xác suất bootstrap (`bootstrap_probability`) | 1.0 |

### Giá trị ghi đè cho Procgen

| Siêu tham số | Mặc định | Procgen |
| :--- | :--- | :--- |
| Số môi trường huấn luyện | 128 | 64 (`procgen_train_environments`) |
| Số bước mỗi lần cập nhật | 32 | 256 (`procgen_steps_per_update`) |
| Kích thước batch | 4,096 | 16,384 |
| Kích thước mini-batch | 128 | 512 |

<em>Với môi trường Procgen, Aftab tự động áp dụng hai giá trị ghi đè ở trên; các giá trị mặc định khác không đổi.</em>

## Ý nghĩa thống kê

### Thí nghiệm về bộ mã hóa

<table>
  <tr>
    <th align="center">Kiểm định hạng có dấu Wilcoxon</th>
    <th align="center">Kiểm định hạng có dấu Wilcoxon (đã hiệu chỉnh)</th>
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
    <th colspan="2" align="center">Xác suất cải thiện</th>
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

### Thí nghiệm Hadamax

<table>
  <tr>
    <th align="center">Kiểm định hạng có dấu Wilcoxon</th>
    <th align="center">Kiểm định hạng có dấu Wilcoxon (đã hiệu chỉnh)</th>
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
    <th colspan="2" align="center">Xác suất cải thiện</th>
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

### Thí nghiệm về giá trị Q

<table>
  <tr>
    <th align="center">Kiểm định hạng có dấu Wilcoxon</th>
    <th align="center">Kiểm định hạng có dấu Wilcoxon (đã hiệu chỉnh)</th>
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
    <th colspan="2" align="center">Xác suất cải thiện</th>
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

## Khả năng tái lập

Do tính ngẫu nhiên của học tăng cường sâu, không thể tái lập chính xác kết quả chỉ bằng các tập dữ liệu cố định.
Thay vào đó, chúng tôi cung cấp tập seed ngẫu nhiên đã dùng trong các thí nghiệm.

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

Tái lập toàn bộ thí nghiệm:

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

EnvPool cung cấp một tập hợp đầy đủ các môi trường Atari:
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

Các môi trường Procgen sử dụng quan sát RGB gốc có kích thước `(3, 64, 64)`.
Aftab đọc cấu hình EnvPool của từng tác vụ và chỉ áp dụng những tùy chọn được hỗ trợ.
Vì vậy, các tùy chọn chỉ dành cho Atari như `noop`, `frame_skip`, `frame_stack`,
`train_episodic_life` và cơ chế cắt ngưỡng phần thưởng của EnvPool sẽ không được truyền cho
Procgen.

EnvPool cung cấp một tập hợp đầy đủ các môi trường Procgen:

https://envpool.readthedocs.io/en/latest/env/procgen.html#available-tasks

## Phần cứng

Tất cả thí nghiệm trong dự án đều được chạy trên GPU [Nvidia A40](https://www.nvidia.com/en-us/data-center/a40).

| Thông số | Chi tiết |
|--------------|----------|
| Bộ nhớ GPU | 48 GB GDDR6 với mã sửa lỗi (ECC) |
| Băng thông bộ nhớ GPU | 696 GB/s |
| Kết nối liên thông | NVIDIA NVLink 112,5 GB/s (hai chiều); PCIe Gen4: 64 GB/s |
| NVLink | Hai chiều, cấu hình thấp (2 khe cắm) |
| Cổng xuất hình | 3x DisplayPort 1.4* |
| Công suất tiêu thụ tối đa | 300 W |
| Kích thước | 4,4" (C) × 10,5" (D), hai khe cắm |
| Tản nhiệt | Thụ động |
| Phần mềm vGPU được hỗ trợ | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| Hồ sơ vGPU được hỗ trợ | Xem Hướng dẫn cấp phép Virtual GPU |
| NVENC / NVDEC | 1x / 2x (bao gồm giải mã AV1) |
| Khởi động an toàn | Khởi động an toàn và có đo lường với gốc tin cậy phần cứng (tùy chọn) |
| Tuân thủ NEBS | Cấp 3 |
| Đầu nối nguồn | CPU 8 chân |

## Trích dẫn

Kho mã nguồn:

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

Bản tiền in:

```bibtex
@misc{2608.07335,
  Author = {Taha Shieenavaz and Shabnam Zareshahraki and Loris Nanni},
  Title = {Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks},
  Year = {2026},
  Eprint = {arXiv:2608.07335},
}
```

### Công trình liên quan

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

## Liên kết hữu ích

- [Wikipedia: Học tăng cường (RL)](https://vi.wikipedia.org/wiki/Học_tăng_cường)
- [Wikipedia: Học tăng cường sâu (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
- [Wikipedia: Q-learning](https://vi.wikipedia.org/wiki/Q-learning_%28học_tăng_cường%29)
- [Wikipedia: PyTorch](https://vi.wikipedia.org/wiki/PyTorch)
- [Wikipedia: Kiểm định giả thuyết thống kê](https://vi.wikipedia.org/wiki/Kiểm_định_giả_thuyết_thống_kê)
- [Wikipedia: Kiểm định hạng có dấu Wilcoxon](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [PyTorch](https://pytorch.org/)

## Phông chữ

Phông chữ Vazirmatn được dùng cho văn bản tiếng Ba Tư và tiếng Anh trong phần đầu của kho GitHub và trang đích của dự án.

[GitHub](https://github.com/rastikerdar/vazirmatn) | [Google Fonts](https://fonts.google.com/specimen/Vazirmatn)

## Giấy phép

© 2025 Taha Shieenavaz.
Được cấp phép theo CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
