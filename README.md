# Peptide Agent Screening Utility

可直接运行的多肽功能智能体评分工具，内置**氨基酸组成保留的序列打乱对照**，用于识别主要由组成偏好而非序列顺序驱动的高分候选。

## 已包含的智能体

- Osteogenic（促成骨）
- AntiOsteoclast（抗破骨）
- DPPIV（DPP-IV 抑制相关）
- Antioxidant（抗氧化）
- Anti-inflammatory（抗炎）

每个智能体目录包含 5 个训练种子的权重及校准信息。本程序使用 5 个种子的均值进行评分。

## NM meta-model and research code

This repository also includes the model and analysis code associated with the Nature Methods manuscript:

- `meta_model/T40/`: five independent seeds of the T40 meta-learned initialization (`meta_seed42.pt`, `meta_seed7.pt`, `meta_seed13.pt`, `meta_seed15.pt`, and `meta_seed2026.pt`).
- `research/`: research scripts for meta-training, held-out evaluation, function-specific adaptation, and the composition-preserved shuffle analyses used in the wound-healing case study.
- `agents/`: five ready-to-use function-specific agents, each containing five adapted seeds and calibration metadata.

The ready-to-use inference entry point is `peptide_screen.py`. The scripts under `research/` are provided as the analysis record and retain the original server-directory configuration; adjust the documented base paths and install the required analysis dependencies before rerunning them. The ESM-2 150M backbone is downloaded separately and is not redistributed in this repository.

All model scores are intended for candidate ranking and prioritization, not as calibrated probabilities of biological activity.

## 安装

建议使用 Python 3.10 或兼容版本，并在独立环境中安装：

```bash
pip install -r requirements.txt
```

首次运行会自动下载 ESM-2 150M 模型（约 560 MB），需要网络连接。检测到 CUDA GPU 时自动使用 GPU；CPU 也可运行小规模数据，但速度较慢。

## 输入

支持三种格式：

1. FASTA：`.fasta` 或 `.fa`；
2. 纯文本：每行一条序列；
3. CSV：必须包含 `seq` 列，可选 `name` 列。

仅支持 20 种标准氨基酸；重复序列会自动去重并保留第一个名称。

示例 `peptides.fasta`：

```fasta
>candidate_1
SLADLGEALKSLLK
>candidate_2
GPWGPLGPAK
```

## 一键评分与筛选

```bash
python peptide_screen.py \
  --agents agents/Osteogenic agents/AntiOsteoclast agents/DPPIV agents/Antioxidant agents/Anti-inflammatory \
  --input peptides.fasta \
  --output scores.csv \
  --shuffle 20 \
  --select 12
```

输出：

- `scores.csv`：全部输入肽的评分、对照统计及综合排序；
- `scores_shortlist12.csv`：通过过滤并去除高度相似序列后的候选清单。实际数量可能少于请求数量，程序不会用不合格候选补足名额。

只评分、不生成候选清单时，省略 `--select` 即可。

## 输出字段

对每个智能体 `<function>` 输出：

| 字段 | 含义 |
|---|---|
| `<function>_score` | 5 个种子的平均 sigmoid 分数，仅用于候选排序，不是活性概率 |
| `<function>_pct` | 该分数在本次输入库内的百分位，范围为 0–1 |
| `<function>_sd` | 5 个种子之间的标准差，反映模型集成的不稳定性 |
| `<function>_z` | 原序列相对 20 个组成不变、顺序打乱变体的 z 值 |
| `<function>_p` | 经验 p 值：`[1 + 打乱分数不低于原分数的数量] / (1 + 打乱次数)` |

此外：

- `mean_score`：所有所选智能体原始分数的算术平均，仅用于综合排序；
- `rank`：按 `mean_score` 降序的排名；
- `z_mean`：各智能体 z 值的平均；
- `p_max`：各智能体经验 p 值的最大值；
- `sd_mean`：各智能体种子间标准差的平均。

`--select K` 默认保留 `z_mean ≥ 1.5`、`p_max ≤ 1.0`、`sd_mean ≤ 0.15` 的候选，再按 3-mer Jaccard 相似度 `≥ 0.5` 聚类，每个序列家族仅保留一个代表。候选按 `z_mean`、`mean_score` 依次排序。

可根据需要调整：

```bash
python peptide_screen.py \
  --agents agents/Osteogenic agents/Antioxidant \
  --input peptides.fasta \
  --output scores.csv \
  --shuffle 100 \
  --select 12 \
  --z-min 1.5 \
  --p-max 0.2 \
  --sd-max 0.15
```

增加 `--shuffle` 可提高经验 p 值的分辨率，但会显著增加计算量。`--p-max 1.0` 表示不额外用 p 值过滤；`--p-max 0.2` 是更严格的启发式设置，不是通用的统计学显著性标准。

## 如何解释打乱对照

打乱变体与原肽具有相同的长度和氨基酸组成，仅改变残基顺序。因此：

- 原肽分数接近其打乱变体，或 z 接近 0：高分可能主要由组成偏好驱动；
- 原肽稳定高于其打乱变体，z 较高：评分包含更多序列顺序信息；
- 同聚肽等打乱后序列不变的情况无法证明序列特异性，程序保守地返回 `z = 0`、`p = 1`。

该对照用于**识别和缓解组成偏差**，不能保证彻底消除模型偏差，也不能替代独立实验验证。`pct` 会随输入库改变，不同输入库之间的百分位不应直接比较。最终候选仍需结合合成可行性、安全性、作用机制和湿实验验证。
