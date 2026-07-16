# 局面別パターンの追加サンプル（1 図 1 インスタンス）

§7.4 の 4 パターンについて、同じ生成規則で乱数 seed だけを変えた
**複数インスタンス（各 6 例）** を、**1 図につき 1 インスタンス**で出力したもの。
参照期間は小・中・大を代表する **W = 2 / 10 / 60**。各図に実測 $x_n$ と
3 通りの予測を重ね、凡例に MAE を併記している（体裁は本文の
`pat_*_w.png` と同一）。

ファイル名は `pat_{pattern}_seed{seed}_w.png`。

| パターン | 日本語 | ファイル |
|---|---|---|
| uptrend  | 上昇継続    | `pat_uptrend_seed0_w.png` … `pat_uptrend_seed5_w.png` |
| range2up | レンジ→上昇 | `pat_range2up_seed0_w.png` … `pat_range2up_seed5_w.png` |
| up2range | 上昇→レンジ | `pat_up2range_seed0_w.png` … `pat_up2range_seed5_w.png` |
| up2down  | 上昇→下降   | `pat_up2down_seed0_w.png` … `pat_up2down_seed5_w.png` |

`seed0` は本文図 `pat_*_w.png` と同一インスタンス（MAE も一致）。
`seed1`..`seed5` は同じパターンの別サンプル。

## 再生成 / 例数の変更

`colab/oyosuri_patterns.py` の `run_all_patterns`（1 図 1 インスタンスで保存）
で作る。

```python
import matplotlib; matplotlib.use("Agg")
import matplotlib_fontja  # 日本語表示（Colab では !pip install -q matplotlib-fontja）
import sys; sys.path.insert(0, "colab")
from oyosuri_patterns import run_all_patterns

# 4 パターン × seed 6 つ = 24 枚を、1 図 1 インスタンスで保存
run_all_patterns(
    n_bricks=150, seeds=(0, 1, 2, 3, 4, 5), windows=(2, 10, 60),
    fname_fmt="pat_{pattern}_seed{seed}_w.png",
    save_dir="docs/thesis/patterns_extra",
)

# 1 パターン・1 seed だけなら
from oyosuri_patterns import run_pattern_overlay
fig, r = run_pattern_overlay("uptrend", seed=6, windows=(2, 10, 60))
fig.savefig("docs/thesis/patterns_extra/pat_uptrend_seed6_w.png",
            dpi=150, bbox_inches="tight")
```

`seeds` を増やせばサンプルが増える。W=60 の rolling 当てはめが重いので、
多数生成するときは seed 単位で並列化（`multiprocessing`）すると速い。
