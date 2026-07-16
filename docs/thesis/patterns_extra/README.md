# 局面別パターンの追加サンプル（コンタクトシート）

§7.4 の 4 パターンについて、同じ生成規則で乱数 seed だけを変えた
**複数インスタンス（各 6 例）** を 1 枚のグリッド図にまとめたもの。
参照期間は小・中・大を代表する **W = 2 / 10 / 60**。各サブプロットに
実測 $x_n$ と 3 通りの予測を重ね、凡例に MAE を併記している
（体裁は本文の `pat_*_w.png` と同一）。

| ファイル | パターン | 日本語 |
|---|---|---|
| `gallery_uptrend_w.png`  | uptrend  | 上昇継続 |
| `gallery_range2up_w.png` | range2up | レンジ→上昇 |
| `gallery_up2range_w.png` | up2range | 上昇→レンジ |
| `gallery_up2down_w.png`  | up2down  | 上昇→下降 |

seed=0 は本文図 `pat_*_w.png` と同一インスタンス（MAE も一致）。
seed=1..5 は同じパターンの別サンプルで、W ごとの追従傾向が
seed をまたいでどれだけ安定しているかをまとめて目視できる。

## 再生成 / 例数の変更

`colab/oyosuri_patterns.py` の gallery 関数で作る。

```python
import matplotlib; matplotlib.use("Agg")
import sys; sys.path.insert(0, "colab")
from oyosuri_patterns import run_pattern_gallery, run_all_pattern_galleries

# 1 パターンだけ（例数・列数を変えられる）
fig, res = run_pattern_gallery("uptrend", n_instances=6, windows=(2, 10, 60))

# 4 パターンを一括生成して保存
run_all_pattern_galleries(
    n_instances=6, windows=(2, 10, 60),
    save_dir="docs/thesis/patterns_extra",
)
```

`n_instances` を増やすとサンプルが増える。W=60 の rolling 当てはめが
重いので、多数生成するときは並列化（`multiprocessing`）推奨。
