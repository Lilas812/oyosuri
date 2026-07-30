# 卒業論文発表スライド

相関付きランダムウォークモデルを用いた時系列解析（7分・16:9・日本語）の
発表用 beamer スライドです。

- `presentation.tex` … スライド本体（各フレームに発表者ノート `\note{}` 付き）
- `figures/` … スライドで使用する図（本文 `docs/thesis/` からコピー）

## 構成（全11枚）

1. 表紙
2. 研究の背景と目的
3. 通常のランダムウォーク：定義と確率分布（n=1〜3 の計算）
4. 相関付きランダムウォーク：定義と行列による定式化（コイン行列・P/Q 分解・時間発展）
5. 相関付きランダムウォーク：行列計算（n=1,2 の Ψ をベクトル×行列で明示）
6. 相関付きRWによる時系列予測：評価関数と当てはめ
7. 金価格データの離散化：平均練行足（変換前・変換後の2グラフ）
8. 実験：金価格 XAUUSD への適用（全期間参照の失敗 → 参照期間 W）
9. モデルの限界：局面転換への追従の遅れ
10. まとめと今後の課題
11. 結び（ご清聴ありがとうございました）

理論パート（3〜6）は、通常RW → 相関付きRW（定義→n=1,2の行列計算）
→ 予測モデルの順です。相関付きランダムウォークの n=3 は、
経路順序ごとの行列積が増えて発表時間内の説明が重くなるため省略しています。
発表者ノートの時間配分は合計約7分です。

## コンパイル方法（日本語 upLaTeX + dvipdfmx）

本文 `docs/thesis/main.tex` と同じ pLaTeX 系のワークフローです。
ファイルは UTF-8 で保存してください。

```sh
cd docs/slides
uplatex  presentation.tex     # 1回目
uplatex  presentation.tex     # 総ページ数・相互参照の確定のため2回目
dvipdfmx presentation.dvi     # presentation.pdf を生成
```

`platex` でもコンパイルできます。`latexmk` を使う場合は次の `.latexmkrc` を置くと
`latexmk presentation.tex` の一発で PDF まで生成できます。

```perl
$latex    = 'uplatex -synctex=1 -halt-on-error -interaction=nonstopmode %O %S';
$dvipdf   = 'dvipdfmx %O -o %D %S';
$pdf_mode = 3;   # dvi -> pdf (dvipdfmx)
```

## 発表者ノートの出し方

各フレーム直後の `\note{...}` が発表原稿（合計の目安 約7分）です。既定では
スライドのみが出力されます。ノートを出力するには、プリアンブル末尾にある
次の行のコメントを外してから再コンパイルしてください。

```tex
% ノートページをスライドの後ろに挿入
\setbeameroption{show notes}

% 発表者ビュー（左：スライド／右：ノート）
\setbeameroption{show notes on second screen=right}
```

## 図の差し替え

次の画像は後から差し替えられるように、画像がない間は挿入枠を表示します。
LaTeX 本文を変更せず、`figures/` に次の名前で画像を置いて再コンパイルしてください。

- `gold-price-before.pdf` / `.png` / `.jpg`
  - 7枚目左：平均練行足への変換前の金価格
- `mean-renko-after.pdf` / `.png` / `.jpg`
  - 7枚目右：変換後の平均練行足
- `full-history-overlay.pdf` / `.png` / `.jpg`
  - 8枚目右：実測値と全期間参照による予測値の重ね描き

拡張子は PDF、PNG、JPG のいずれか1つで構いません。同じ名前の画像が複数ある場合は
PDF、PNG、JPG の順に使用されます。画像が見つからない場合も
`\safeincludegraphics` がプレースホルダを表示するため、コンパイルは停止しません。
