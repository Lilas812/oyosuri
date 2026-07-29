# 卒業論文発表スライド

相関付きランダムウォークモデルを用いた時系列解析（7分・16:9・日本語）の
発表用 beamer スライドです。

- `presentation.tex` … スライド本体（各フレームに発表者ノート `\note{}` 付き）
- `figures/` … スライドで使用する図（本文 `docs/thesis/` からコピー）

## 構成（全12枚）

1. 表紙
2. 研究の背景と目的
3. 通常のランダムウォーク：定義と確率分布（n=1〜3 の計算）
4. 相関付きランダムウォーク：定義と行列による定式化（コイン行列・P/Q 分解・時間発展）
5. 相関付きランダムウォーク：行列計算（n=1,2 の Ψ をベクトル×行列で明示）
6. 相関付きランダムウォーク：n=3 と非可換性（PQ≠QP・経路和 Ξ₃・完全な確率式）
7. 相関付きRWによる時系列予測：評価関数と当てはめ
8. 金価格データの離散化：平均練行足
9. 実験：金価格 XAUUSD への適用（全期間参照の失敗 → 参照期間 W）
10. モデルの限界：局面転換への追従の遅れ
11. まとめと今後の課題
12. 結び（ご清聴ありがとうございました）

理論パート（3〜7）は、通常RW → 相関付きRW（定義→行列計算→n=3）→ 予測モデル
の順に、行列とベクトルの積・時間発展を実際に計算して見せる構成です。
このため所要時間は約7分半〜8分とやや長めです。厳密に7分に収めたい場合は、
スライド6（n=3）を非可換性の指摘だけに絞る、スライド5・6の一部を口頭補足に
回す、実験（9）を簡略化する、などで調整できます。

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

`figures/` 内の PNG を置き換えるとスライドに反映されます。`\graphicspath` に
`../thesis/` も含めているため、`docs/thesis/` 側の元図を直接参照させることも
できます。画像が見つからない場合は `\safeincludegraphics` がプレースホルダを
表示し、コンパイルは停止しません。
