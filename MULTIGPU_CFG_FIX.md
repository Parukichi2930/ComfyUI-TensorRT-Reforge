# TensorRT + MultiGPU CFG Split 修正

2026-10-27。初回版は単GPU内のバッチ分割に対応していたが、標準 `MultiGPU_WorkUnits` ノードが必要とする再読み込みfactoryを登録していなかった。そのため2枚目のGPU向け複製で `cached_patcher_init is not initialized` が発生した。

`trt_loader.py` のモデル読込処理を再利用できる関数へ分け、`cached_patcher_init` に登録した。複製時は対象GPUを選択した状態で標準ComfyUIの `deepclone_multigpu` を呼ぶ。GPU上のengineをコピーしたり、GPU番号だけを付け替えたりせず、各GPUで独立したengineとexecution contextを生成する。ComfyUI本体とPython依存パッケージの変更は不要。

接続は `TensorRT Loader Reforge → MultiGPU CFG Split（max_gpus=2）→ KSampler`。ComfyUIを `--cuda-device all` 付きで起動する。同一型のRTX 3060 12GB×2枚で検証した。engineの再ビルドは不要だった。

`tools/validate_multigpu.py` による検証:

- 別々のモデル、engine、execution contextがcuda:0/1に存在することを確認。
- 非標準stream上でUNetを同時実行し、出力差0を確認。
- 標準ComfyUI sampler、CFG4、1024×1024、4ステップで、cuda:0に正条件 `(0,)`、cuda:1に負条件 `(1,)` が各4回渡ることを記録。
- 同じ入力で単GPU版と2GPU版の最終latentを比較し、最大絶対誤差0、平均絶対誤差0。
- モデルのcloneと `match_multigpu_clones` の経路も確認。

さらに、報告されたワークフロー（node 11=`MultiGPU_WorkUnits`、max_gpus=2、seed1241、CFG4、20ステップ、1024×1024）を、修正導入・再起動後のDesktop実サーバーで実行し、画像生成の成功を確認した。検証時は出力先のみ `SaveImage` から `PreviewImage` へ変更した。このリポジトリには検証結果の要約と再現用スクリプトを掲載し、生成画像・実行環境のローカル記録は同梱しない。ホストPythonのパッケージ一覧は作業前後で全104件が一致し、PyTorch/TensorRTの変更はない。

通常の1枚生成では正条件と負条件の2つの計算を分担するため、有効に使うGPUは2枚まで。`max_gpus` は利用枚数の上限であり、4に設定しても画像やUNet自体を4分割する機能ではない。複数画像の並列生成などは別の構成が必要になる。

異種GPU、3枚以上での生成、NoobAI実モデル、LoRA/Refitは未検証。速度比較ベンチマークは行っていない。

TensorRTはデシリアライズ時にengineをGPUへ結び付けるため、対象GPUを選んでから読み込む必要がある。[NVIDIA TensorRT公式ガイド](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/architecture/how-trt-works.html)
