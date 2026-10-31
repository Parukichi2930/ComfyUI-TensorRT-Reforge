# Windows / TensorRT 11.4 / SDXL 互換パッチ

作成日: 2026-10-27（日本時間）。対象は標準 SDXL base UNet を使う WAI Illustrious / NoobAI。[実機検証報告](VALIDATION_TRT11_SDXL.md)も同梱。

## 対象環境と差分の基点

| 項目 | 対象 / 検証環境 |
|---|---|
| OS | Windows 11 x64 |
| Python | ComfyUI Desktop が使用する Python 3.13.12 |
| PyTorch | 2.12.1+cu130 / torch.version.cuda = 13.0 |
| TensorRT | tensorrt-cu13 / bindings / libs = 11.4.0.106 |
| GPU | NVIDIA GeForce RTX 3060 12GB、SM 8.6 |
| NVIDIA driver | 検証機は 616.56 |
| ONNX / ONNX Script | 検証時 1.23.2 / 0.7.2 |
| 上流 Reforge | `0ecd7ca18bb75dc1176b8edde161d6d4b4573760` |
| ComfyUI | `426f5eea52a7ece1dd599fff4cb6cd1f352ddd24` |

このパッチは上記 ComfyUI API に合わせている。古い Desktop 同梱版への互換性は保証しない。TensorRT の CUDA 系統は cu13 を使用する。PyTorch の CUDA 表示、インストール済み Toolkit、ドライバの対応 CUDA は別の情報であり、同じ数字である必要はない。NVIDIA の 11.4 ドキュメントは CUDA 13 系 pip パッケージと R580 以降のドライバを案内している。検証では既存の 2.12.1+cu130 / 11.4.0.106 をそのまま使用した。

## 導入（既存 Python 環境へのインストールなし）

1. 対象 ComfyUI インスタンスを停止する。
2. 同名ノードや元の TensorRT ノードがある場合は、バックアップを `custom_nodes` の外へ退避する。下のコマンドでこの派生版を配置する。元版との同時導入は避ける。
3. `$ComfyRoot` を対象インスタンスのComfyUI本体フォルダに変更する。`$ComfyPython` が実際に使用するPythonを指すことを確認し、追加ライブラリだけをノード内の `.deps` に配置する。PyTorch 2.12.1+cu130とTensorRT 11.4 cu13が既にある環境を前提とする。

```powershell
$ComfyRoot = 'C:\path\to\ComfyUI'
$ComfyPython = Join-Path $ComfyRoot '.venv\Scripts\python.exe'
$NodeRoot = Join-Path $ComfyRoot 'custom_nodes\ComfyUI-TensorRT-Reforge'
git clone https://github.com/HarukiKosuda2930/ComfyUI-TensorRT-Reforge.git $NodeRoot
& $ComfyPython (Join-Path $NodeRoot 'tools\install_deps.py')
& $ComfyPython (Join-Path $NodeRoot 'tools\diagnose.py')
```

`install_deps.py` は `pip --target <ノード>/.deps` を使用し、ホストに既にある全パッケージの版を制約する。`.venv` のパッケージを更新・削除せず、PyTorch/TensorRT をインストールしない。制約に矛盾があれば失敗して停止する。ノード起動時は既存 site-packages を優先し、不足ライブラリを `.deps` から読む。再導入時に `.deps` が既にあれば、必要に応じてそのフォルダだけを退避してから再実行する。

Manager の自動インストールではなく、上記の手動配置と付属スクリプトを使う。元リポジトリの requirements や cu12 インストール手順は使用しない。

4. Desktop の対象インスタンスの **Manage → Startup Args → Startup Arguments** に次を追加して起動する（旧 Desktop では設定内の Server-Config に相当）。既に `--enable-dynamic-vram` があれば外す。

```text
--disable-dynamic-vram --use-pytorch-cross-attention
```

エクスポート時は Dynamic VRAM を無効にする。パッチも通常の ModelPatcher と指定 dtype でソースをロードする。ONNX 用の attention 切替は処理中だけ行い、終了時に復元する。生成だけ行う場合も、まず検証時と同じ引数で確認する。

## WAI / NoobAI のエクスポート

`TensorRT Exporter Reforge` を追加し、標準 SDXL base のチェックポイントを選択する。

| 設定 | 最初の確認用 |
|---|---|
| precision | `fp16` |
| onnx_exporter | `auto`（SDXL では `legacy` / `dynamo=False`） |
| workspace_gib | `1` |
| enable_lora | `False` |
| opt / min / max batch | すべて `1` |
| opt / min / max width | すべて `1024`（負荷確認は `512`） |
| opt / min / max height | すべて `1024`（負荷確認は `512`） |
| opt / min context mult | `1` |
| max context mult | `3`（77〜231 embeddings） |
| num_video_frames | `1` |
| filename_prefix | `tensorrt/` |

min/max の `0` は opt 値で補完されるが、最初は上表のように明示すると確認しやすい。画像サイズは64の倍数。832×1216などを使う場合は、その固定サイズで別エンジンを作る。可変画像サイズのプロファイルも作成できるが、実 SDXL で検証したのは固定512/1024である。

出力は `output/tensorrt/<チェックポイント名>_<プロファイル>_fp16.SDXL.engine` と、同名に `.json` を追加したメタデータ。SDXL FP16 の重みだけで2GBを超えるため、ONNX は外部データを使用する。`model.onnx` 以外にも多数の重みファイルが生成される。ビルド完了まで移動・削除しない。workspace の1GiBは作業領域の上限であり、GPU全使用量の上限ではない。ONNX と engine のために少なくとも十数GBのディスク空きが必要になる。

エンジンはGPU・TensorRT・OSに依存する。同じ環境でビルドして使い、環境を変更した場合は再ビルドする。ファイルサイズと構築時間はモデルとプロファイルに依存する。

## 生成ワークフロー

1. `TensorRT Loader Reforge` の `model_type=SDXL`、`sampling=auto` を選び、生成した engine を指定する。
2. ローダーの MODEL を通常の `KSampler` の model 入力へ接続する。
3. 同じチェックポイントの `CheckpointLoaderSimple` などから CLIP と VAE を取得し、通常のテキストエンコード、VAE Decodeへ接続する。チェックポイント側の MODEL 出力は使用しない。
4. `Empty Latent Image` の幅・高さは engine に合わせ、batch_size=1 から確認する。

ローダーは `models/tensorrt` と `output/tensorrt` を検索する。移動するときは `.engine` と `.engine.json` を同じフォルダに置く。このリポジトリにはengineもモデル本体も含めない。CLIP/VAEを含むチェックポイントを別途用意し、自分の環境でengineを構築する。

CFGでComfyUI内部のバッチが2になっても、batch1 engineに1件ずつ分割して処理する。プロファイル外の解像度・テキスト長は、入力名と許容形状を示して停止する。

### 2枚のGPUへCFGを分割する場合

2026-10-27の追加修正で、標準の `MultiGPU CFG Split`（`MultiGPU_WorkUnits`）に対応した。起動引数に `--cuda-device all` を追加して、`TensorRT Loader Reforge → MultiGPU CFG Split（max_gpus=2）→ KSampler` と接続する。

ローダーは各GPUで同じengineファイルを再読み込みし、別々のengineとexecution contextを作る。GPU名・世代が異なる構成への同じengineの流用は未検証。検証したのはRTX 3060 12GB×2枚で、各GPUに約5.3GiBのTensorRT重み・作業領域を持つ。1枚内で順次実行するバッチ分割とは別の機能である。詳細は [MultiGPU修正・検証](MULTIGPU_CFG_FIX.md) を参照。

通常の1枚生成で正条件・負条件が各1つなら、分担する計算は2つなので、有効に使うGPUは2枚まで。`max_gpus=4` にしても画像やUNet自体を4分割する機能ではない。

### NoobAI の予測方式

`auto` はエクスポート時に ComfyUI が判定した `model_type` と `sampling_settings` を復元する。元コードの一律EPSロードを避け、v-prediction / zero-terminal-SNR を維持する。

チェックポイントに必要な判定マーカーがなく、ComfyUI自体が誤判定する場合は、モデルの配布元を確認して明示する。

- 通常のε予測モデル: `eps`
- v予測、通常ノイズ設定: `v_prediction`
- NoobAI-XL V-Pred 1.0など、v予測＋zero-terminal-SNR: `v_prediction_zsnr`

旧 engine でメタデータがない場合は `auto` で停止する。正しい方式を指定するか再エクスポートする。NoobAIという名前だけで予測方式を決めない。モデル作者の推奨サンプラー・scheduler設定も引き継ぐ。

## 変更の内容

| 箇所 | 変更 |
|---|---|
| requirements / 起動 | TensorRT cu12固定、ORT nightly索引、SDXLに不要なonnxruntime-gpu依存を除去。AnimaのORTは必要時のみimport |
| TensorRT strong typing | TRT11は `create_network(0)`、TRT10はSTRONGLY_TYPEDフラグ。FP16等の精度BuilderFlagに依存しない |
| ONNX | SDXL autoはopset18＋legacy exporter。重みと浮動小数入力をFP16、timestepをFP32に明示。外部データ、dynamic_axes、標準attentionを使用 |
| プロファイル | SDXLの1チャンクを77 embeddingsに修正。opt形状をダミー入力に使用し、max形状での不要なメモリ確保を抑制 |
| ビルド | workspace制限、環境別timing cache、構築失敗時のNoneチェック、入力プロファイルの検証 |
| メタデータ | prediction_type、sampling_settings、UNet設定、プロファイル、環境情報を保存・復元。モデル名をengine名に含める |
| ComfyUI API | Loader既定値をSDXLに修正。fast_disk、clone引数に対応。MultiGPUは再読み込みfactoryと対象GPU上での複製に対応。通常のCUDAオブジェクトのdeepcopyは拒否 |
| CUDA実行 | current_stream、device/dtype変換とcontiguous化、バッファのrecord_stream、実行コンテキスト形状からの出力確保、API戻り値の検査 |
| 実行順序 | 同じexecution contextを排他利用し、前回のGPU処理完了を確認してから形状を再設定 |
| 制限の明示 | ControlNet、実行時transformer patches、TRT11 LoRA/Refitはエラーで停止 |

## 対応範囲

- 対象: 標準 SDXL base UNet（latent channels=4、context_dim=2048、ADM=2816）。WAI / Illustrious / NoobAIのこの構造のチェックポイント。
- TensorRT化するのはUNet。CLIP、VAE、samplerはComfyUI/PyTorch側で動く。
- 実 NoobAI チェックポイントのビルド・画質、LoRA/Refit、ControlNet、IPAdapter、FreeU、量子化モデル、FP8/BF16、SDXL refiner/inpaint、異種GPU、3枚以上での生成、Dynamo経路は未検証または対象外。
- TensorRT engineは通常のPyTorch重みのようにCPUへ部分オフロードされない。生成時は他の大きなGPUモデルを同時に保持しない構成から確認する。
- 初回のビルド・数値比較はDesktopと同じPython・ComfyUIリビジョンを使う別プロセスで行った。導入後はDesktopを再起動し、実サーバーでも単GPUと2GPUの20ステップ生成を確認した。

## 検証の再実行

CPU補助テスト:

```powershell
& $ComfyPython -m unittest discover -s (Join-Path $NodeRoot 'tests') -v
```

小型モデルによるGPU検証（実行にはGPUを使用する）:

```powershell
& $ComfyPython (Join-Path $NodeRoot 'tools\validate_gpu.py') --comfy-root $ComfyRoot --output (Join-Path $ComfyRoot 'output\trt-smoke')
```

実SDXLのエクスポート・1ステップ出力比較:

```powershell
$Checkpoint = Join-Path $ComfyRoot 'models\checkpoints\WAI-illustrious-SDXL_-_v17-0.safetensors'
& $ComfyPython (Join-Path $NodeRoot 'tools\validate_gpu.py') --comfy-root $ComfyRoot --output (Join-Path $ComfyRoot 'output\trt-wai-test') --checkpoint $Checkpoint --resolution 1024
```

`validate_workflow.py` も付属。`--comfy-root`、`--checkpoint`、`--engine`、`--output`、`--resolution` を指定すると、実CLIP→Euler/CFG4→VAEの統合確認を行い、PNGとJSONを出力する。`--steps` は既定4で、検証では20も実行した。画質評価や速度比較を目的としない。

## 更新・ロールバック

更新時はComfyUIを停止し、この派生版のフォルダで `git pull --ff-only` を実行する。ローカル変更がある場合は先に退避する。

ロールバックはComfyUIを停止し、追加ノードを `custom_nodes` 外へ退避して元のノードに戻す。依存ライブラリはノードの `.deps` 内なので、PyTorch環境のダウングレードや再インストールは不要。

## 一次資料

- [上流 Reforge](https://github.com/zaochuan5854/ComfyUI-TensorRT-Reforge/tree/0ecd7ca18bb75dc1176b8edde161d6d4b4573760)
- [TensorRT 11.4 Python移行例](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/api/migration/tensorrt-10x-to-11x-python-api-patterns.html)
- [TensorRT strong typing / 精度制御](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/inference-library/precision-control.html)
- [TensorRT 11.4 Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/getting-started/support-matrix.html)
- [TensorRT pipパッケージとCUDA系統](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/installing-tensorrt/install-pip.html)
- [PyTorch 2.12 ONNX API](https://docs.pytorch.org/docs/2.12/onnx.html)
- [Comfy Desktop: Manage / Startup Args](https://docs.comfy.org/installation/desktop/usage/manage)
- [ComfyUI model_patcher（確認リビジョン）](https://github.com/Comfy-Org/ComfyUI/blob/426f5eea52a7ece1dd599fff4cb6cd1f352ddd24/comfy/model_patcher.py)
- [ComfyUI SDXL予測方式の判定](https://github.com/Comfy-Org/ComfyUI/blob/426f5eea52a7ece1dd599fff4cb6cd1f352ddd24/comfy/supported_models.py)
- [NoobAI-XL V-Pred 1.0 モデルカード](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0)
