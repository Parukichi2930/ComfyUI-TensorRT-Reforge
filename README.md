# ComfyUI-TensorRT-Reforge — TensorRT 11 / SDXL fork

Windows ComfyUI向けに、**TensorRT 11.4・CUDA 13でのSDXL生成と、2GPUへのCFG分割**を修正したコミュニティ派生版です。既存のPyTorch環境を変更せずに導入する手順を用意しています。

This fork adds TensorRT 11 strong typing, SDXL ONNX export/runtime fixes, and native two-GPU CFG splitting. Tested on Windows with RTX 3060 12GB. Documentation is primarily in Japanese. Model weights and compiled engines are not distributed.

## 元リポジトリと変更内容

[zaochuan5854/ComfyUI-TensorRT-Reforge](https://github.com/zaochuan5854/ComfyUI-TensorRT-Reforge) の `0ecd7ca18bb75dc1176b8edde161d6d4b4573760` を基点にしています。その前身である [ComfyUI_TensorRT](https://github.com/comfyanonymous/ComfyUI_TensorRT) と、各プロジェクトの作者・貢献者に感謝します。

- TensorRT 11のstrong typingに対応し、古い精度フラグへの依存を除去。
- SDXLのFP16・opset18・外部重みを使うONNX書き出し、入力形状と予測方式の保存・復元を修正。
- 現在のComfyUI API、CUDA current stream、出力バッファ、CFGバッチ分割を修正。
- 標準の `MultiGPU CFG Split` 向けに、対象GPU上で独立したengineを再読み込みする処理を追加。
- ONNX補助ライブラリをノード内の `.deps` に配置。インストーラーはPyTorch/TensorRTを追加・更新しません。

変更日と概要は [CHANGELOG.md](CHANGELOG.md)、具体的な導入・エクスポート手順は [WINDOWS_TRT11_SDXL.md](WINDOWS_TRT11_SDXL.md) を参照してください。

## 検証した環境

| 項目 | 検証環境 |
|---|---|
| OS / Python | Windows 11 x64 / Python 3.13.12 |
| ComfyUI | `426f5eea52a7ece1dd599fff4cb6cd1f352ddd24`（v0.43.0） |
| PyTorch | `2.12.1+cu130` |
| TensorRT | cu13 / `11.4.0.106` |
| GPU | RTX 3060 12GB、単GPUと同型2枚 |
| 実モデル | WAI Illustrious SDXL v17 |

WAI v17の512/1024用engine構築と推論、1024×1024・CFG4・20ステップの単GPU／2GPU生成を確認しました。GPU検証は2026-10-27の作業記録に基づきます。全環境への互換性や速度向上率を保証するものではありません。[実機検証報告](VALIDATION_TRT11_SDXL.md)

**NoobAIの実チェックポイントは未検証です。** 標準SDXL base構造を対象に、EPS／v-prediction／zero-terminal-SNRのメタデータを扱います。モデル名だけで予測方式を選ばず、配布元の指定を確認してください。

## 導入

PyTorch `2.12.1+cu130` とTensorRT `11.4.0.106`（cu13）が既にある環境を前提にしています。ComfyUIを停止し、既存の同名ノードや元のTensorRTノードがあれば `custom_nodes` の外へ退避してください。ノードIDが重複するため、元版との同時導入は避けてください。

PowerShellで `$ComfyRoot` を実際のComfyUI本体フォルダへ変更します。Desktopの配置によってPythonの場所が異なる場合は `$ComfyPython` も変更してください。

```powershell
$ComfyRoot = 'C:\path\to\ComfyUI'
$ComfyPython = Join-Path $ComfyRoot '.venv\Scripts\python.exe'
$NodeRoot = Join-Path $ComfyRoot 'custom_nodes\ComfyUI-TensorRT-Reforge'
git clone https://github.com/HarukiKosuda2930/ComfyUI-TensorRT-Reforge.git $NodeRoot
& $ComfyPython (Join-Path $NodeRoot 'tools\install_deps.py')
& $ComfyPython (Join-Path $NodeRoot 'tools\diagnose.py')
```

追加ライブラリはノード内の `.deps` に置き、既存のsite-packagesを優先します。既存環境と依存条件が衝突する場合はインストーラーが停止します。元版のcu12向けインストール手順や、Managerによる依存関係の自動導入は使用せず、上記手順で配置してください。

ComfyUIの起動引数に次を追加し、起動します。既存の `--enable-dynamic-vram` は外してください。

```text
--disable-dynamic-vram --use-pytorch-cross-attention
```

## SDXLでの生成

1. 利用条件を確認したSDXLチェックポイントを自分で用意します。
2. `TensorRT Exporter Reforge` で、まずFP16・LoRA無効・batch1・1024×1024固定・context mult min/opt/max=1/1/3でengineを構築します。
3. `.engine` と対応する `.engine.json` を揃えて配置し、`TensorRT Loader Reforge` で `model_type=SDXL`、`sampling=auto` を選択します。
4. LoaderのMODELをKSamplerへ接続します。CLIPとVAEには同じチェックポイントを使い、latentはengineと同じ解像度・batch1にします。

engineはUNet用です。CLIP、VAE、samplerはComfyUI/PyTorch側で動きます。ビルドにはモデル重み・ONNX外部データ・engineを置く十分なディスク容量が必要です。engineは使用する環境で構築してください。

## 2GPUへのCFG分割

起動引数へ `--cuda-device all` を追加し、次の順に接続します。

```text
TensorRT Loader Reforge → MultiGPU CFG Split（max_gpus=2）→ KSampler
```

各GPUが独立したengineと実行コンテキストを持ち、正条件・負条件を並列に計算します。標準的な1枚生成では計算対象が2つなので、`max_gpus=4` にしても画像やUNet自体を4分割しません。同型のRTX 3060×2枚で確認しました。異種GPUや3枚以上は未検証です。[修正内容と検証](MULTIGPU_CFG_FIX.md)

## 制限

- この派生版の実機検証対象は標準SDXL base UNetです。上流の他モデル対応を検証済みとは扱っていません。
- TensorRT 11でのLoRA/Refit、ControlNet、実行時transformer patchesは未対応で、明示的にエラーになります。
- IPAdapter、FreeU、量子化、FP8/BF16、SDXL refiner/inpaint、Dynamo経路は未検証または対象外です。
- モデル本体、engine、ONNX重み、TensorRT本体、仮想環境、`.deps` はこのリポジトリに含みません。

## 検証の再実行

CPU補助テストは、PyTorchとTensorRTが導入済みのPythonで実行します。

```powershell
& $ComfyPython -m unittest discover -s (Join-Path $NodeRoot 'tests') -v
```

実機向けの `tools/validate_gpu.py`、`tools/validate_workflow.py`、`tools/validate_multigpu.py` も付属します。使い方は [導入手順](WINDOWS_TRT11_SDXL.md) と各スクリプトの `--help` を参照してください。

## ライセンス

元リポジトリと本派生版のコードは [MIT License](LICENSE) で公開します。元の `Copyright (c) 2024 ComfyUI contributors` と許諾・免責条項を維持し、派生版の追加変更について2026年の著作権表示を追記しています。

モデル、TensorRT、PyTorchなどの外部依存物にはそれぞれのライセンスが適用されます。モデルをengineへ変換しても、元モデルの利用・再配布条件は別途確認してください。
