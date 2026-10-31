# TensorRT 11 SDXL 実機検証報告

2026-10-27（日本時間）。互換パッチを作成し、**RTX 3060 12GBで実SDXLのエクスポート・推論・1024×1024画像生成まで確認した**。NoobAI実チェックポイントは未検証。

同日追加: `MultiGPU CFG Split` 用の再読み込みfactoryを実装し、RTX 3060×2枚への正負条件の振り分けを検証した。初回版で発生した `cached_patcher_init is not initialized` は修正済み。[追加検証報告](MULTIGPU_CFG_FIX.md)を参照。

## 確認した環境

- Windows 11 x64、Python 3.13.12。
- PyTorch `2.12.1+cu130`、`torch.version.cuda=13.0`。
- TensorRT cu13 / bindings / libs `11.4.0.106`。
- NVIDIA GeForce RTX 3060、SM 8.6、ドライバ616.56。検証はCUDA device 0を使用。
- ONNX `1.23.2`、ONNX Script `0.7.2`。ONNX Runtimeは使用していない。
- ComfyUI `426f5eea52a7ece1dd599fff4cb6cd1f352ddd24`。確認したDesktopインスタンスも同じリビジョンだった。
- Reforge差分基点 `0ecd7ca18bb75dc1176b8edde161d6d4b4573760`。

## 実施結果

| 検証 | 結果 |
|---|---|
| CPU補助テスト4件 | PASS。バッチ分割、動的軸、予測方式/zero-SNR、TRT11でのネットワークフラグ依存除去 |
| 小型ONNX→TRT11 engine構築 | PASS。opset18、FP16/FP32混在のstrong typing |
| 非標準CUDA stream、非連続テンソル、CFG batch2→engine batch1 | PASS |
| 動的画像サイズ/77・154・231 embeddings、入力と異なる出力チャンネル数 | PASS |
| プロファイル外入力の拒否 | PASS |
| ComfyUI V3 Schema/Loader/ModelPatcher.clone(fast_disk) | PASS |
| V_PREDICTION + zero-terminal-SNRメタデータ復元 | PASS。小型engineを使って実ComfyUIクラスを検証 |
| WAI Illustrious v17、512×512 | ONNX書き出し、TRT engine構築、UNet出力比較 PASS |
| WAI Illustrious v17、1024×1024 | ONNX書き出し、TRT engine構築、UNet出力比較 PASS |
| 実SDXLの77/231 embeddings、1024×1024、非標準stream | PyTorchとの出力比較 PASS |
| 実CLIP→TRT Loader→ModelPatcher.clone→ComfyUI sampling→VAE | 1024×1024、Euler / normal / CFG4 / seed1234で4ステップと20ステップともPASS |
| 依存ライブラリのノード内インストーラー | PASS。追加ファイルはノードの`.deps`内 |
| ホストPythonのパッケージ一覧 | 作業前後で全104件が一致。PyTorch/TensorRTの変更なし |
| 差分の空白検査とクリーン基点への適用 | 初回パッチの配布時に確認。GitHub公開時は構文検査とCPU補助テストを再実行 |

実モデルは `WAI-illustrious-SDXL_-_v17-0.safetensors`。エンジンはFP16、batch1固定、各解像度固定、context min/opt/max=77/77/231、workspace 1GiBで構築した。

## UNet数値比較

FP16 PyTorch UNetとTensorRTを、同じランダムlatent/conditioning、timestep=500で比較した。判定は `torch.testing.assert_close(atol=0.08, rtol=0.08)`。最終1024追加検証の測定値:

| 解像度 / context長 | 最大絶対誤差 | 平均絶対誤差 | RMSE |
|---|---:|---:|---:|
| 512×512 / 77（初回確認） | 0.01171875 | 0.00131729 | 0.00180819 |
| 1024×1024 / 77 | 0.013671875 | 0.00141425 | 0.00185843 |
| 1024×1024 / 231 | 0.013671875 | 0.00158801 | 0.00206761 |

これは限られた入力での数値確認であり、全timestep・全プロンプトの同一性や画質同等性を保証するものではない。画像生成テストではlatentとVAE出力の有限値を検査し、20ステップの出力画像も目視確認した。速度の比較ベンチマークは行っていない。

## このリポジトリの公開範囲

修正済みカスタムノード、再現用テストスクリプト、導入手順と本検証報告を公開する。モデル本体、engine、ONNX重み、`.deps`、仮想環境、TensorRT本体は含めない。

検証用に構築した1024用engineは約5.16GBだった。利用者は自分の環境で構築し、`.engine` と `.engine.json` を揃えて `models/tensorrt` へ配置する。同じチェックポイントのCLIP/VAEを使用する。他のモデル・GPU・OS・TensorRT版へそのまま流用できるとは限らない。

GPU検証結果は2026-10-27の作業記録に基づく。2026-10-31のGitHub公開準備ではランタイムコードを変更せず、README・ライセンス・配布対象を整備した。

## 未検証・対象外

NoobAI実チェックポイントの構築と画質、LoRA/Refit、ControlNet/IPAdapter/FreeU、量子化、FP32経路、Dynamo経路、SDXL refiner/inpaint、異種GPU・3枚以上での生成は未検証または対象外。TRT11のLoRA/Refit、ControlNetと実行時transformer patches、通常のCUDAオブジェクトのdeepcopyはエラーで停止する。MultiGPUの複製はengineの再読み込みを使用する。

初回のビルド・数値比較は同じPythonとComfyUIソースを別プロセスから使用した。その後、ユーザーの依頼により既存Desktopインスタンスへカスタムノードを導入し、起動設定を更新して再起動した。実サーバーでも単GPUと2GPUの20ステップ生成が成功した。ComfyUI本体とホストPythonパッケージは変更していない。

標準ワークフローでTensorRTがdefault streamの性能警告を出すことがある。呼び出し元のPyTorch current stream自体がdefaultの場合であり、パッチは現在のstreamを引き継ぐ。非標準streamでの順序も別途検証済み。
