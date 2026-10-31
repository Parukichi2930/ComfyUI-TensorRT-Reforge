# Changelog

## 0.2.0 — 2026-10-31

Windows / TensorRT 11 / SDXL向け派生版の初回GitHub公開。

- TensorRT 11 strong typing、SDXL FP16 ONNX export、外部重み、77-token context profilesを修正。
- SDXLの予測方式とsampling settingsをengineメタデータへ保存・復元。
- ComfyUI V3 API、ModelPatcher.clone、current CUDA stream、出力形状・バッファ管理へ対応。
- 標準MultiGPU CFG Split向けのreload factoryを登録し、選択したGPU上で独立したTensorRT engine/contextを再生成。
- ONNX補助ライブラリをノード内へ配置するインストーラー、診断・検証スクリプトを追加。
- READMEをこの派生版の検証範囲と導入方法に合わせて更新し、MITの元著作権表示を維持。
- モデル本体、engine、ONNX重み、依存ライブラリの実体は配布対象外。

基点: [zaochuan5854/ComfyUI-TensorRT-Reforge@0ecd7ca](https://github.com/zaochuan5854/ComfyUI-TensorRT-Reforge/commit/0ecd7ca18bb75dc1176b8edde161d6d4b4573760)。ランタイム修正は2026-10-27の実機検証済みコードを引き継ぎ、公開準備で変更していない。検証した範囲と制限は [実機検証報告](VALIDATION_TRT11_SDXL.md) に記載。
