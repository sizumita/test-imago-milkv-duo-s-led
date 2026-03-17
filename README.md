# milkv-led

`imago:usb` で `/dev/bus/usb/001/002` の Anker PowerConf C300 から 640x480 / MJPEG / 30fps の JPEG を継続取得し、
`wasi-nn-cvitek` が有効な `imagod` 上で YOLOv5 INT8 `.cvimodel` を実行して、
物体検出マーカー付きの JPEG を `/captures/last-frame.jpg` に保存し続ける Wasm guest です。

## milkvのセットアップ

- 公式サイトを参照してUSB-A側をセットアップする
- sshの鍵を設定する

## 前提

- `imagod` 側で `imago:usb` native plugin が有効
- `imagod` 側が `wasi-nn-cvitek` feature 付きで起動している
- `resources.usb.paths` に `/dev/bus/usb/001/002` が含まれる
- `Bus 001 Device 002: ID 291a:3361 Anker Anker PowerConf C300` が接続済み
- target 側に CVITEK TPU runtime の共有ライブラリが解決できる

## モデル asset

repo には実モデルとその metadata を同梱しています。

- `assets/models/yolo.cvimodel`
- `assets/models/yolo.toml`
- `assets/models/coco.names`

`assets/models/yolo.cvimodel` は 2026-03-17 に次の official 配布物から取得した `cv181x` 用 model です。

- `https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x/raw/main/cvimodel/yolov5_cv181x_int8_sym.cvimodel`

SHA-256:

- `adc384586f6a97e57ebd58e905e41eba3491a48b79fd3624e2f333a8089ef523`

`assets/models/yolo.toml` は Milk-V Duo S 実機上の `libcviruntime.so` から採った実 metadata に合わせています。

- input tensor: `images`, dims=`[1,3,640,640]`, fmt=`INT8`, qscale=`126.99897766113281`
- 3 head x 3 tensor の split output
- stride 8: `output0_Gather__reshape`, `372_Gather__reshape`, `373_Gather__reshape`
- stride 16: `389_Gather__reshape`, `390_Gather__reshape`, `391_Gather__reshape`
- stride 32: `407_Gather__reshape`, `408_Gather__reshape`, `409_Gather__reshape`

参照:

- [Milk-V TPU Introduction](https://milkv.io/docs/duo/application-development/tpu/tpu-introduction)
- [samples_extra/run_detector_yolov5_fused_preprocess.sh](https://github.com/milkv-duo/tpu-sdk-sg200x/blob/main/samples/samples_extra/run_detector_yolov5_fused_preprocess.sh)

## metadata helper

`wasi-nn-cvitek` は guest に `qscale` / `zero_point` を返さないため、
`assets/models/yolo.toml` は必須です。
Milk-V Duo S 上では compiler が無いことがあるので、まずは `tools/print_cvimodel_io.py` を使うのが簡単です。
この script は target 側の `libcviruntime.so` を直接読むので、Milk-V target で実行してください。

例:

```bash
scp assets/models/yolo.cvimodel root@192.168.2.4:/tmp/milkv-led-yolo.cvimodel
ssh root@192.168.2.4 'python3 - /tmp/milkv-led-yolo.cvimodel' < tools/print_cvimodel_io.py
```

`python3` が使えず compiler がある場合は `tools/print_cvimodel_io.c` でも同じ情報が取れます。

helper は少なくとも次を出します。

- input tensor 名
- output tensor 名
- output dimensions
- `qscale`
- `zero_point`

その値で `assets/models/yolo.toml` の次を更新してください。

- `[input].name`
- `[input].qscale`
- `[[heads]].[box_tensor|objectness_tensor|classes_tensor].name`
- `[[heads]].[box_tensor|objectness_tensor|classes_tensor].dimensions`
- `[[heads]].[box_tensor|objectness_tensor|classes_tensor].qscale`
- `[[heads]].[box_tensor|objectness_tensor|classes_tensor].zero_point`

`stride` と `anchors` はどの head かに応じて対応付けます。
初期値は 80x80 / 40x40 / 20x20 の YOLOv5 head を想定しています。

## 実装メモ

- UVC の VideoStreaming interface `3`, alt setting `1`, endpoint `0x83` を使用
- `SET_CUR(PROBE)` -> `GET_CUR(PROBE)` -> `SET_CUR(COMMIT)` の後に isochronous IN を 1 packet ずつ読む
- JPEG を decode して 640x640 に letterbox resize し、`images` input 用に RGB を INT8 / NCHW bytes へ量子化して `graph.load(..., autodetect, tpu)` で読み込んだ model に渡す
- output tensor は `yolo.toml` の 9 tensor split-head 契約で dequantize し、YOLOv5 decode と class-wise NMS を適用する
- 検出結果は bitmap font でラベルを描画し、`/captures/last-frame.jpg.tmp` へ書いてから rename で入れ替える
- 初回失敗でも service は終了せず、初回成功までは 5 秒ごと、成功後は 30 秒ごとに再試行する

## build と deploy

```bash
env RUSTC_WRAPPER= cargo check --target wasm32-wasip2
env RUSTC_WRAPPER= cargo build --target wasm32-wasip2 --release
imago service deploy --target default --detach
```

ログ確認:

```bash
imago service logs milkv-led --tail 200
```

成功時は次のようなログが出ます。

- `model loaded ...`
- `detections=N outputs=[...]`
- `saved annotated frame ...`

## 失敗時の見方

- `failed to read /app/assets/models/yolo.cvimodel`
  - model asset が欠けています
- `missing '...' output` / `dimensions mismatch`
  - helper 出力と `yolo.toml` が合っていません
- `RuntimeError: ... wasi-nn backend is not enabled`
  - target の `imagod` が `wasi-nn-cvitek` で起動していません
- `failed to decode jpeg`
  - camera から取れた JPEG が壊れているか、再構成が崩れています
- `detections=0`
  - 推論は動いていますが threshold が高いか、input / metadata がずれています

## 出力

- guest から見える保存先: `/captures/last-frame.jpg`
- host 側の release artifact 上の保存先: `assets/captures/last-frame.jpg`
