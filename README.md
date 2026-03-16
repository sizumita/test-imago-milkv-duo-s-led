# milkv-led

`imago:usb` 経由で `/dev/bus/usb/001/002` の Anker PowerConf C300 に接続し、
640x480 / MJPEG / 30fps の UVC stream から JPEG を継続取得して
`/captures/last-frame.jpg` を更新し続ける Wasm guest です。

前提:
- `imagod` 側で `imago:usb` native plugin が有効
- `resources.usb.paths` に `/dev/bus/usb/001/002` が含まれる
- `Bus 001 Device 002: ID 291a:3361 Anker Anker PowerConf C300` が接続済み

実装メモ:
- UVC の VideoStreaming interface `3`, alt setting `1`, endpoint `0x83` を使用
- format index `1` の MJPEG, frame index `5` の 640x480 を固定
- `SET_CUR(PROBE)` -> `GET_CUR(PROBE)` -> `SET_CUR(COMMIT)` の後に isochronous IN を 1 packet ずつ読む
- 初回 capture が失敗しても service は終了せず、5 秒間隔で再試行する
- 初回成功後は 30 秒ごとに `last-frame.jpg` を更新する
- 画像は `/captures/last-frame.jpg.tmp` へ書いてから rename で入れ替えるため、失敗時も前回成功画像を保持する

出力:
- guest から見えるパスは `/captures/last-frame.jpg`
- host 側では release に展開された `assets/captures/last-frame.jpg` に保存される
