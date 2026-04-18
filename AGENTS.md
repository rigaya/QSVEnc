# QSVEnc

IntelのGPU/APUに搭載されているHWエンコーダ(QSV)をlibvplを介して呼び出す。単体で動作するコマンドライン版とAviUtl/AviUtl2出力プラグイン版がある。

QSVEnc / NVEnc / VCEEnc / rkmppenc と共通化できる部分は共通ファイルで同じ実装を用い、そのうえに固有の実装を重ねている。

## ディレクトリ/ファイル構成
  
- `QSVPipeline`
  QSVEncのコア実装。Windows/Linux両対応。

  - 共通ファイル
    - 以下は QSVEnc / NVEnc / VCEEnc / rkmppenc で共通使用する。
      - afs*[.h/.cpp]
      - cl_func[.h/.cpp]
      - convert_*[.h/.cpp]
      - cpu_info[.h/.cpp]
      - gpu_info[.h/.cpp]
      - logo[.h/.cpp]
      - rgy_*[.h/.cpp]
    - 注意点
      - QSVEnc / QSVEnc / VCEEnc / rkmppenc での共通性を維持する
      - 共通化が難しい場合は、部分的なら`rgy_version.h`のマクロ(`ENCODER_QSV`, `ENCODER_NVENC`, `ENCODER_VCEENC`, `ENCODER_MPP`)で切り替える。部分的ですまない場合は、固有実装のほうで実装する。

  - 二層構造により部分的な共通化を行う。以下に2つの例を挙げる。
    - パラメータ類
      - qsv_prm[.h/.cpp]
        QSVEnc固有のパラメータ (固有実装)
      - rgy_prm[.h/.cpp]
        エンコーダ共有パラメータ
    - コマンドライン
      - qsv_cmd[.h/.cpp]
        QSVEnc固有のパラメータ (固有実装)
      - rgy_cmd[.h/.cpp]
        エンコーダ共有パラメータ

- `QSVEncC`
  `QSVPipeline`の実装を使用したCLI。Windows/Linux両対応。

- `QSVEnc`
  AviUtl/AviUtl2用プラグイン。`QSVEncC`を呼び出してエンコードする。
  Win32ビルドはAviUtl向け(.auo)、x64ビルドはAviUtl2向け(.auo2)。

- `build_pkg`
  Linuxパッケージ作成用。

- `data`
  ドキュメント用のデータ。

- `docker`
  ビルド用のベースdockerfile。

- `GPUFeatures`
  `QSVEncC --check-features`の結果集。適宜追加。Readme.mdから参照
  
- `resource`
  ビルド用のデータ。

- 以下は依存ライブラリ。基本触らない。
  - `ffmpeg_lgpl` (Windowsでのみ使用)
  - `PerfMonitor`
  - `cppcodec`
  - `dtl`
  - `jitify`
  - `json`
  - `libvpl`
  - `tinyxml2`
  - `ttmath`

## ドキュメント

- QSVEncC_Options[.md/.ja.md/.cn.md]

  コマンドラインオプションについての記載。`rgy_cmd.cpp`のヘルプともに、オプションを追加したら更新すること。

- Readme[.md/.ja.md/.cn.md]
- Build[.md/.ja.md/.cn.md]
- Install[.md/.ja.md/.cn.md]


## パラメータ追加

パラメータ追加時は、下記を編集したうえで、適宜必要な追加の実装を行う。

追加するパラメータが libvpl由来のQSVEnc独自となるパラメータか、
QSVEnc / NVEnc / VCEEnc / rkmppenc共通で使用できるパラメータかにより、
実装箇所が異なる。

また、パラメータの定義追加時には、その分類をよく検討し、適切な構造体に追加すること。

|対象 | ファイル (libvpl関連独自) | ファイル (QSVEnc / NVEnc / VCEEnc / rkmppenc共通) |
|:--|:--|
| パラメータ定義・初期化            | `qsv_prm[.h/.cpp]` | `rgy_prm[.h/.cpp]` |
| コマンド読み取り・生成・ヘルプ生成 | `qsv_cmd[.h/.cpp]` | `rgy_cmd[.h/.cpp]` |
| ドキュメント | `QSVEncC_Options.[en/ja].md` | `QSVEncC_Options.[en/ja].md` |