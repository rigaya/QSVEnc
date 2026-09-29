
# QSVEncCのインストール方法

- [Windows](./Install.ja.md#windows)
- Linux
  - [Linux (Ubuntu 20.04 以降)](./Install.ja.md#linux-ubuntu-2004-以降)
  - [Linux (Fedora 32)](./Install.ja.md#linux-fedora-32)
  - その他のLinux OS  
    配布 deb を利用できない Linux OS 向けには、ソースコードからビルドする必要があります。ビルド方法については、[こちら](./Build.ja.md)を参照してください。


## Windows 

### 1. Intelグラフィックスドライバをインストールします。
### 2. Windows用実行ファイルをダウンロードして展開します。  
実行ファイルは[こちら](https://github.com/rigaya/QSVEnc/releases)からダウンロードできます。QSVEncC_x.xx_Win32.7z が 32bit版、QSVEncC_x.xx_x64.7z が 64bit版です。通常は、64bit版を使用します。

実行時は展開したフォルダからそのまま実行できます。

64bit版の配布archiveには`libvmaf.dll`とNVIDIA backend版の`libvship.dll`が含まれます。VMAF評価はCPUで実行できますが、同梱のlibvshipによる評価にはNVIDIA GPUと対応ドライバが必要です。評価を使用しない通常のエンコードにはこれらのDLLは不要です。

## Linux (Ubuntu 20.04 以降)

Linux では、Intel GPU のエンコードに QSV と VA-API の 2 種類を使えます。

| 方法 | QSV | VA-API |
|:--|:--|:--|
| 必要なランタイム | Tiger Lake 以降は VPL (`libmfx-gen`)、それ以前の対応 GPU は Media SDK (`libmfx1`) | VA ドライバ (`iHD` / `i965`) のみ。VPL / Media SDK は不要 |
| HW デコード (`--avhw`) | 可 | 不可。`--avsw` と入力の自動選択はソフトウェアデコード |
| フィルタ | MFX VPP / OpenCL | OpenCL のみ。OpenCL がなければフィルタなしのエンコード |
| エンコード設定 | MFX の詳細設定に対応 | レート制御・GOP・品質などの基本設定。非対応の MFX 固有設定は WARN で無視、一部はエラー |

詳細な対応範囲は [--backend](./QSVEncC_Options.ja.md#--backend-autoqsvvaapi) を参照してください。性能の目安として、Arc A310 / Ubuntu 26.04 / kobuk-team PPA の iHD 26.3.2 で、同じ入力を `--avsw` で処理した場合の速度は QSV 比で H.264 約 96%、HEVC 約 101% でした。GPU・ドライバ・設定によって変わります。

VPL ランタイムがない GPU / ディストリビューションや、i965 しか使えない旧世代では VA-API を使用してください。例えば Ubuntu 26.04 には `libmfx1` がなく、Kaby Lake (Gen9、HD 630 など) は VPL ランタイムがない環境になります。VA-API は Ubuntu 24.04 の Docker (標準ドライバ 24.1) と Ubuntu 26.04 の実機 (kobuk-team PPA の iHD 26.3.2) で確認しています。26.04 標準の iHD 26.1.2 での動作は未確認です。

既定の `--backend auto` は QSV を試し、デバイスを利用できなければ VA-API に切り替えます。次の場合は QSV を試さず VA-API を選択します。

- `QSVENC_VPL_DISABLE=1` を指定した場合。
- `LIBVA_DRIVER_NAME` に `iHD` 以外を指定した場合（大文字小文字は無視）。
- ドライバの探索先に `iHD_drv_video.so` が見つからない場合（i965 のみの環境など）。

QSV の選択後に起きるエンコードエラーでは切り替えません。VPL ランタイムが起動時にクラッシュする環境では、`QSVENC_VPL_DISABLE=1 qsvencc ...` で VPL の読み込みを回避してください。VA-API を明示するには `--backend vaapi`、QSV を明示するには `--backend qsv` を指定します。

公式配布パッケージではlibvmafを静的リンクするため、VMAF評価に`libvmaf.so`は不要です。libvship評価を使用する場合は、実行時ローダーが`libvship.so`と、選択したbackendが必要とする共有ライブラリを検索できる場所へ配置します（システムのライブラリ検索パス、または`LD_LIBRARY_PATH`）。ソースから通常設定でビルドした場合はlibvmafを動的ロードするため、VMAF評価には`libvmaf.so`も必要です。評価を使用しない通常のエンコードには、これらのライブラリは不要です。

### 1. 事前準備

#### 1-1. QSV を使用する場合: Intel Media ドライバ用のリポジトリの登録

以下の Intel リポジトリのコマンドは Ubuntu 22.04 / 24.04 向けです。VA-API だけを使用する場合は 1-2. に進んでください。

:::note warn  
**Gen11以前のGPUを含む環境の場合、この工程をスキップし、2. に進んでください。**

Intel repositoryは最新のuser mode driverを提供しますが、intel-opencl-icd 24.35以降はGen12以降向けとなっているため、OpenCLデバイスを検出できない場合があります。Gen11以前のGPUを含む環境では、Ubuntu標準repoを使用してください。

Gen11以前とGen12以降のGPUが同じPCに混在する場合は、この工程を実施したうえで、[3-1.](#3-1-gen11以前とgen12以降のgpuを併存させる場合) の手順でGen11以前向けのOpenCLランタイムを追加してください。

- Gen11以前: Broadwell, Skylake, Kaby Lake, Coffee Lake, Apollo Lake, Gemini Lake, Ice Lake, Elkhart Lake
- Gen12以降: Tiger Lake, Rocket Lake, Alder Lake, Raptor Lake, Arc dGPU など
::

[こちらのリンク](https://dgpu-docs.intel.com/driver/client/overview.html)に沿って、ドライバをインストールします。

まず、必要なツールを導入します。

```Shell
sudo apt-get install -y gpg-agent wget
```

次に、Intelのリポジトリを追加します。

```Shell
# Ubuntu 24.04
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list

# Ubuntu 22.04
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
```

#### 1-2. VA-API を使用する場合

Ubuntu 24.04 / 26.04 の標準リポジトリから VA ドライバを導入できます。iHD は `intel-media-va-driver-non-free` (multiverse、推奨) または free 版の `intel-media-va-driver` を使用します。

```Shell
sudo apt install --no-install-recommends libva2 libva-drm2 libva-x11-2 intel-media-va-driver-non-free vainfo
```

free 版を使う場合は、上のパッケージ名を `intel-media-va-driver` に置き換えてください。対応範囲は GPU とドライバの版で変わります。Ubuntu 24.04 の i3-N305 で確認した free 版 24.1 は、H.264 / HEVC の EncSliceLP のみで、ICQ / AVBR は非対応でした。同じ版の non-free 版では EncSlice / EncSliceLP と ICQ、H.264 の AVBR に対応しました。実際の能力は 5. の `--check-features` で確認してください。

i965 を使う旧世代では、実機でエンコードを確認した `i965-va-driver-shaders` (multiverse、ビルド済みシェーダ入り) を導入します。universe の `i965-va-driver` はシェーダを含まず、エンコードできない世代がありうるため、ここでは推奨しません（動作未確認）。

```Shell
sudo apt install --no-install-recommends libva2 libva-drm2 libva-x11-2 i965-va-driver-shaders vainfo
```

OpenCL フィルタと、avsw 入力の途中での解像度変更への対応には、GPU に対応する OpenCL ランタイムも必要です。フィルタなしで、解像度が変わらない入力をエンコードするだけなら不要です。

```Shell
sudo apt install intel-opencl-icd clinfo
```

Ubuntu 26.04 の Gen9 (HD 630 など) は、通常の `intel-opencl-icd` の代わりに、標準リポジトリ (universe) の [intel-opencl-icd-legacy](https://packages.ubuntu.com/resolute/intel-opencl-icd-legacy) を使用します。

```Shell
sudo apt install intel-opencl-icd-legacy clinfo
```

26.04 の OpenCL 実機確認は kobuk-team PPA の `intel-opencl-icd` 26.31 と、[3-1.](#3-1-gen11以前とgen12以降のgpuを併存させる場合) の Intel 配布ランタイムで行っています。標準の `intel-opencl-icd` 26.05 / `intel-opencl-icd-legacy` での動作は未確認です。

代わりに Mesa の rusticl (`sudo apt install mesa-opencl-icd clinfo`) も使用できます。`qsvencc` は `RUSTICL_ENABLE` が未設定なら `iris` を自動設定します。ただし、確認した環境では Intel OpenCL より大幅に遅くなりました。GPU に対応する OpenCL がない場合はフィルタなしで使用してください。

`libva-x11-2` は公式 deb の依存パッケージに含まれるため、画面を使わない場合も必要です。公式 deb は Ubuntu 20.04 をベースにビルドし、実行時はシステムの libva を使用します。ソースからビルドする場合は、実行環境に対応する libva を使用し、`ldd ./qsvencc` が示す追加の共有ライブラリも導入してください。ビルド方法は [こちら](./Build.ja.md) を参照してください。

### 2. GPU を使うため、ユーザーを下記グループに追加

QSV / VA-API / OpenCL の利用には、`video` と `render` グループの設定が必要です。変更後はログインし直して反映してください。

```Shell
# QSV
sudo gpasswd -a ${USER} video
# OpenCL
sudo gpasswd -a ${USER} render
```

### 3. qsvenccのインストール

公式配布は Ubuntu 20.04 をベースにビルドした 1 つの deb です。Ubuntu 20.04 以降（26.04 を含む）と、必要な依存パッケージを導入できる apt 系ディストリビューションで共通の deb を使用します。GPU に対応するドライバ・ランタイムは別途必要です。

qsvenccのdebファイルを[こちら](https://github.com/rigaya/QSVEnc/releases)からダウンロードします。

その後、下記のようにインストールします。"x.xx"はインストールするバージョンに置き換えてください。

```Shell
sudo apt install ./qsvencc_x.xx_amd64.deb
```

公式 deb の QSV / OpenCL ランタイムは Recommends (推奨パッケージ) です。VA-API だけで使用する場合は、1-2. のドライバを導入してから `sudo apt install --no-install-recommends ./qsvencc_x.xx_amd64.deb` で推奨パッケージの自動導入を省略できます。ファイル名は使用する配布パッケージに合わせてください。

### 3-1. Gen11以前とGen12以降のGPUを併存させる場合

以下は Intel 配布の `intel-opencl-icd-legacy1` を手動導入する手順です。Ubuntu 26.04 の HD 630 でも、この方法で OpenCL を確認しています。標準リポジトリの `intel-opencl-icd-legacy` とはパッケージ名が異なります。

Ubuntu 22.04 / 24.04 で、Gen11以前 (例: Kaby LakeのiGPU) とGen12以降 (例: Arc dGPU) が同じPCに混在する場合、Ubuntu標準repoのintel-opencl-icdではGen12以降の新しいGPUに対応できず、Intel repositoryのintel-opencl-icdではGen11以前のOpenCLデバイスを検出できません。

この場合は、1. の手順でIntel repositoryを登録してGen12以降向けのintel-opencl-icdを導入したうえで、Intelが公開しているGen11以前向けのOpenCLランタイム (legacy1) を追加します。legacy1は通常のintel-opencl-icdとは別パッケージ・別ICD (```/etc/OpenCL/vendors/intel_legacy1.icd```) となっており、併存させることができます。なお、複数のIntel OpenCLプラットフォームが存在する環境に対応したQSVEncC 8.31以降が必要です。

```Shell
mkdir -p ~/neo-legacy1 && cd ~/neo-legacy1
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.24/intel-igc-core_1.0.17537.24_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.24/intel-igc-opencl_1.0.17537.24_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.35.30872.36/intel-opencl-icd-legacy1_24.35.30872.36_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.35.30872.36/ww35.sum

# チェックサムの確認
grep intel-opencl-icd-legacy1_ ww35.sum | sha256sum -c -

sudo apt install ./intel-igc-core_1.0.17537.24_amd64.deb ./intel-igc-opencl_1.0.17537.24_amd64.deb ./intel-opencl-icd-legacy1_24.35.30872.36_amd64.deb

# intel-igc-* は /usr/local/lib に導入されるため、ライブラリキャッシュを更新
sudo ldconfig
```

導入後、```clinfo -l``` でGen11以前とGen12以降の両方のGPUが表示されることを確認してください。

### 3-2. VA-API / OpenCL の認識状況の確認

`vainfo` で、対象 GPU のコーデックに `VAEntrypointEncSlice` または `VAEntrypointEncSliceLP` が表示されることを確認します。GPU が複数ある場合は、`renderD128` を対象ノード（`renderD129` など）に置き換えてください。

```Shell
vainfo --display drm --device /dev/dri/renderD128
clinfo -l
```

i965 を使う場合は `LIBVA_DRIVER_NAME=i965` を指定します。この指定は `qsvencc` にも使用します。rusticl を `clinfo` で確認する場合は、自動設定されないため `RUSTICL_ENABLE=iris` を指定してください。

```Shell
LIBVA_DRIVER_NAME=i965 vainfo --display drm --device /dev/dri/renderD128
RUSTICL_ENABLE=iris clinfo -l
```

### 4. 追加オプション
下記機能を使用するには、追加でインストールが必要です。

- avs読み込み  
  [AvisynthPlus](https://github.com/AviSynth/AviSynthPlus)のインストールが必要です。
  
- vpy読み込み
  [VapourSynth](https://www.vapoursynth.com/)のインストールが必要です。

- --vpp-onnx
  OpenVINO Runtimeのインストールが必要です。Ubuntuでは[Intel公式APTリポジトリ](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-apt.html)から導入します。

  ```Shell
  sudo apt-get install -y gnupg wget
  wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
  sudo gpg --output /etc/apt/trusted.gpg.d/intel.gpg --dearmor GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

  # 使用しているUbuntuのバージョンに合わせて、下記のいずれか1行のみ実行します。

  # Ubuntu 24.04の場合
  echo "deb https://apt.repos.intel.com/openvino ubuntu24 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list

  # Ubuntu 22.04の場合
  # echo "deb https://apt.repos.intel.com/openvino ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list

  sudo apt update
  sudo apt install openvino
  ```

### 5. qsvencc での認識状況の確認

VA-API を明示して、エンコードの可否、デバイス番号、詳細な能力を確認します。

```Shell
qsvencc --backend vaapi --check-hw
qsvencc --backend vaapi --check-device
qsvencc --backend vaapi --check-features
qsvencc --backend vaapi --avsw -i input.mp4 -c h264 --cqp 25 -o output.mp4
```

QSV の確認は `qsvencc --backend qsv --check-hw` を使用します。VA-API の `-d` は、VPL で使える GPU には QSV と同じ番号、それ以外には後ろの番号を割り当てます。VPL が使えない場合（無効指定、iHD 以外の指定、iHD がない場合を含む）は Intel の render node 順になります。バックエンドやドライバを切り替えたら、`--backend vaapi --check-device` で番号を確認してください。

```Shell
LIBVA_DRIVER_NAME=i965 qsvencc --backend vaapi --check-device
LIBVA_DRIVER_NAME=i965 qsvencc --backend vaapi -d 1 --avsw -i input.mp4 -c h264 --cqp 25 -o output.mp4
```

GPU を開けず `Permission denied` が出る場合は、2. の `render` / `video` グループの設定を確認してください。

### 6. その他

- qsvencc実行時に、"Failed to load OpenCL." というエラーが出る場合  
  /lib/x86_64-linux-gnu/libOpenCL.so が存在することを確認してください。 libOpenCL.so.1 しかない場合は、下記のようにシンボリックリンクを作成してください。
  
  ```Shell
  sudo ln -s /lib/x86_64-linux-gnu/libOpenCL.so.1 /lib/x86_64-linux-gnu/libOpenCL.so
  ```

- qsvenccでFixedFunction(FF)モードが使用できない
- Arc GPU / JasperLake 等でエンコードできない

  原因として、HuCファームウェアがロードされていないことが考えられます。[詳細](https://01.org/linuxgraphics/downloads/firmware)

  FixedFunction(FF)モード(別名 Low Powerモード)を使用するには、HuCファームウェアがロードされている必要があります。
  
  Arc GPU / JasperLakeでは、FFモードしか対応していないため、HuCファームウェアがロードされていないとQSVエンコードできません。

  HuCがロードされているかは、下記で確認できます。
  ```
  sudo cat /sys/kernel/debug/dri/0/i915_huc_load_status
  ```

  HuCのモジュールが存在するかは、下記で確認できます。
  ```
  sudo modinfo i915 | grep -i "huc"
  ```

  ご使用のCPUの世代に該当するモジュールがあれば、HuCファームウェアのロードを有効にすれば
  FixedFunctionモードを利用可能です。

  HuCファームウェアのロードを有効にするには、ファイル```/etc/modprobe.d/i915.conf```にカーネルパラメータを追加し、システムを再起動します。
  ```
  options i915 enable_guc=2
  ```  

## Linux (Fedora 32)

### 1. Intel Media ドライバとOpenCLランタイムのインストール  

```Shell
#Media
sudo dnf install intel-media-driver
#OpenCL
sudo dnf install -y 'dnf-command(config-manager)'
sudo dnf config-manager --add-repo https://repositories.intel.com/graphics/rhel/8.3/intel-graphics.repo
sudo dnf update --refresh
sudo dnf install intel-opencl intel-media intel-mediasdk level-zero intel-level-zero-gpu
```

### 2. QSVとOpenCLの使用のため、ユーザーを下記グループに追加
```Shell
# QSV
sudo gpasswd -a ${USER} video
# OpenCL
sudo gpasswd -a ${USER} render
```

### 3. qsvenccのインストール
qsvenccのrpmファイルを[こちら](https://github.com/rigaya/QSVEnc/releases)からダウンロードします。

その後、下記のようにインストールします。"x.xx"はインストールするバージョンに置き換えてください。

```Shell
sudo dnf install ./qsvencc_x.xx_1.x86_64.rpm
```

### 4. 追加オプション
下記機能を使用するには、追加でインストールが必要です。

- avs読み込み  
  [AvisynthPlus](https://github.com/AviSynth/AviSynthPlus)のインストールが必要です。
  
- vpy読み込み  
  [VapourSynth](https://www.vapoursynth.com/)のインストールが必要です。

### 5. その他

- qsvencc実行時に、"Failed to load OpenCL." というエラーが出る場合  
  /lib/x86_64-linux-gnu/libOpenCL.so が存在することを確認してください。 libOpenCL.so.1 しかない場合は、下記のようにシンボリックリンクを作成してください。
  
  ```Shell
  sudo ln -s /lib/x86_64-linux-gnu/libOpenCL.so.1 /lib/x86_64-linux-gnu/libOpenCL.so
  ```
