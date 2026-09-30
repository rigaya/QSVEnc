
# QSVEncCのインストール方法

- [Windows](./Install.ja.md#windows)
- Linux
  - [Linux (Ubuntu 20.04以降)](./Install.ja.md#linux-ubuntu-2004以降)
  - [Linux (Fedora 32)](./Install.ja.md#linux-fedora-32)
  - その他のLinux OS  
    配布debを利用できないLinux OS向けには、ソースコードからビルドする必要があります。ビルド方法については、[こちら](./Build.ja.md)を参照してください。


## Windows 

### 1. Intelグラフィックスドライバをインストールします。
### 2. Windows用実行ファイルをダウンロードして展開します。  
実行ファイルは[こちら](https://github.com/rigaya/QSVEnc/releases)からダウンロードできます。QSVEncC_x.xx_Win32.7zが32bit版、QSVEncC_x.xx_x64.7zが64bit版です。通常は、64bit版を使用します。

実行時は展開したフォルダからそのまま実行できます。

64bit版の配布archiveには`libvmaf.dll`が含まれます。VMAF評価はCPUで実行できますが、同梱のlibvshipによる評価にはNVIDIA GPUと対応ドライバが必要です。評価を使用しない通常のエンコードにはこれらのDLLは不要です。

## Linux (Ubuntu 20.04以降)

Linuxでは、Intel GPUのエンコードにQSVとVA-APIの2種類を使えます。

| 方法 | QSV | VA-API |
|:--|:--|:--|
| 必要なランタイム | Tiger Lake以降はVPL (`libmfx-gen`)、それ以前の対応GPUはMedia SDK (`libmfx1`) | VAドライバ(`iHD` / `i965`)のみ。VPL / Media SDKは不要 |
| HWデコード(`--avhw`) | 可 | 不可。`--avsw`と入力の自動選択はソフトウェアデコード |
| フィルタ | MFX VPP / OpenCL | OpenCLのみ。OpenCLがなければフィルタなしのエンコード |
| エンコード設定 | MFXの詳細設定に対応 | レート制御・GOP・品質などの基本設定のみ、一部のオプションは非対応 |

詳細な対応範囲は[--backend](./QSVEncC_Options.ja.md#--backend-autoqsvvaapi)を参照してください。

VPLランタイムがないGPU / ディストリビューションや、i965しか使えない旧世代ではVA-APIを使用してください。例えばUbuntu 26.04には`libmfx1`がなく、Gen11以前のiGPUはVPLランタイムがない環境になります。

既定の`--backend auto`はQSVを試し、デバイスを利用できなければVA-APIに切り替えます。次の場合はQSVを試さずVA-APIを選択します。

- `QSVENC_VPL_DISABLE=1`を指定した場合。
- `LIBVA_DRIVER_NAME`に`iHD`以外を指定した場合(大文字小文字は無視)。
- ドライバの探索先に`iHD_drv_video.so`が見つからない場合(i965のみの環境など)。

VPLランタイムが起動時にクラッシュする環境では、`QSVENC_VPL_DISABLE=1 qsvencc ...`でVPLの読み込みを回避してください。VA-APIを明示するには`--backend vaapi`、QSVを明示するには`--backend qsv`を指定します。

### 1. ドライバ

GPUの世代によって、使用できる方法と必要なパッケージが異なります。

| GPU | QSV | VA-API |
|:--|:--|:--|
| Gen12以降<br>(Tiger Lake, Rocket Lake, Alder Lake, Raptor Lake, Arc dGPUなど) | iHD + VPL | iHD |
| Gen8〜Gen11<br>(Broadwell, Skylake, Kaby Lake, Coffee Lake, Apollo Lake, Gemini Lake, Ice Lake, Elkhart Lakeなど) | iHD + Media SDK<br>(Ubuntu 26.04は`libmfx1`がないため不可) | iHD |
| Haswell以前 | 不可 | i965 |

パッケージの導入元は、Ubuntuのバージョンで分けます。

| Ubuntu | 導入元 |
|:--|:--|
| 22.04 / 24.04 | Intelのリポジトリ(1-1.で登録) |
| 26.04 | Ubuntu標準のリポジトリ |

#### 1-1. Intelのリポジトリの登録 (Ubuntu 22.04 / 24.04)

[こちらのリンク](https://dgpu-docs.intel.com/driver/client/overview.html)に沿って、リポジトリを登録します。GPUの世代によらず登録して問題ありません(Gen11以前のOpenCLは2-1.の表を参照)。

```Shell
sudo apt-get install -y gpg-agent wget

# Ubuntu 24.04
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list

# Ubuntu 22.04
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

sudo apt update
```

#### 1-2. ドライバの導入

1.の表で必要なものを導入します。

| 対象 | パッケージ |
|:--|:--|
| iHD | `intel-media-va-driver-non-free` |
| VPL | `libmfxgen1` (Intelのリポジトリ) / `libmfx-gen1.2` (Ubuntu標準) |
| Media SDK | `libmfx1` |
| i965 | `i965-va-driver-shaders` |

```Shell
# 共通
sudo apt install libva2 libva-drm2 libva-x11-2 vainfo

# Gen12以降 (QSV / VA-API)
sudo apt install intel-media-va-driver-non-free libmfxgen1      # 22.04 / 24.04 (Intelのリポジトリ)
sudo apt install intel-media-va-driver-non-free libmfx-gen1.2   # 26.04

# Gen8〜Gen11
sudo apt install intel-media-va-driver-non-free libmfx1         # 22.04 / 24.04 (QSV / VA-API)
sudo apt install intel-media-va-driver-non-free                 # 26.04 (VA-APIのみ。libmfx1がないため)

# Haswell以前 (VA-APIのみ)
sudo apt install i965-va-driver-shaders
```

i965は`i965-va-driver-shaders`(ビルド済みシェーダ入り)を使用します。

### 2. OpenCL

OpenCLフィルタを使用するには、GPUに対応するOpenCLランタイムが必要です。GPUに対応するOpenCLがない場合は、フィルタなしで使用してください。

#### 2-1. Intel (推奨)

GPUの世代とUbuntuのバージョンで、使用するパッケージが異なります。

| GPU | 22.04 / 24.04 (Intelのリポジトリ) | 26.04 |
|:--|:--|:--|
| Gen12以降 | `intel-opencl-icd` | `intel-opencl-icd` |
| Gen9〜Gen11 | legacy1 (Intel配布、下記の手順で手動導入) | `intel-opencl-icd-legacy` |
| 混在 | 上の2つを併用 | 上の2つを併用 |
| Gen8以前 | なし | なし |

Intelのリポジトリの`intel-opencl-icd`はGen12以降専用のため、Gen9〜Gen11はlegacy系のランタイムを別に導入します。legacy系はGen12以降向けとは別パッケージ・別ICDで、併存させることができます。複数のIntel OpenCLプラットフォームが存在する環境に対応したQSVEncC 8.31以降が必要です。

```Shell
# Gen12以降
sudo apt install intel-opencl-icd clinfo

# Gen9〜Gen11 (26.04)
sudo apt install intel-opencl-icd-legacy clinfo
```

22.04 / 24.04のGen9〜Gen11向けのlegacy1は、下記のように導入します。Ubuntu標準のリポジトリの`intel-opencl-icd-legacy`とはパッケージ名が異なります。

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

導入後、`clinfo -l`で対象のGPUが表示されることを確認してください。

#### 2-2. Mesa

MesaのOpenCL(rusticl)も使用できます。ただし、確認した環境ではIntelのOpenCLより大幅に遅くなりました。

```Shell
sudo apt install mesa-opencl-icd clinfo
```

rusticlはIntel GPUが既定で無効になっていますが、`qsvencc`は`RUSTICL_ENABLE`が未設定なら`iris`を自動設定します。

### 3. グループ追加

QSV / VA-API / OpenCLの利用には、`video`と`render`グループの設定が必要です。変更後はログインし直して反映してください。

```Shell
sudo gpasswd -a ${USER} video
sudo gpasswd -a ${USER} render
```

### 4. qsvenccのインストール

Ubuntu 20.04以降(26.04を含む)と、必要な依存パッケージを導入できるapt系ディストリビューションで共通のdebを使用します。

qsvenccのdebファイルを[こちら](https://github.com/rigaya/QSVEnc/releases)からダウンロードし、下記のようにインストールします。"x.xx"はインストールするバージョンに置き換えてください。

```Shell
sudo apt install ./qsvencc_x.xx_amd64.deb
```

### 5. QSV / VA-API / OpenCLの認識状況の確認

`vainfo`で、対象GPUのコーデックに`VAEntrypointEncSlice`または`VAEntrypointEncSliceLP`が表示されることを確認します。GPUが複数ある場合は、`renderD128`を対象ノード(`renderD129`など)に置き換えてください。OpenCLは`clinfo -l`で確認します。

```Shell
vainfo --display drm --device /dev/dri/renderD128
clinfo -l
```

i965を使う場合は`LIBVA_DRIVER_NAME=i965`を指定します。この指定は`qsvencc`にも使用します。rusticlを`clinfo`で確認する場合は、自動設定されないため`RUSTICL_ENABLE=iris`を指定してください。

```Shell
LIBVA_DRIVER_NAME=i965 vainfo --display drm --device /dev/dri/renderD128
RUSTICL_ENABLE=iris clinfo -l
```

qsvenccで、エンコードの可否、デバイス番号、詳細な能力を確認します。

```Shell
# QSV
qsvencc --backend qsv --check-hw
qsvencc --backend qsv --check-device
qsvencc --backend qsv --check-features

# VA-API
qsvencc --backend vaapi --check-hw
qsvencc --backend vaapi --check-device
qsvencc --backend vaapi --check-features
```

VA-APIの`-d`は、VPLで使えるGPUにはQSVと同じ番号、それ以外には後ろの番号を割り当てます。VPLが使えない場合(無効指定、iHD以外の指定、iHDがない場合を含む)はIntelのrender node順になります。バックエンドやドライバを切り替えたら、`--backend vaapi --check-device`で番号を確認してください。

```Shell
LIBVA_DRIVER_NAME=i965 qsvencc --backend vaapi --check-device
LIBVA_DRIVER_NAME=i965 qsvencc --backend vaapi -d 1 --avsw -i input.mp4 -c h264 --cqp 25 -o output.mp4
```

GPUを開けず`Permission denied`が出る場合は、3.の`render` / `video`グループの設定を確認してください。

### 6. 追加オプション
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

### 7. その他

- qsvencc実行時に、"Failed to load OpenCL."というエラーが出る場合  
  /lib/x86_64-linux-gnu/libOpenCL.soが存在することを確認してください。libOpenCL.so.1しかない場合は、下記のようにシンボリックリンクを作成してください。
  
  ```Shell
  sudo ln -s /lib/x86_64-linux-gnu/libOpenCL.so.1 /lib/x86_64-linux-gnu/libOpenCL.so
  ```

- qsvenccでFixedFunction(FF)モードが使用できない
- Arc GPU / JasperLake等でエンコードできない

  原因として、HuCファームウェアがロードされていないことが考えられます。[詳細](https://01.org/linuxgraphics/downloads/firmware)

  FixedFunction(FF)モード(別名Low Powerモード)を使用するには、HuCファームウェアがロードされている必要があります。
  
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

### 1. Intel MediaドライバとOpenCLランタイムのインストール  

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

- qsvencc実行時に、"Failed to load OpenCL."というエラーが出る場合  
  /lib/x86_64-linux-gnu/libOpenCL.soが存在することを確認してください。libOpenCL.so.1しかない場合は、下記のようにシンボリックリンクを作成してください。
  
  ```Shell
  sudo ln -s /lib/x86_64-linux-gnu/libOpenCL.so.1 /lib/x86_64-linux-gnu/libOpenCL.so
  ```
