
# QSVEncCのインストール方法

- [Windows](./Install.ja.md#windows)
- Linux
  - [Linux (Ubuntu 20.04 - 24.04)](./Install.ja.md#linux-ubuntu-2004---2404)
  - [Linux (Fedora 32)](./Install.ja.md#linux-fedora-32)
  - その他のLinux OS  
    その他のLinux OS向けには、ソースコードからビルドする必要があります。ビルド方法については、[こちら](./Build.ja.md)を参照してください。


## Windows 

### 1. Intelグラフィックスドライバをインストールします。
### 2. Windows用実行ファイルをダウンロードして展開します。  
実行ファイルは[こちら](https://github.com/rigaya/QSVEnc/releases)からダウンロードできます。QSVEncC_x.xx_Win32.7z が 32bit版、QSVEncC_x.xx_x64.7z が 64bit版です。通常は、64bit版を使用します。

実行時は展開したフォルダからそのまま実行できます。

64bit版の配布archiveには`libvmaf.dll`とNVIDIA backend版の`libvship.dll`が含まれます。VMAF評価はCPUで実行できますが、同梱のlibvshipによる評価にはNVIDIA GPUと対応ドライバが必要です。評価を使用しない通常のエンコードにはこれらのDLLは不要です。

## Linux (Ubuntu 22.04 - 24.04)

公式配布パッケージではlibvmafを静的リンクするため、VMAF評価に`libvmaf.so`は不要です。libvship評価を使用する場合は、実行時ローダーが`libvship.so`と、選択したbackendが必要とする共有ライブラリを検索できる場所へ配置します（システムのライブラリ検索パス、または`LD_LIBRARY_PATH`）。ソースから通常設定でビルドした場合はlibvmafを動的ロードするため、VMAF評価には`libvmaf.so`も必要です。評価を使用しない通常のエンコードには、これらのライブラリは不要です。

### 1. Intel Media ドライバ用のリポジトリの登録

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

### 2. QSVとOpenCLの使用のため、ユーザーを下記グループに追加
```Shell
# QSV
sudo gpasswd -a ${USER} video
# OpenCL
sudo gpasswd -a ${USER} render
```

### 3. qsvenccのインストール
qsvenccのdebファイルを[こちら](https://github.com/rigaya/QSVEnc/releases)からダウンロードします。

その後、下記のようにインストールします。"x.xx"はインストールするバージョンに置き換えてください。

```Shell
# Ubuntu 24.04
sudo apt install ./qsvencc_x.xx_Ubuntu24.04_amd64.deb

# Ubuntu 22.04
sudo apt install ./qsvencc_x.xx_Ubuntu22.04_amd64.deb
```

### 3-1. Gen11以前とGen12以降のGPUを併存させる場合

Gen11以前 (例: Kaby LakeのiGPU) とGen12以降 (例: Arc dGPU) が同じPCに混在する場合、Ubuntu標準repoのintel-opencl-icdではGen12以降の新しいGPUに対応できず、Intel repositoryのintel-opencl-icdではGen11以前のOpenCLデバイスを検出できません。

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

### 3-2. VA-API だけで動かす場合に必要なパッケージ

`--backend vaapi` または `auto` の VA-API 経路だけを使用する場合、oneVPL / Media SDK のランタイム (`libvpl2`、`libmfx-gen1.2` など) は不要です。Ubuntu 24.04 の標準リポジトリでは、VA-API のライブラリと Intel のドライバを次のように導入できます。

```Shell
sudo apt install --no-install-recommends libva2 libva-drm2 libva-x11-2 intel-media-va-driver
```

`libva-x11-2` は既存のビルド済みバイナリのリンク依存に含まれるため、画面を使わない場合も導入します。ソースからビルドしたバイナリは、`ldd ./qsvencc` で表示される追加の共有ライブラリも必要です。公式 deb パッケージでは QSV / OpenCL 用のランタイムは Recommends (推奨パッケージ) です。VA-API だけで使う場合は、上記のドライバを導入したうえで `sudo apt install --no-install-recommends ./qsvencc_x.xx_Ubuntu24.04_amd64.deb` とすると、推奨パッケージの自動導入を省略できます。

ビルド時と実行時の libva の互換性も必要です。新しい libva でビルドして `vaMapBuffer2` を参照するバイナリは、Ubuntu 24.04 標準の libva 2.20 では起動できません。その場合は対応する libva を用意するか、実行環境の libva に合わせてビルドしてください。

OpenCL フィルタも使用する場合は、GPU に対応する OpenCL ランタイム (例: `intel-opencl-icd`) を追加してください。フィルタなしなら不要です。`/dev/dri/renderD*` にアクセスできるよう、2. のグループ設定も行ってください。

free 版の `intel-media-va-driver` と non-free 版ではエンコードの対応範囲が異なる場合があります。`qsvencc --backend vaapi --check-features` で実際のドライバの能力を確認してください。

```Shell
qsvencc --backend vaapi --check-features
qsvencc --backend vaapi --avsw -i input.mp4 -c h264 --cqp 25 -o output.mp4
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

### 5. その他

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
