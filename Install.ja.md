
# QSVEncCのインストール方法

- [Windows](./Install.ja.md#windows)
- Linux
  - [Linux (Ubuntu 20.04)](./Install.ja.md#linux-ubuntu-2004)
  - その他のLinux OS  
    その他のLinux OS向けには、ソースコードからビルドする必要があります。ビルド方法については、[こちら](./Build.ja.md)を参照してください。


## Windows 

### 1. Intelグラフィックスドライバをインストールします。
### 2. Windows用実行ファイルをダウンロードして展開します。  
実行ファイルは[こちら](https://github.com/rigaya/QSVEnc/releases)からダウンロードできます。QSVEncC_x.xx_Win32.7z が 32bit版、QSVEncC_x.xx_x64.7z が 64bit版です。通常は、64bit版を使用します。

実行時は展開したフォルダからそのまま実行できます。
  
## Linux (Ubuntu 20.04)

### 1. Intel Media ドライバのインストール  
OpenCL関連は[こちらのリンク](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal.html)に従ってインストールします。

```Shell
sudo apt-get install -y gpg-agent wget
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | sudo apt-key add -
sudo apt-add-repository 'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main'
sudo apt-get update
sudo apt install intel-media-va-driver-non-free intel-opencl-icd intel-level-zero-gpu level-zero
```

### 2. QSVとOpenCLの使用のため、ユーザーを下記グループに追加
```Shell
# QSV
sudo gpasswd -a ${USER} video
# OpenCL
sudo gpasswd -a ${USER} render
```

### 3. QSVEncCのインストール
QSVEncCのdebファイルを[こちら](https://github.com/rigaya/QSVEnc/releases)からダウンロードします。

その後、下記のようにインストールします。"x.xx"はインストールするバージョンに置き換えてください。

```Shell
sudo apt install ./qsvencc_x.xx_Ubuntu20.04_amd64.deb
```

### 4. その他

- QSVEncC実行時に、"Failed to load OpenCL." というエラーが出る場合  
  /lib/x86_64-linux-gnu/libOpenCL.so が存在することを確認してください。 libOpenCL.so.1 しかない場合は、下記のようにシンボリックリンクを作成してください。
  
  ```Shell
  sudo ln -s /lib/x86_64-linux-gnu/libOpenCL.so.1 /lib/x86_64-linux-gnu/libOpenCL.so
  ```