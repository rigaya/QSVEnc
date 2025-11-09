# create-release-note

これから作成する新しいバージョンのユーザーに影響のある更新履歴を作成するタスクです。
なるべくひとつの更新については、git logのメッセージを使用して、簡潔に記載してください。
ただし、わかりにくい場合は少し改変や追記を行ってください。

すでに更新内容の記載がある場合は、重複しないように更新してください。

- 前のタグから最新版までのgit logから、更新内容を抽出する。
  - ユーザーに影響なさそうな小変更は無視すること。
  - githubのissue (#xxx)が含まれる変更は必ず含めること。
  - 「XXXの変更を反映」という変更は無視すること。

- QSVEnc/QSVEnc_readme.txt の ```【どうでもいいメモ】```の下に、更新内容を日本語で書く。
  新バージョンの記載がない場合は、他のバージョンと同様の見出しをつけましょう。

- ReleaseNotes.md の上部に、```バージョン番号```を見出しにして更新内容を英語で書く。
  - オプション名(--xxx)が出るときは、QSVEncC_Options.en.md へのリンクを作成する。
    オプションのリストとリンクは、QSVEncC_Options.en.md 上部の (## Command line exampleまで)に記載がある
　　例: [--vpp-afs](./QSVEncC_Options.en.md#--vpp-afs-param1value1param2value2)

- この変更のトピックとなる変更を1～2つピックアップし、100文字程度でまとめた文章例を(日本語で)作成する。
  - git logのメッセージの文体に近い形としつつ、ですます調で過去形にすること。

- ブログに記載する文章例を作成する。
  - QSVEnc/QSVEnc_readme.txt に記載した内容をHTML形式で記載。
  - 各項目については、見出しとして扱う必要はなく、```<strong>- XXXという機能を追加。</strong>(--option-name)```などのようにすること。
  - ```<br>```は不要。(かわりに通常の改行を用いる)
  - 最後に下記を記載する。
    ```
    <hr size="1" />
    <span style="color:#CC3333">※Aviutl向けには、Aviutl_QSVEnc_8.xx.zip をダウンロードしてください。</span>
    <a href="javascript:void(0);" onclick="javascript:openDownloadUrl('QSVEnc',0);" title="ダウンロード"><strong><u>QSVEnc ダウンロード>></u></strong></a>
    
    <a href="https://github.com/rigaya/QSVEnc/blob/master/QSVEnc_auo_readme.md"title="QSVEncの導入"><u>QSVEncの導入</u></a>
    
    QSVEncCのオプションについてはこちら。
    <a href="https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.ja.md" title="QSVEncCオプション一覧>"><u>QSVEncCオプション一覧></u></a>
    
    <hr size="1" />
    ```
