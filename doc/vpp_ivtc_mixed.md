# --vpp-ivtc mixed=on 設計

## 目的

日本の放送TS等に見られる「RFF区間（ソフトテレシネ由来の24000/1001fps）」と「TFF/BFF区間（本物のインタレース30000/1001fps）」が秒〜分単位で切り替わるストリームに対して、全体を24000/1001fps出力にする。

### 理想的な処理

- RFF区間: RFF/TFF/BFFに基づき repeat field を次フレームと組み合わせて再構築し、decimateせずCFR 24000/1001fps格子へemitする
- インタレ区間: field match + deinterlace を行い、インタレ区間内でcycle=5/drop=1を適用して30fps→24fps化する

## 前提

- 入力: `--avsw` デコード
- 入力コンテナ: 29.97fps (30000/1001fps) MPEG-2 TS
- libavcodecは1コーデッドフレーム→1出力フレーム（RFFフレームの複製は行わない）
- RFF区間のフレームは基本的にプログレッシブだが、混合TSでは repeat field を次フレーム側へずらして再構築しないと縞が見えるケースがある
- インタレ区間のフレームは`picstruct = FRAME_TFF / FRAME_BFF`

## 制約

- `expand=on` との併用不可（排他）
- `cycle` パラメータはユーザー指定不可（mixed内部で自動決定: インタレ区間にcycle=5適用）
- 出力PTSは `baseFps=24000/1001` のグローバルCFRカウンタで生成する。入力PTSは初期seedと不連続検出に使い、RFF由来のduration揺れは出力タイムラインに混ぜない。

## フレーム区間判定

### 判定に使うメタデータ

avsw経路（`rgy_input_avcodec.cpp` L3524-3534）でフレームに付与されるフラグ:

| フラグ | 意味 |
|---|---|
| `RGY_FRAME_FLAG_RFF` | `findPos.repeat_pict > 1` のフレーム（RFF付き） |
| `RGY_FRAME_FLAG_RFF_TFF` | RFF区間内でTFF表示のフレーム |
| `RGY_FRAME_FLAG_RFF_BFF` | RFF区間内でBFF表示のフレーム |

`decRFFStatus` は `RGY_FRAME_FLAG_RFF` が来るたびにトグルされ、RFF区間内のRFF=0フレームにも `RGY_FRAME_FLAG_RFF_TFF/BFF` を付与する。ただしこの持ち越しフラグはインタレ区間へ漏れることがあるため、`picstruct` による上書き判定が必要。

### 判定ロジック

```cpp
bool isRffSection(const RGYFrameInfo *frame) {
    const auto flags = frame->flags;
    if (flags & RGY_FRAME_FLAG_RFF) return true;
    if ((flags & (RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF)) == 0) return false;
    return frame->picstruct == RGY_PICSTRUCT_FRAME;
}

bool isInterlacedSection(const RGYFrameInfo *frame) {
    if (isRffSection(frame)) return false;
    const auto ps = frame->picstruct;
    return (ps == RGY_PICSTRUCT_FRAME_TFF || ps == RGY_PICSTRUCT_FRAME_BFF);
}
```

- RFF区間フレーム (`isRffSection` = true): RFF再構築対象。`RGY_FRAME_FLAG_RFF` は常にRFF扱い、`RFF_TFF/BFF` のみの場合は `picstruct=FRAME` のときだけRFF扱い
- インタレ区間フレーム (`isInterlacedSection` = true): deint + decimate対象
- どちらにも該当しない (progressive, RFF無し): passthrough対象

### 境界の挙動

`decRFFStatus` のトグルにより、RFF区間→インタレ区間の遷移で `RFF_TFF/BFF` だけが残ることがある。この場合でも `picstruct=TFF/BFF/FRAME_TFF/FRAME_BFF` ならインタレ区間を優先する。repeat field pending は interlaced 区間へ入る時点で破棄し、別区間のフィールドを混ぜない。

## 処理フロー

```
入力フレーム
    │
    ├─ isRffSection(frame) == true
    │   └─ RFF区間パス: repeat field pending と現在フレームから再構築してdirect emit
    │
    └─ isInterlacedSection(frame) == true
        └─ インタレ区間パス: field match + BWDIF deint → cycle bufferへ追加
            └─ インタレ区間で5フレーム蓄積 → flushCycleMixed (SAD最小1フレームドロップ) → 4フレーム emit
```

## RFF区間パス

### フレーム処理

- ピクセル内容: field match / deinterlace はスキップし、1入力フレームにつき1出力フレームをdirect emitする
- RFF repeat field pending があれば、現在フレームのcopyと top/bottom field 再構築候補の両方を評価し、combが少ない候補を採用する
- pending がないフレーム、または現在フレームcopyがcleanな場合は `RFF_RECON_COPY` としてdirect emitする
- RFF区間は `33ms → 50ms → 33ms → 50ms` 相当の表示時間揺れを `40ms → 40ms → 40ms → 40ms` に正規化するだけで、5→4 decimateは行わない

### RFFフィールド再構築

- `RGY_FRAME_FLAG_RFF` が立ったフレームは、`RGY_FRAME_FLAG_RFF_TFF/BFF` または `picstruct` から repeat top / repeat bottom を判定し、repeat field を pending として保持する
- 次のRFF区間フレームで pending があれば、pending field と現在フレームの相補 field を合成して `RFF_RECON_FIELD` 候補を作る
- copy候補とfield再構築候補は両parityでcomb量を測定し、copy候補がcleanでない、かつfield再構築候補が明確にcombを減らす場合だけ `RFF_RECON_FIELD` を採用する
- これにより、すでにprogressiveにdecodeされているRFFフレームへ別時刻のfieldを混ぜて縞を作ることを避ける
- interlaced 区間またはPTS不連続に入る時は pending を破棄し、異なる時間/区間の field を混ぜない

### タイムスタンプ補正

RFF区間の入力フレームはdurationが不均一（RFF付き=1.5倍、RFF無し=1.0倍）。

補正方式: **グローバルCFRカウンタ**

最初の有効入力PTSをseedにし、emitごとに `24000/1001fps` の周期でPTS/durationを生成する。RFF区間は表示フレーム数をdecimateで減らさず、RFF由来の不均一durationだけをCFRへ正規化する。

## インタレ区間パス

### フレーム処理

1. **field match**: 既存IVTCの scoreCandidates で C/P/N を評価し最適マッチ選択
2. **combing detection + BWDIF deinterlace**: combedフレームに対して adaptive deint
3. **cycle buffer蓄積**: 処理済みフレームを cycle=5 バッファに積む
4. **decimate**: 5フレーム揃ったら SAD 最小のフレームを1つドロップ → 4フレーム emit

### 適用オプション

- `guide`, `post`, `cadlock`, `combthresh`, `dthresh` 等 → インタレ区間にのみ適用

### タイムスタンプ

emit順に `24000/1001fps` のCFR PTSを割り当てる:

```
emit[i].pts      = seed_pts + rescale(global_emit_index, 1001/24000, timebase)
emit[i].duration = rescale(global_emit_index + 1, 1001/24000, timebase) - rescale(global_emit_index, 1001/24000, timebase)
```

## 区間遷移の処理

### 区間切替

RFF/progressive と interlaced の区間切替では、interlaced cycle に残っているフレームを **dropなし** で partial flush してから direct emit 側へ切り替える。RFF区間を interlaced cycle に混ぜて5→4 dropすると尺が短くなるため、cycleはインタレ区間だけで閉じる。

この境界partial flushにより、切替直前のインタレ区間では5枚未満の端数がdropされずに残ることがある。これは境界での大きなPTS破綻を避けるための処理であり、drop cycleの位相が境界でリセットされるぶん、素材によっては最終フレーム数が数フレーム単位で変動し得る。

### PTS不連続によるリセット

TSではフレームdropやストリーム欠損により、フレーム間のPTSが大きく飛ぶことがある。

連続する2フレーム間のPTS差が 30000/1001fps の1フレーム期間（≈33.37ms）の2倍（≈66.7ms）を超えた場合、**状態をリセット**する:

- cycle buffer内のフレームを partial flush（ドロップなしで全emit）
- cadence tracker リセット
- 次フレームから新規にcycle蓄積を開始

これにより、PTS飛びをまたいで無関係なフレームをcycleに混ぜてしまうことを防ぐ。出力PTS自体はCFRカウンタを継続し、入力側のPTS揺れやRFF由来の中間PTSを下流へ伝搬させない。

## 出力フレームレート

- 全体として24000/1001fpsに近い出力になる
- RFF区間: field reconstruction + direct CFR emit → 24fps
- インタレ区間: field match/deint + decimate → 24fps
- baseFps は 24000/1001 として報告（下流のエンコーダに通知）

## パラメータ

```
--vpp-ivtc mixed=on[,guide=1][,post=2][,cadlock=auto][,combthresh=0.12][,dthresh=7][,log=<path>]
```

- `mixed`: on/off。デフォルトoff。
- `expand`: mixed=on時は指定不可（内部でoff扱い）
- `cycle`: mixed=on時は指定不可（インタレ区間に内部でcycle=5適用）
- その他のパラメータ: インタレ区間のfield match / deinterlace処理に適用

## 実装箇所

### 新規追加

- `VppIvtc` 構造体に `int mixed` フィールド追加 (`rgy_prm.h`)
- コマンドライン解析に `mixed` 追加 (`rgy_cmd.cpp`)

### 主要改変

- `rgy_filter_ivtc.cpp`:
  - `processInputToCycle()` に区間判定ロジック追加
  - RFF区間パス: repeat field pending を使って再構築したフレームを direct emit queue に追加
  - インタレ区間パス: 既存 field match + deint を流用して mixed cycle slot に追加
  - `flushCycleMixed()` でグローバルCFRカウンタにより出力PTSを生成

### 流用可能な既存コード

- `scoreCandidates()`: そのまま流用
- `synthesizeToCycleBwdif()`: そのまま流用
- `computePairDiff()` (SAD計算): そのまま流用
- emit queue / popEmit: そのまま流用
- per-frame TSV log: 区間種別カラム追加

## expand=on との違い

| 観点 | expand=on | mixed=on |
|---|---|---|
| 対象入力 | 純粋ソフトテレシネDVD | 混合TS (RFF/インタレ切替) |
| RFF区間 | 30fpsに展開→decimate | repeat field再構築→direct CFR emit |
| インタレ区間 | pending stateリークで破壊 | 正しくdeint+decimate |
| pre-scan | 必要 | 不要（フレーム単位判定） |
| 区間判定 | なし（全編一律） | RGY_FRAME_FLAG_RFF系フラグで判定 |
