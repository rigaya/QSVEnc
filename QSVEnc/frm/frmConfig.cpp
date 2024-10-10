// -----------------------------------------------------------------------------------------
// x264guiEx/x265guiEx/svtAV1guiEx/ffmpegOut/QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2010-2022 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// --------------------------------------------------------------------------------------------

//以下warning C4100を黙らせる
//C4100 : 引数は関数の本体部で 1 度も参照されません。
#pragma warning( push )
#pragma warning( disable: 4100 )

#include "auo_version.h"
#include "auo_frm.h"
#include "auo_faw2aac.h"
#include "frmConfig.h"
#include "frmSaveNewStg.h"
#include "frmOtherSettings.h"
#include "frmBitrateCalculator.h"
#include "mfxstructures.h"

using namespace AUO_NAME_R;

/// -------------------------------------------------
///     設定画面の表示
/// -------------------------------------------------
[STAThreadAttribute]
void ShowfrmConfig(CONF_GUIEX *conf, const SYSTEM_DATA *sys_dat) {
    if (!sys_dat->exstg->s_local.disable_visual_styles)
        System::Windows::Forms::Application::EnableVisualStyles();
    System::IO::Directory::SetCurrentDirectory(String(sys_dat->aviutl_dir).ToString());
    frmConfig frmConf(conf, sys_dat);
    frmConf.ShowDialog();
}

/// -------------------------------------------------
///     frmSaveNewStg 関数
/// -------------------------------------------------
System::Boolean frmSaveNewStg::checkStgFileName(String^ stgName) {
    String^ fileName;
    if (stgName->Length == 0)
        return false;

    if (!ValidiateFileName(stgName)) {
        MessageBox::Show(LOAD_CLI_STRING(AUO_CONFIG_ERR_INVALID_CHAR), LOAD_CLI_STRING(AUO_GUIEX_ERROR), MessageBoxButtons::OK, MessageBoxIcon::Error);
        return false;
    }
    if (String::Compare(Path::GetExtension(stgName), L".stg", true))
        stgName += L".stg";
    if (File::Exists(fileName = Path::Combine(fsnCXFolderBrowser->GetSelectedFolder(), stgName)))
        if (MessageBox::Show(stgName + LOAD_CLI_STRING(AUO_CONFIG_ALREADY_EXISTS), LOAD_CLI_STRING(AUO_CONFIG_OVERWRITE_CHECK), MessageBoxButtons::YesNo, MessageBoxIcon::Question)
            != System::Windows::Forms::DialogResult::Yes)
            return false;
    StgFileName = fileName;
    return true;
}

System::Void frmSaveNewStg::setStgDir(String^ _stgDir) {
    StgDir = _stgDir;
    fsnCXFolderBrowser->SetRootDirAndReload(StgDir);
}


/// -------------------------------------------------
///     frmBitrateCalculator 関数
/// -------------------------------------------------
System::Void frmBitrateCalculator::Init(int VideoBitrate, int AudioBitrate, bool BTVBEnable, bool BTABEnable, int ab_max, const AuoTheme themeTo, const DarkenWindowStgReader *dwStg) {
    guiEx_settings exStg(true);
    exStg.load_fbc();
    enable_events = false;
    dwStgReader = dwStg;
    CheckTheme(themeTo);
    fbcTXSize->Text = exStg.s_fbc.initial_size.ToString("F2");
    fbcChangeTimeSetMode(exStg.s_fbc.calc_time_from_frame != 0);
    fbcRBCalcRate->Checked = exStg.s_fbc.calc_bitrate != 0;
    fbcRBCalcSize->Checked = !fbcRBCalcRate->Checked;
    fbcTXMovieFrameRate->Text = Convert::ToString(exStg.s_fbc.last_fps);
    fbcNUMovieFrames->Value = exStg.s_fbc.last_frame_num;
    fbcNULengthHour->Value = Convert::ToDecimal((int)exStg.s_fbc.last_time_in_sec / 3600);
    fbcNULengthMin->Value = Convert::ToDecimal((int)(exStg.s_fbc.last_time_in_sec % 3600) / 60);
    fbcNULengthSec->Value =  Convert::ToDecimal((int)exStg.s_fbc.last_time_in_sec % 60);
    SetBTVBEnabled(BTVBEnable);
    SetBTABEnabled(BTABEnable, ab_max);
    SetNUVideoBitrate(VideoBitrate);
    SetNUAudioBitrate(AudioBitrate);
    enable_events = true;
}
System::Void frmBitrateCalculator::CheckTheme(const AuoTheme themeTo) {
    //変更の必要がなければ終了
    if (themeTo == themeMode) return;

    //一度ウィンドウの再描画を完全に抑止する
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 0, 0);
    SetAllColor(this, themeTo, this->GetType(), dwStgReader);
    SetAllMouseMove(this, themeTo);
    //一度ウィンドウの再描画を再開し、強制的に再描画させる
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 1, 0);
    this->Refresh();
    themeMode = themeTo;
}
System::Void frmBitrateCalculator::frmBitrateCalculator_FormClosing(System::Object^  sender, System::Windows::Forms::FormClosingEventArgs^  e) {
    guiEx_settings exStg(true);
    exStg.load_fbc();
    exStg.s_fbc.calc_bitrate = fbcRBCalcRate->Checked;
    exStg.s_fbc.calc_time_from_frame = fbcPNMovieFrames->Visible;
    exStg.s_fbc.last_fps = Convert::ToDouble(fbcTXMovieFrameRate->Text);
    exStg.s_fbc.last_frame_num = Convert::ToInt32(fbcNUMovieFrames->Value);
    exStg.s_fbc.last_time_in_sec = Convert::ToInt32(fbcNULengthHour->Value) * 3600
                                 + Convert::ToInt32(fbcNULengthMin->Value) * 60
                                 + Convert::ToInt32(fbcNULengthSec->Value);
    if (fbcRBCalcRate->Checked)
        exStg.s_fbc.initial_size = Convert::ToDouble(fbcTXSize->Text);
    exStg.save_fbc();
    frmConfig^ fcg = dynamic_cast<frmConfig^>(this->Owner);
    if (fcg != nullptr)
        fcg->InformfbcClosed();
}
System::Void frmBitrateCalculator::fbcRBCalcRate_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
    if (fbcRBCalcRate->Checked && Convert::ToDouble(fbcTXSize->Text) <= 0.0) {
        guiEx_settings exStg(true);
        exStg.load_fbc();
        fbcTXSize->Text = exStg.s_fbc.initial_size.ToString("F2");
    }
}
System::Void frmBitrateCalculator::fbcBTVBApply_Click(System::Object^  sender, System::EventArgs^  e) {
    frmConfig^ fcg = dynamic_cast<frmConfig^>(this->Owner);
    if (fcg != nullptr)
        fcg->SetVideoBitrate((int)fbcNUBitrateVideo->Value);
}
System::Void frmBitrateCalculator::fbcBTABApply_Click(System::Object^  sender, System::EventArgs^  e) {
    frmConfig^ fcg = dynamic_cast<frmConfig^>(this->Owner);
    if (fcg != nullptr)
        fcg->SetAudioBitrate((int)fbcNUBitrateAudio->Value);
}
System::Void frmBitrateCalculator::fbcMouseEnter_SetColor(System::Object^  sender, System::EventArgs^  e) {
    fcgMouseEnterLeave_SetColor(sender, themeMode, DarkenWindowState::Hot, dwStgReader);
}
System::Void frmBitrateCalculator::fbcMouseLeave_SetColor(System::Object^  sender, System::EventArgs^  e) {
    fcgMouseEnterLeave_SetColor(sender, themeMode, DarkenWindowState::Normal, dwStgReader);
}
System::Void frmBitrateCalculator::SetAllMouseMove(Control ^top, const AuoTheme themeTo) {
    if (themeTo == themeMode) return;
    System::Type^ type = top->GetType();
    if (type == CheckBox::typeid) {
        top->MouseEnter += gcnew System::EventHandler(this, &frmBitrateCalculator::fbcMouseEnter_SetColor);
        top->MouseLeave += gcnew System::EventHandler(this, &frmBitrateCalculator::fbcMouseLeave_SetColor);
    }
    for (int i = 0; i < top->Controls->Count; i++) {
        SetAllMouseMove(top->Controls[i], themeTo);
    }
}

/// -------------------------------------------------
///     frmConfig 関数  (frmBitrateCalculator関連)
/// -------------------------------------------------
System::Void frmConfig::CloseBitrateCalc() {
    frmBitrateCalculator::Instance::get()->Close();
}
System::Void frmConfig::fcgTSBBitrateCalc_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
    if (fcgTSBBitrateCalc->Checked) {
        bool videoBitrateMode = true;

        frmBitrateCalculator::Instance::get()->Init(
            (int)fcgNUBitrate->Value,
            (fcgNUAudioBitrate->Visible) ? (int)fcgNUAudioBitrate->Value : 0,
            videoBitrateMode,
            fcgNUAudioBitrate->Visible,
            (int)fcgNUAudioBitrate->Maximum,
            themeMode,
            dwStgReader
            );
        frmBitrateCalculator::Instance::get()->Owner = this;
        frmBitrateCalculator::Instance::get()->Show();
    } else {
        frmBitrateCalculator::Instance::get()->Close();
    }
}
System::Void frmConfig::SetfbcBTABEnable(bool enable, int max) {
    frmBitrateCalculator::Instance::get()->SetBTABEnabled(fcgNUAudioBitrate->Visible, max);
}
System::Void frmConfig::SetfbcBTVBEnable(bool enable) {
    frmBitrateCalculator::Instance::get()->SetBTVBEnabled(enable);
}

System::Void frmConfig::SetVideoBitrate(int bitrate) {
    SetNUValue(fcgNUBitrate, bitrate);
}

System::Void frmConfig::SetAudioBitrate(int bitrate) {
    SetNUValue(fcgNUAudioBitrate, bitrate);
}
System::Void frmConfig::InformfbcClosed() {
    fcgTSBBitrateCalc->Checked = false;
}


/// -------------------------------------------------
///     frmConfig 関数
/// -------------------------------------------------
/////////////   LocalStg関連  //////////////////////
System::Void frmConfig::LoadLocalStg() {
    guiEx_settings *_ex_stg = sys_dat->exstg;
    _ex_stg->load_encode_stg();
    LocalStg.CustomTmpDir    = String(_ex_stg->s_local.custom_tmp_dir).ToString();
    LocalStg.CustomAudTmpDir = String(_ex_stg->s_local.custom_audio_tmp_dir).ToString();
    LocalStg.CustomMP4TmpDir = String(_ex_stg->s_local.custom_mp4box_tmp_dir).ToString();
    LocalStg.LastAppDir      = String(_ex_stg->s_local.app_dir).ToString();
    LocalStg.LastBatDir      = String(_ex_stg->s_local.bat_dir).ToString();
    LocalStg.vidEncName      = String(_ex_stg->s_vid.filename).ToString();
    LocalStg.vidEncPath      = String(_ex_stg->s_vid.fullpath).ToString();
    LocalStg.MP4MuxerExeName = String(_ex_stg->s_mux[MUXER_MP4].filename).ToString();
    LocalStg.MP4MuxerPath    = String(_ex_stg->s_mux[MUXER_MP4].fullpath).ToString();
    LocalStg.MKVMuxerExeName = String(_ex_stg->s_mux[MUXER_MKV].filename).ToString();
    LocalStg.MKVMuxerPath    = String(_ex_stg->s_mux[MUXER_MKV].fullpath).ToString();
    LocalStg.TC2MP4ExeName   = String(_ex_stg->s_mux[MUXER_TC2MP4].filename).ToString();
    LocalStg.TC2MP4Path      = String(_ex_stg->s_mux[MUXER_TC2MP4].fullpath).ToString();
    LocalStg.MP4RawExeName   = String(_ex_stg->s_mux[MUXER_MP4_RAW].filename).ToString();
    LocalStg.MP4RawPath      = String(_ex_stg->s_mux[MUXER_MP4_RAW].fullpath).ToString();

    LocalStg.audEncName->Clear();
    LocalStg.audEncExeName->Clear();
    LocalStg.audEncPath->Clear();
    for (int i = 0; i < _ex_stg->s_aud_ext_count; i++) {
        LocalStg.audEncName->Add(String(_ex_stg->s_aud_ext[i].dispname).ToString());
        LocalStg.audEncExeName->Add(String(_ex_stg->s_aud_ext[i].filename).ToString());
        LocalStg.audEncPath->Add(String(_ex_stg->s_aud_ext[i].fullpath).ToString());
    }
    if (_ex_stg->s_local.large_cmdbox)
        fcgTXCmd_DoubleClick(nullptr, nullptr); //初期状態は縮小なので、拡大
}

System::Boolean frmConfig::CheckLocalStg() {
    bool error = false;
    String^ err = "";
    //映像エンコーダのチェック
    if (LocalStg.vidEncPath->Length > 0
        && !File::Exists(LocalStg.vidEncPath)) {
        if (!error) err += L"\n\n";
        error = true;
        err += LOAD_CLI_STRING(AUO_CONFIG_VID_ENC_NOT_EXIST) + L"\n [ " + LocalStg.vidEncPath + L" ]\n";
    }
    //音声エンコーダのチェック (実行ファイル名がない場合はチェックしない)
    if (fcgCBAudioUseExt->Checked
        && LocalStg.audEncExeName[fcgCXAudioEncoder->SelectedIndex]->Length) {
        String^ AudioEncoderPath = LocalStg.audEncPath[fcgCXAudioEncoder->SelectedIndex];
        if (AudioEncoderPath->Length > 0
            && !File::Exists(AudioEncoderPath)
            && (fcgCXAudioEncoder->SelectedIndex != sys_dat->exstg->get_faw_index(!fcgCBAudioUseExt->Checked)) ) {
            //音声実行ファイルがない かつ
            //選択された音声がfawでない
            if (!error) err += L"\n\n";
            error = true;
            err += LOAD_CLI_STRING(AUO_CONFIG_AUD_ENC_NOT_EXIST) + L"\n [ " + AudioEncoderPath + L" ]\n";
        }
    }
    //FAWのチェック
    if (fcgCBFAWCheck->Checked) {
        if (sys_dat->exstg->get_faw_index(!fcgCBAudioUseExt->Checked) == FAW_INDEX_ERROR) {
            if (!error) err += L"\n\n";
            error = true;
            err += LOAD_CLI_STRING(AUO_CONFIG_FAW_STG_NOT_FOUND_IN_INI1) + L"\n"
                +  LOAD_CLI_STRING(AUO_CONFIG_FAW_STG_NOT_FOUND_IN_INI2) + L"\n"
                +  LOAD_CLI_STRING(AUO_CONFIG_FAW_STG_NOT_FOUND_IN_INI3);
        }
    }
    if (error)
        MessageBox::Show(this, err, LOAD_CLI_STRING(AUO_GUIEX_ERROR), MessageBoxButtons::OK, MessageBoxIcon::Error);
    return error;
}

System::Void frmConfig::SaveLocalStg() {
    guiEx_settings *_ex_stg = sys_dat->exstg;
    _ex_stg->load_encode_stg();
    _ex_stg->s_local.large_cmdbox = fcgTXCmd->Multiline;
    GetCHARfromString(_ex_stg->s_local.custom_tmp_dir,        sizeof(_ex_stg->s_local.custom_tmp_dir),        LocalStg.CustomTmpDir);
    GetCHARfromString(_ex_stg->s_local.custom_mp4box_tmp_dir, sizeof(_ex_stg->s_local.custom_mp4box_tmp_dir), LocalStg.CustomMP4TmpDir);
    GetCHARfromString(_ex_stg->s_local.custom_audio_tmp_dir,  sizeof(_ex_stg->s_local.custom_audio_tmp_dir),  LocalStg.CustomAudTmpDir);
    GetCHARfromString(_ex_stg->s_local.app_dir,               sizeof(_ex_stg->s_local.app_dir),               LocalStg.LastAppDir);
    GetCHARfromString(_ex_stg->s_local.bat_dir,               sizeof(_ex_stg->s_local.bat_dir),               LocalStg.LastBatDir);
    GetCHARfromString(_ex_stg->s_vid.fullpath,                sizeof(_ex_stg->s_vid.fullpath),                LocalStg.vidEncPath);
    GetCHARfromString(_ex_stg->s_mux[MUXER_MP4].fullpath,     sizeof(_ex_stg->s_mux[MUXER_MP4].fullpath),     LocalStg.MP4MuxerPath);
    GetCHARfromString(_ex_stg->s_mux[MUXER_MKV].fullpath,     sizeof(_ex_stg->s_mux[MUXER_MKV].fullpath),     LocalStg.MKVMuxerPath);
    GetCHARfromString(_ex_stg->s_mux[MUXER_TC2MP4].fullpath,  sizeof(_ex_stg->s_mux[MUXER_TC2MP4].fullpath),  LocalStg.TC2MP4Path);
    GetCHARfromString(_ex_stg->s_mux[MUXER_MP4_RAW].fullpath, sizeof(_ex_stg->s_mux[MUXER_MP4_RAW].fullpath), LocalStg.MP4RawPath);
    for (int i = 0; i < _ex_stg->s_aud_ext_count; i++)
        GetCHARfromString(_ex_stg->s_aud_ext[i].fullpath, sizeof(_ex_stg->s_aud_ext[i].fullpath), LocalStg.audEncPath[i]);
    _ex_stg->save_local();
}

System::Void frmConfig::SetLocalStg() {
    fcgTXVideoEncoderPath->Text   = LocalStg.vidEncPath;
    fcgTXMP4MuxerPath->Text       = LocalStg.MP4MuxerPath;
    fcgTXMKVMuxerPath->Text       = LocalStg.MKVMuxerPath;
    fcgTXTC2MP4Path->Text         = LocalStg.TC2MP4Path;
    fcgTXMP4RawPath->Text         = LocalStg.MP4RawPath;
    fcgTXCustomAudioTempDir->Text = LocalStg.CustomAudTmpDir;
    fcgTXCustomTempDir->Text      = LocalStg.CustomTmpDir;
    fcgTXMP4BoxTempDir->Text      = LocalStg.CustomMP4TmpDir;
    fcgLBVideoEncoderPath->Text   = LocalStg.vidEncName      + LOAD_CLI_STRING(AUO_CONFIG_SPECIFY_EXE_PATH);
    fcgLBMP4MuxerPath->Text       = LocalStg.MP4MuxerExeName + LOAD_CLI_STRING(AUO_CONFIG_SPECIFY_EXE_PATH);
    fcgLBMKVMuxerPath->Text       = LocalStg.MKVMuxerExeName + LOAD_CLI_STRING(AUO_CONFIG_SPECIFY_EXE_PATH);
    fcgLBTC2MP4Path->Text         = LocalStg.TC2MP4ExeName   + LOAD_CLI_STRING(AUO_CONFIG_SPECIFY_EXE_PATH);
    fcgLBMP4RawPath->Text         = LocalStg.MP4RawExeName   + LOAD_CLI_STRING(AUO_CONFIG_SPECIFY_EXE_PATH);

    fcgTXVideoEncoderPath->SelectionStart = fcgTXVideoEncoderPath->Text->Length;
    fcgTXMP4MuxerPath->SelectionStart     = fcgTXMP4MuxerPath->Text->Length;
    fcgTXTC2MP4Path->SelectionStart       = fcgTXTC2MP4Path->Text->Length;
    fcgTXMKVMuxerPath->SelectionStart     = fcgTXMKVMuxerPath->Text->Length;
    fcgTXMP4RawPath->SelectionStart       = fcgTXMP4RawPath->Text->Length;
}

//////////////       その他イベント処理   ////////////////////////
System::Void frmConfig::ActivateToolTip(bool Enable) {
    fcgTTEx->Active = Enable;
}

System::Void frmConfig::fcgTSBOtherSettings_Click(System::Object^  sender, System::EventArgs^  e) {
    frmOtherSettings::Instance::get()->stgDir = String(sys_dat->exstg->s_local.stg_dir).ToString();
    frmOtherSettings::Instance::get()->SetTheme(themeMode, dwStgReader);
    frmOtherSettings::Instance::get()->ShowDialog();
    char buf[MAX_PATH_LEN];
    GetCHARfromString(buf, sizeof(buf), frmOtherSettings::Instance::get()->stgDir);
    if (_stricmp(buf, sys_dat->exstg->s_local.stg_dir)) {
        //変更があったら保存する
        strcpy_s(sys_dat->exstg->s_local.stg_dir, sizeof(sys_dat->exstg->s_local.stg_dir), buf);
        sys_dat->exstg->save_local();
        InitStgFileList();
    }
    //再読み込み
    guiEx_settings stg;
    stg.load_encode_stg();
    log_reload_settings();
    sys_dat->exstg->s_local.default_audenc_use_in = stg.s_local.default_audenc_use_in;
    sys_dat->exstg->s_local.default_audio_encoder_ext = stg.s_local.default_audio_encoder_ext;
    sys_dat->exstg->s_local.default_audio_encoder_in = stg.s_local.default_audio_encoder_in;
    sys_dat->exstg->s_local.get_relative_path = stg.s_local.get_relative_path;
    SetStgEscKey(stg.s_local.enable_stg_esc_key != 0);
    ActivateToolTip(stg.s_local.disable_tooltip_help == FALSE);
    if (str_has_char(stg.s_local.conf_font.name))
        SetFontFamilyToForm(this, gcnew FontFamily(String(stg.s_local.conf_font.name).ToString()), this->Font->FontFamily);
}
System::Boolean frmConfig::EnableSettingsNoteChange(bool Enable) {
    if (fcgTSTSettingsNotes->Visible == Enable &&
        fcgTSLSettingsNotes->Visible == !Enable)
        return true;
    if (CountStringBytes(fcgTSTSettingsNotes->Text) > fcgTSTSettingsNotes->MaxLength - 1) {
        MessageBox::Show(this, LOAD_CLI_STRING(AUO_CONFIG_TEXT_LIMIT_LENGTH), LOAD_CLI_STRING(AUO_GUIEX_ERROR), MessageBoxButtons::OK, MessageBoxIcon::Error);
        fcgTSTSettingsNotes->Focus();
        fcgTSTSettingsNotes->SelectionStart = fcgTSTSettingsNotes->Text->Length;
        return false;
    }
    fcgTSTSettingsNotes->Visible = Enable;
    fcgTSLSettingsNotes->Visible = !Enable;
    if (Enable) {
        fcgTSTSettingsNotes->Text = fcgTSLSettingsNotes->Text;
        fcgTSTSettingsNotes->Focus();
        bool isDefaultNote = fcgTSLSettingsNotes->Overflow != ToolStripItemOverflow::Never;
        fcgTSTSettingsNotes->Select((isDefaultNote) ? 0 : fcgTSTSettingsNotes->Text->Length, fcgTSTSettingsNotes->Text->Length);
    } else {
        SetfcgTSLSettingsNotes(fcgTSTSettingsNotes->Text);
        CheckOtherChanges(nullptr, nullptr);
    }
    return true;
}


///////////////////  メモ関連  ///////////////////////////////////////////////
System::Void frmConfig::fcgTSLSettingsNotes_DoubleClick(System::Object^  sender, System::EventArgs^  e) {
    EnableSettingsNoteChange(true);
}
System::Void frmConfig::fcgTSTSettingsNotes_Leave(System::Object^  sender, System::EventArgs^  e) {
    EnableSettingsNoteChange(false);
}
System::Void frmConfig::fcgTSTSettingsNotes_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
    if (e->KeyCode == Keys::Return)
        EnableSettingsNoteChange(false);
}
System::Void frmConfig::fcgTSTSettingsNotes_TextChanged(System::Object^  sender, System::EventArgs^  e) {
    SetfcgTSLSettingsNotes(fcgTSTSettingsNotes->Text);
    CheckOtherChanges(nullptr, nullptr);
}

/////////////    音声設定関連の関数    ///////////////
System::Void frmConfig::fcgCBAudio2pass_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
    if (fcgCBAudio2pass->Checked) {
        fcgCBAudioUsePipe->Checked = false;
        fcgCBAudioUsePipe->Enabled = false;
    } else if (CurrentPipeEnabled) {
        fcgCBAudioUsePipe->Checked = true;
        fcgCBAudioUsePipe->Enabled = true;
    }
}

System::Void frmConfig::fcgCXAudioEncoder_SelectedIndexChanged(System::Object ^sender, System::EventArgs ^e) {
    setAudioExtDisplay();
}

System::Void frmConfig::fcgCXAudioEncMode_SelectedIndexChanged(System::Object ^sender, System::EventArgs ^e) {
    AudioExtEncodeModeChanged();
}

System::Int32 frmConfig::GetCurrentAudioDefaultBitrate() {
    AUDIO_SETTINGS *astg = (fcgCBAudioUseExt->Checked) ? &sys_dat->exstg->s_aud_ext[std::max(fcgCXAudioEncoder->SelectedIndex, 0)] : &sys_dat->exstg->s_aud_int[std::max(fcgCXAudioEncoderInternal->SelectedIndex, 0)];
    const int encMode = std::max((fcgCBAudioUseExt->Checked) ? fcgCXAudioEncMode->SelectedIndex : fcgCXAudioEncModeInternal->SelectedIndex, 0);
    return astg->mode[encMode].bitrate_default;
}

System::Void frmConfig::setAudioExtDisplay() {
    AUDIO_SETTINGS *astg = &sys_dat->exstg->s_aud_ext[fcgCXAudioEncoder->SelectedIndex];
    //～の指定
    if (str_has_char(astg->filename)) {
        fcgLBAudioEncoderPath->Text = String(astg->filename).ToString() + LOAD_CLI_STRING(AUO_CONFIG_SPECIFY_EXE_PATH);
        fcgTXAudioEncoderPath->Enabled = true;
        fcgTXAudioEncoderPath->Text = LocalStg.audEncPath[fcgCXAudioEncoder->SelectedIndex];
        fcgBTAudioEncoderPath->Enabled = true;
    } else {
        //filename空文字列(wav出力時)
        fcgLBAudioEncoderPath->Text = L"";
        fcgTXAudioEncoderPath->Enabled = false;
        fcgTXAudioEncoderPath->Text = L"";
        fcgBTAudioEncoderPath->Enabled = false;
    }
    fcgTXAudioEncoderPath->SelectionStart = fcgTXAudioEncoderPath->Text->Length;
    fcgCXAudioEncMode->BeginUpdate();
    fcgCXAudioEncMode->Items->Clear();
    for (int i = 0; i < astg->mode_count; i++) {
        fcgCXAudioEncMode->Items->Add(String(astg->mode[i].name).ToString());
    }
    fcgCXAudioEncMode->EndUpdate();
    bool pipe_enabled = (astg->pipe_input && (!(fcgCBAudio2pass->Checked && astg->mode[fcgCXAudioEncMode->SelectedIndex].enc_2pass != 0)));
    CurrentPipeEnabled = pipe_enabled;
    fcgCBAudioUsePipe->Enabled = pipe_enabled;
    fcgCBAudioUsePipe->Checked = pipe_enabled;
    if (fcgCXAudioEncMode->Items->Count > 0)
        fcgCXAudioEncMode->SelectedIndex = 0;
}

System::Void frmConfig::AudioExtEncodeModeChanged() {
    int index = fcgCXAudioEncMode->SelectedIndex;
    AUDIO_SETTINGS *astg = &sys_dat->exstg->s_aud_ext[fcgCXAudioEncoder->SelectedIndex];
    if (astg->mode[index].bitrate) {
        fcgCXAudioEncMode->Width = fcgCXAudioEncModeSmallWidth;
        fcgLBAudioBitrate->Visible = true;
        fcgNUAudioBitrate->Visible = true;
        fcgNUAudioBitrate->Minimum = astg->mode[index].bitrate_min;
        fcgNUAudioBitrate->Maximum = astg->mode[index].bitrate_max;
        fcgNUAudioBitrate->Increment = astg->mode[index].bitrate_step;
        SetNUValue(fcgNUAudioBitrate, (conf->aud.ext.bitrate != 0) ? conf->aud.ext.bitrate : astg->mode[index].bitrate_default);
    } else {
        fcgCXAudioEncMode->Width = fcgCXAudioEncModeLargeWidth;
        fcgLBAudioBitrate->Visible = false;
        fcgNUAudioBitrate->Visible = false;
        fcgNUAudioBitrate->Minimum = 0;
        fcgNUAudioBitrate->Maximum = 65536;
    }
    fcgCBAudio2pass->Enabled = astg->mode[index].enc_2pass != 0;
    if (!fcgCBAudio2pass->Enabled) fcgCBAudio2pass->Checked = false;
    SetfbcBTABEnable(fcgNUAudioBitrate->Visible, (int)fcgNUAudioBitrate->Maximum);

    bool delay_cut_available = astg->mode[index].delay > 0;
    fcgLBAudioDelayCut->Visible = delay_cut_available;
    fcgCXAudioDelayCut->Visible = delay_cut_available;
    if (delay_cut_available) {
        const bool delay_cut_edts_available = str_has_char(astg->cmd_raw) && str_has_char(sys_dat->exstg->s_mux[MUXER_MP4_RAW].delay_cmd);
        const int current_idx = fcgCXAudioDelayCut->SelectedIndex;
        const int items_to_set = _countof(AUDIO_DELAY_CUT_MODE) - 1 - ((delay_cut_edts_available) ? 0 : 1);
        fcgCXAudioDelayCut->BeginUpdate();
        fcgCXAudioDelayCut->Items->Clear();
        for (int i = 0; i < items_to_set; i++) {
            String^ string = nullptr;
            if (AUDIO_DELAY_CUT_MODE[i].mes != AUO_MES_UNKNOWN) {
                string = LOAD_CLI_STRING(AUDIO_DELAY_CUT_MODE[i].mes);
            }
            if (string == nullptr || string->Length == 0) {
                string = String(AUDIO_DELAY_CUT_MODE[i].desc).ToString();
            }
            fcgCXAudioDelayCut->Items->Add(string);
        }
        fcgCXAudioDelayCut->EndUpdate();
        fcgCXAudioDelayCut->SelectedIndex = (current_idx >= items_to_set) ? 0 : current_idx;
    } else {
        fcgCXAudioDelayCut->SelectedIndex = 0;
    }
}

System::Void frmConfig::fcgCBAudioUseExt_CheckedChanged(System::Object ^sender, System::EventArgs ^e) {
    fcgPNAudioExt->Visible = fcgCBAudioUseExt->Checked;
    fcgPNAudioInternal->Visible = !fcgCBAudioUseExt->Checked;

    //fcgCBAudioUseExt->Checkedの場合は、処理順を「前」で固定する
    if (fcgCBAudioUseExt->Checked) {
        fcgCXAudioEncTiming->SelectedIndex = 1; // 処理順「前」
    }
    fcgCXAudioEncTiming->Enabled = !fcgCBAudioUseExt->Checked;

    //一度ウィンドウの再描画を完全に抑止する
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 0, 0);
    //なぜか知らんが、Visibleプロパティをfalseにするだけでは非表示にできない
    //しょうがないので参照の削除と挿入を行う
    fcgtabControlMux->TabPages->Clear();
    if (false && fcgCBAudioUseExt->Checked) {
        fcgtabControlMux->TabPages->Insert(0, fcgtabPageMP4);
        fcgtabControlMux->TabPages->Insert(1, fcgtabPageMKV);
        fcgtabControlMux->TabPages->Insert(2, fcgtabPageBat);
        fcgtabControlMux->TabPages->Insert(3, fcgtabPageMux);
    } else {
        fcgtabControlMux->TabPages->Insert(0, fcgtabPageInternal);
        fcgtabControlMux->TabPages->Insert(1, fcgtabPageBat);
        fcgtabControlMux->TabPages->Insert(2, fcgtabPageMux);
    }
    //一度ウィンドウの再描画を再開し、強制的に再描画させる
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 1, 0);
    this->Refresh();
}

System::Void frmConfig::fcgCXAudioEncoderInternal_SelectedIndexChanged(System::Object ^sender, System::EventArgs ^e) {
    setAudioIntDisplay();
}
System::Void frmConfig::fcgCXAudioEncModeInternal_SelectedIndexChanged(System::Object ^sender, System::EventArgs ^e) {
    AudioIntEncodeModeChanged();
}

System::Void frmConfig::setAudioIntDisplay() {
    AUDIO_SETTINGS *astg = &sys_dat->exstg->s_aud_int[fcgCXAudioEncoderInternal->SelectedIndex];
    fcgCXAudioEncModeInternal->BeginUpdate();
    fcgCXAudioEncModeInternal->Items->Clear();
    for (int i = 0; i < astg->mode_count; i++) {
        fcgCXAudioEncModeInternal->Items->Add(String(astg->mode[i].name).ToString());
    }
    fcgCXAudioEncModeInternal->EndUpdate();
    if (fcgCXAudioEncModeInternal->Items->Count > 0)
        fcgCXAudioEncModeInternal->SelectedIndex = 0;
}
System::Void frmConfig::AudioIntEncodeModeChanged() {
    const int imode = fcgCXAudioEncModeInternal->SelectedIndex;
    if (imode >= 0 && fcgCXAudioEncoderInternal->SelectedIndex >= 0) {
        AUDIO_SETTINGS *astg = &sys_dat->exstg->s_aud_int[fcgCXAudioEncoderInternal->SelectedIndex];
        if (astg->mode[imode].bitrate) {
            fcgCXAudioEncModeInternal->Width = fcgCXAudioEncModeSmallWidth;
            fcgLBAudioBitrateInternal->Visible = true;
            fcgNUAudioBitrateInternal->Visible = true;
            fcgNUAudioBitrateInternal->Minimum = astg->mode[imode].bitrate_min;
            fcgNUAudioBitrateInternal->Maximum = astg->mode[imode].bitrate_max;
            fcgNUAudioBitrateInternal->Increment = astg->mode[imode].bitrate_step;
            SetNUValue(fcgNUAudioBitrateInternal, (conf->aud.in.bitrate > 0) ? conf->aud.in.bitrate : astg->mode[imode].bitrate_default);
        } else {
            fcgCXAudioEncModeInternal->Width = fcgCXAudioEncModeLargeWidth;
            fcgLBAudioBitrateInternal->Visible = false;
            fcgNUAudioBitrateInternal->Visible = false;
            fcgNUAudioBitrateInternal->Minimum = 0;
            fcgNUAudioBitrateInternal->Maximum = 65536;
        }
    }
    SetfbcBTABEnable(fcgNUAudioBitrateInternal->Visible, (int)fcgNUAudioBitrateInternal->Maximum);
}

///////////////   設定ファイル関連   //////////////////////
System::Void frmConfig::CheckTSItemsEnabled(CONF_GUIEX *current_conf) {
    bool selected = (CheckedStgMenuItem != nullptr);
    fcgTSBSave->Enabled = (selected && memcmp(cnf_stgSelected, current_conf, sizeof(CONF_GUIEX)));
    fcgTSBDelete->Enabled = selected;
}

System::Void frmConfig::UncheckAllDropDownItem(ToolStripItem^ mItem) {
    ToolStripDropDownItem^ DropDownItem = dynamic_cast<ToolStripDropDownItem^>(mItem);
    if (DropDownItem == nullptr)
        return;
    for (int i = 0; i < DropDownItem->DropDownItems->Count; i++) {
        UncheckAllDropDownItem(DropDownItem->DropDownItems[i]);
        ToolStripMenuItem^ item = dynamic_cast<ToolStripMenuItem^>(DropDownItem->DropDownItems[i]);
        if (item != nullptr)
            item->Checked = false;
    }
}

System::Void frmConfig::CheckTSSettingsDropDownItem(ToolStripMenuItem^ mItem) {
    UncheckAllDropDownItem(fcgTSSettings);
    CheckedStgMenuItem = mItem;
    fcgTSSettings->Text = (mItem == nullptr) ? LOAD_CLI_STRING(AUO_CONFIG_PROFILE) : mItem->Text;
    if (mItem != nullptr)
        mItem->Checked = true;
    fcgTSBSave->Enabled = false;
    fcgTSBDelete->Enabled = (mItem != nullptr);
}

ToolStripMenuItem^ frmConfig::fcgTSSettingsSearchItem(String^ stgPath, ToolStripItem^ mItem) {
    if (stgPath == nullptr)
        return nullptr;
    ToolStripDropDownItem^ DropDownItem = dynamic_cast<ToolStripDropDownItem^>(mItem);
    if (DropDownItem == nullptr)
        return nullptr;
    for (int i = 0; i < DropDownItem->DropDownItems->Count; i++) {
        ToolStripMenuItem^ item = fcgTSSettingsSearchItem(stgPath, DropDownItem->DropDownItems[i]);
        if (item != nullptr)
            return item;
        item = dynamic_cast<ToolStripMenuItem^>(DropDownItem->DropDownItems[i]);
        if (item      != nullptr &&
            item->Tag != nullptr &&
            0 == String::Compare(item->Tag->ToString(), stgPath, true))
            return item;
    }
    return nullptr;
}

ToolStripMenuItem^ frmConfig::fcgTSSettingsSearchItem(String^ stgPath) {
    return fcgTSSettingsSearchItem((stgPath != nullptr && stgPath->Length > 0) ? Path::GetFullPath(stgPath) : nullptr, fcgTSSettings);
}

System::Void frmConfig::SaveToStgFile(String^ stgName) {
    size_t nameLen = CountStringBytes(stgName) + 1;
    char *stg_name = (char *)malloc(nameLen);
    GetCHARfromString(stg_name, nameLen, stgName);
    init_CONF_GUIEX(cnf_stgSelected, FALSE);
    FrmToConf(cnf_stgSelected);
    String^ stgDir = Path::GetDirectoryName(stgName);
    if (!Directory::Exists(stgDir))
        Directory::CreateDirectory(stgDir);
    int result = guiEx_config::save_guiEx_conf(cnf_stgSelected, stg_name);
    free(stg_name);
    switch (result) {
        case CONF_ERROR_FILE_OPEN:
            MessageBox::Show(LOAD_CLI_STRING(AUO_CONFIG_ERR_OPEN_STG_FILE), LOAD_CLI_STRING(AUO_GUIEX_ERROR), MessageBoxButtons::OK, MessageBoxIcon::Error);
            return;
        case CONF_ERROR_INVALID_FILENAME:
            MessageBox::Show(LOAD_CLI_STRING(AUO_CONFIG_ERR_INVALID_CHAR), LOAD_CLI_STRING(AUO_GUIEX_ERROR), MessageBoxButtons::OK, MessageBoxIcon::Error);
            return;
        default:
            break;
    }
    init_CONF_GUIEX(cnf_stgSelected, FALSE);
    FrmToConf(cnf_stgSelected);
}

System::Void frmConfig::fcgTSBSave_Click(System::Object^  sender, System::EventArgs^  e) {
    if (CheckedStgMenuItem != nullptr)
        SaveToStgFile(CheckedStgMenuItem->Tag->ToString());
    CheckTSSettingsDropDownItem(CheckedStgMenuItem);
}

System::Void frmConfig::fcgTSBSaveNew_Click(System::Object^  sender, System::EventArgs^  e) {
    frmSaveNewStg::Instance::get()->setStgDir(String(sys_dat->exstg->s_local.stg_dir).ToString());
    frmSaveNewStg::Instance::get()->SetTheme(themeMode, dwStgReader);
    if (CheckedStgMenuItem != nullptr)
        frmSaveNewStg::Instance::get()->setFilename(CheckedStgMenuItem->Text);
    frmSaveNewStg::Instance::get()->ShowDialog();
    String^ stgName = frmSaveNewStg::Instance::get()->StgFileName;
    if (stgName != nullptr && stgName->Length)
        SaveToStgFile(stgName);
    RebuildStgFileDropDown(nullptr);
    CheckTSSettingsDropDownItem(fcgTSSettingsSearchItem(stgName));
}

System::Void frmConfig::DeleteStgFile(ToolStripMenuItem^ mItem) {
    if (System::Windows::Forms::DialogResult::OK ==
        MessageBox::Show(LOAD_CLI_STRING(AUO_CONFIG_ASK_STG_FILE_DELETE) + L"[" + mItem->Text + L"]",
        LOAD_CLI_STRING(AUO_GUIEX_ERROR), MessageBoxButtons::OKCancel, MessageBoxIcon::Exclamation))
    {
        File::Delete(mItem->Tag->ToString());
        RebuildStgFileDropDown(nullptr);
        CheckTSSettingsDropDownItem(nullptr);
        SetfcgTSLSettingsNotes(L"");
    }
}

System::Void frmConfig::fcgTSBDelete_Click(System::Object^  sender, System::EventArgs^  e) {
    DeleteStgFile(CheckedStgMenuItem);
}

System::Void frmConfig::fcgTSSettings_DropDownItemClicked(System::Object^  sender, System::Windows::Forms::ToolStripItemClickedEventArgs^  e) {
    ToolStripMenuItem^ ClickedMenuItem = dynamic_cast<ToolStripMenuItem^>(e->ClickedItem);
    if (ClickedMenuItem == nullptr)
        return;
    if (ClickedMenuItem->Tag == nullptr || ClickedMenuItem->Tag->ToString()->Length == 0)
        return;
    CONF_GUIEX load_stg;
    char stg_path[MAX_PATH_LEN];
    GetCHARfromString(stg_path, sizeof(stg_path), ClickedMenuItem->Tag->ToString());
    if (guiEx_config::load_guiEx_conf(&load_stg, stg_path) == CONF_ERROR_FILE_OPEN) {
        if (MessageBox::Show(LOAD_CLI_STRING(AUO_CONFIG_ERR_OPEN_STG_FILE) + L"\n"
                           + LOAD_CLI_STRING(AUO_CONFIG_ASK_STG_FILE_DELETE),
                           LOAD_CLI_STRING(AUO_GUIEX_ERROR), MessageBoxButtons::YesNo, MessageBoxIcon::Error)
                           == System::Windows::Forms::DialogResult::Yes)
            DeleteStgFile(ClickedMenuItem);
        return;
    }
    ConfToFrm(&load_stg);
    CheckTSSettingsDropDownItem(ClickedMenuItem);
    memcpy(cnf_stgSelected, &load_stg, sizeof(CONF_GUIEX));
}

System::Void frmConfig::RebuildStgFileDropDown(ToolStripDropDownItem^ TS, String^ dir) {
    array<String^>^ subDirs = Directory::GetDirectories(dir);
    for (int i = 0; i < subDirs->Length; i++) {
        ToolStripMenuItem^ DDItem = gcnew ToolStripMenuItem(L"[ " + subDirs[i]->Substring(dir->Length+1) + L" ]");
        DDItem->DropDownItemClicked += gcnew System::Windows::Forms::ToolStripItemClickedEventHandler(this, &frmConfig::fcgTSSettings_DropDownItemClicked);
        DDItem->ForeColor = Color::Blue;
        DDItem->Tag = nullptr;
        RebuildStgFileDropDown(DDItem, subDirs[i]);
        TS->DropDownItems->Add(DDItem);
    }
    array<String^>^ stgList = Directory::GetFiles(dir, L"*.stg");
    for (int i = 0; i < stgList->Length; i++) {
        ToolStripMenuItem^ mItem = gcnew ToolStripMenuItem(Path::GetFileNameWithoutExtension(stgList[i]));
        mItem->Tag = stgList[i];
        TS->DropDownItems->Add(mItem);
    }
}

System::Void frmConfig::RebuildStgFileDropDown(String^ stgDir) {
    fcgTSSettings->DropDownItems->Clear();
    if (stgDir != nullptr)
        CurrentStgDir = stgDir;
    if (!Directory::Exists(CurrentStgDir))
        Directory::CreateDirectory(CurrentStgDir);
    RebuildStgFileDropDown(fcgTSSettings, Path::GetFullPath(CurrentStgDir));
}

///////////////   言語ファイル関連   //////////////////////

System::Void frmConfig::CheckTSLanguageDropDownItem(ToolStripMenuItem^ mItem) {
    UncheckAllDropDownItem(fcgTSLanguage);
    fcgTSLanguage->Text = (mItem == nullptr) ? LOAD_CLI_STRING(AuofcgTSSettings) : mItem->Text;
    if (mItem != nullptr)
        mItem->Checked = true;
}
System::Void frmConfig::SetSelectedLanguage(const char *language_text) {
    for (int i = 0; i < fcgTSLanguage->DropDownItems->Count; i++) {
        ToolStripMenuItem^ item = dynamic_cast<ToolStripMenuItem^>(fcgTSLanguage->DropDownItems[i]);
        char item_text[MAX_PATH_LEN];
        GetCHARfromString(item_text, sizeof(item_text), item->Tag->ToString());
        if (strncmp(item_text, language_text, strlen(language_text)) == 0) {
            CheckTSLanguageDropDownItem(item);
            break;
        }
    }
}

System::Void frmConfig::SaveSelectedLanguage(const char *language_text) {
    sys_dat->exstg->set_and_save_lang(language_text);
}

System::Void frmConfig::fcgTSLanguage_DropDownItemClicked(System::Object^  sender, System::Windows::Forms::ToolStripItemClickedEventArgs^  e) {
    ToolStripMenuItem^ ClickedMenuItem = dynamic_cast<ToolStripMenuItem^>(e->ClickedItem);
    if (ClickedMenuItem == nullptr)
        return;
    if (ClickedMenuItem->Tag == nullptr || ClickedMenuItem->Tag->ToString()->Length == 0)
        return;

    char language_text[MAX_PATH_LEN];
    GetCHARfromString(language_text, sizeof(language_text), ClickedMenuItem->Tag->ToString());
    SaveSelectedLanguage(language_text);
    load_lng(language_text);
    overwrite_aviutl_ini_auo_info();
    LoadLangText();
    CheckTSLanguageDropDownItem(ClickedMenuItem);
}

System::Void frmConfig::InitLangList() {
    if (list_lng != nullptr) {
        delete list_lng;
    }
#define ENABLE_LNG_FILE_DETECT 1
#if ENABLE_LNG_FILE_DETECT
    auto lnglist = find_lng_files();
    list_lng = new std::vector<std::string>();
    for (const auto& lang : lnglist) {
        list_lng->push_back(lang);
    }
#endif

    fcgTSLanguage->DropDownItems->Clear();

    for (const auto& auo_lang : list_auo_languages) {
        String^ label = String(auo_lang.code).ToString() + L" (" + String(auo_lang.name).ToString() + L")";
        ToolStripMenuItem^ mItem = gcnew ToolStripMenuItem(label);
        mItem->DropDownItemClicked += gcnew System::Windows::Forms::ToolStripItemClickedEventHandler(this, &frmConfig::fcgTSLanguage_DropDownItemClicked);
        mItem->Tag = String(auo_lang.code).ToString();
        fcgTSLanguage->DropDownItems->Add(mItem);
    }
#if ENABLE_LNG_FILE_DETECT
    for (size_t i = 0; i < list_lng->size(); i++) {
        auto filename = String(PathFindFileNameA((*list_lng)[i].c_str())).ToString();
        ToolStripMenuItem^ mItem = gcnew ToolStripMenuItem(filename);
        mItem->DropDownItemClicked += gcnew System::Windows::Forms::ToolStripItemClickedEventHandler(this, &frmConfig::fcgTSLanguage_DropDownItemClicked);
        mItem->Tag = filename;
        fcgTSLanguage->DropDownItems->Add(mItem);
    }
#endif
    SetSelectedLanguage(sys_dat->exstg->get_lang());
}

//////////////   初期化関連     ////////////////
System::Void frmConfig::InitData(CONF_GUIEX *set_config, const SYSTEM_DATA *system_data) {
    if (set_config->size_all != CONF_INITIALIZED) {
        //初期化されていなければ初期化する
        init_CONF_GUIEX(set_config, FALSE);
    }
    conf = set_config;
    sys_dat = system_data;
}

System::Void frmConfig::InitComboBox() {
    //コンボボックスに値を設定する
    setComboBox(fcgCXEncMode,         list_encmode);
    setComboBox(fcgCXEncCodec,        list_out_enc_codec);
    setComboBox(fcgCXHyperMode,       list_hyper_mode);
    setComboBox(fcgCXFunctionMode,    list_qsv_function_mode);
    setComboBox(fcgCXCodecLevel,      list_avc_level);
    setComboBox(fcgCXCodecProfile,    list_avc_profile);
    setComboBox(fcgCXOutputCsp,       list_output_csp, (int)_countof(list_output_csp) - 4);
    setComboBox(fcgCXBitDepth,        bit_depth_desc);
    setComboBox(fcgCXQualityPreset,   list_quality);
    setComboBox(fcgCXInterlaced,      list_interlaced_mfx_gui);
    setComboBox(fcgCXAspectRatio,     aspect_desc);
    setComboBox(fcgCXTrellis,         list_avc_trellis);
    setComboBox(fcgCXLookaheadDS,     list_lookahead_ds);
    
    setComboBox(fcgCXScenarioInfo,    list_scenario_info);
    setComboBox(fcgCXMVPred,          list_mv_presicion);
    setComboBox(fcgCXInterPred,       list_pred_block_size);
    setComboBox(fcgCXIntraPred,       list_pred_block_size);

    setComboBox(fcgCXMVCostScaling,   list_mv_cost_scaling);

    setComboBox(fcgCXAudioTempDir,    audtempdir_desc);
    setComboBox(fcgCXMP4BoxTempDir,   mp4boxtempdir_desc);
    setComboBox(fcgCXTempDir,    tempdir_desc);

    setComboBox(fcgCXColorPrim,       list_colorprim, "auto");
    setComboBox(fcgCXColorMatrix,     list_colormatrix, "auto");
    setComboBox(fcgCXTransfer,        list_transfer, "auto");
    setComboBox(fcgCXVideoFormat,     list_videoformat, "auto");

    setComboBox(fcgCXVppDenoiseMethod, list_vpp_denoise);
    setComboBox(fcgCXVppDenoiseDctStep, list_vpp_denoise_dct_step_gui);
    setComboBox(fcgCXVppDenoiseDctBlockSize, list_vpp_denoise_dct_block_size);
    setComboBox(fcgCXVppDenoiseConv3DMatrix, list_vpp_convolution3d_matrix);
    setComboBox(fcgCXVppDenoiseNLMeansPatch,  list_vpp_nlmeans_block_size);
    setComboBox(fcgCXVppDenoiseNLMeansSearch, list_vpp_nlmeans_block_size);
    setComboBox(fcgCXVppDenoiseFFT3DBlockSize, list_vpp_fft3d_block_size);
    setComboBox(fcgCXVppDenoiseFFT3DTemporal, list_vpp_fft3d_temporal_gui);
    setComboBox(fcgCXVppDenoiseFFT3DPrecision, list_vpp_fp_prec);
    setComboBox(fcgCXVppDetailEnhance, list_vpp_detail_enahance);

    setComboBox(fcgCXVppResizeAlg,   list_vpp_resize);
    setComboBox(fcgCXVppDeinterlace, list_deinterlace_gui);
    setComboBox(fcgCXVppAfsAnalyze,  list_vpp_afs_analyze);
    setComboBox(fcgCXVppNnediNsize,  list_vpp_nnedi_nsize);
    setComboBox(fcgCXVppNnediNns,    list_vpp_nnedi_nns);
    setComboBox(fcgCXVppNnediQual,   list_vpp_nnedi_quality);
    setComboBox(fcgCXVppNnediPrec,   list_vpp_fp_prec);
    setComboBox(fcgCXVppNnediPrescreen, list_vpp_nnedi_pre_screen_gui);
    setComboBox(fcgCXVppNnediErrorType, list_vpp_nnedi_error_type);
    setComboBox(fcgCXVppYadifMode,      list_vpp_yadif_mode_gui);
    setComboBox(fcgCXVppDebandSample,   list_vpp_deband_gui);
    setComboBox(fcgCXVppDeband,         list_vpp_deband_names);
    setComboBox(fcgCXVppLibplaceboDebandDither, list_vpp_libplacebo_deband_dither_mode);
    setComboBox(fcgCXVppLibplaceboDebandLUTSize, list_vpp_libplacebo_deband_lut_size);

    setComboBox(fcgCXImageStabilizer, list_vpp_image_stabilizer);
    setComboBox(fcgCXRotate,          list_rotate_angle_ja);

    setComboBox(fcgCXAudioEncTiming, audio_enc_timing_desc);
    setComboBox(fcgCXAudioDelayCut,  AUDIO_DELAY_CUT_MODE);

    setMuxerCmdExNames(fcgCXMP4CmdEx, MUXER_MP4);
    setMuxerCmdExNames(fcgCXMKVCmdEx, MUXER_MKV);
    setMuxerCmdExNames(fcgCXInternalCmdEx, MUXER_INTERNAL);

    setAudioEncoderNames();

    setPriorityList(fcgCXMuxPriority);
    setPriorityList(fcgCXAudioPriority);
}

System::Void frmConfig::SetTXMaxLen(TextBox^ TX, int max_len) {
    TX->MaxLength = max_len;
    TX->Validating += gcnew System::ComponentModel::CancelEventHandler(this, &frmConfig::TX_LimitbyBytes);
}

System::Void frmConfig::SetTXMaxLenAll() {
    //MaxLengthに最大文字数をセットし、それをもとにバイト数計算を行うイベントをセットする。
    SetTXMaxLen(fcgTXVideoEncoderPath,   sizeof(sys_dat->exstg->s_vid.fullpath) - 1);
    SetTXMaxLen(fcgTXAudioEncoderPath,   sizeof(sys_dat->exstg->s_aud_ext[0].fullpath) - 1);
    SetTXMaxLen(fcgTXMP4MuxerPath,       sizeof(sys_dat->exstg->s_mux[MUXER_MP4].fullpath) - 1);
    SetTXMaxLen(fcgTXMKVMuxerPath,       sizeof(sys_dat->exstg->s_mux[MUXER_MKV].fullpath) - 1);
    SetTXMaxLen(fcgTXTC2MP4Path,         sizeof(sys_dat->exstg->s_mux[MUXER_TC2MP4].fullpath) - 1);
    SetTXMaxLen(fcgTXMP4RawPath,         sizeof(sys_dat->exstg->s_mux[MUXER_MP4_RAW].fullpath) - 1);
    SetTXMaxLen(fcgTXCustomTempDir,      sizeof(sys_dat->exstg->s_local.custom_tmp_dir) - 1);
    SetTXMaxLen(fcgTXCustomAudioTempDir, sizeof(sys_dat->exstg->s_local.custom_audio_tmp_dir) - 1);
    SetTXMaxLen(fcgTXMP4BoxTempDir,      sizeof(sys_dat->exstg->s_local.custom_mp4box_tmp_dir) - 1);
    SetTXMaxLen(fcgTXBatBeforeAudioPath, sizeof(conf->oth.batfile.before_audio) - 1);
    SetTXMaxLen(fcgTXBatAfterAudioPath,  sizeof(conf->oth.batfile.after_audio) - 1);
    SetTXMaxLen(fcgTXBatBeforePath,      sizeof(conf->oth.batfile.before_process) - 1);
    SetTXMaxLen(fcgTXBatAfterPath,       sizeof(conf->oth.batfile.after_process) - 1);
    fcgTSTSettingsNotes->MaxLength     = sizeof(conf->oth.notes) - 1;
}

System::Void frmConfig::InitStgFileList() {
    RebuildStgFileDropDown(String(sys_dat->exstg->s_local.stg_dir).ToString());
    stgChanged = false;
    CheckTSSettingsDropDownItem(nullptr);
}

System::Boolean frmConfig::fcgCheckCodec() {
    System::Boolean result = false;
    if (featuresHW == nullptr || fcgCXDevice->SelectedIndex < 0) {
        return result;
    }

    for (int codecIdx = 1; list_out_enc_codec[codecIdx].desc; codecIdx++) {
        const auto codec = get_out_enc_codec_by_index(fcgCXEncCodec->SelectedIndex);
        const bool codecAvail = featuresHW->getCodecAvail(fcgCXDevice->SelectedIndex, codec);
        if (!codecAvail) {
            fcgCXEncCodec->Items[codecIdx] = L"-----------------";
            if (fcgCXEncCodec->SelectedIndex == codecIdx) {
                fcgCXEncCodec->SelectedIndex = 0;
                result = true;
            }
        } else {
            fcgCXEncCodec->Items[codecIdx] = String(list_out_enc_codec[codecIdx].desc).ToString();
        }
    }
    const auto codec = get_out_enc_codec_by_index(fcgCXEncCodec->SelectedIndex);
    // codec が RGY_CODEC_AV1 の時、CQPの上限は255、それ以外なら51
    const int qp_max = (codec == RGY_CODEC_AV1) ? 255 : 51;
    fcgNUQPI->Maximum = qp_max;
    fcgNUQPP->Maximum = qp_max;
    fcgNUQPB->Maximum = qp_max;
    fcgNUQPMin->Maximum = qp_max;
    fcgNUQPMax->Maximum = qp_max;
    return result;
}

System::Void frmConfig::fcgCheckFixedFunc() {
    if (featuresHW == nullptr || fcgCXDevice->SelectedIndex < 0) {
        return;
    }
    const auto codec = get_out_enc_codec_by_index(fcgCXEncCodec->SelectedIndex);
    if (!featuresHW->getCodecAvail(fcgCXDevice->SelectedIndex, codec)) {
        return;
    }

    const bool codecFFAvail = featuresHW->getCodecAvail(fcgCXDevice->SelectedIndex, codec, true);
    const bool codecPGAvail = featuresHW->getCodecAvail(fcgCXDevice->SelectedIndex, codec, false);
    if (codecFFAvail && codecPGAvail) {
        if (!fcgCXFunctionMode->Enabled) {
            fcgCXFunctionMode->SelectedIndex = get_cx_index(list_qsv_function_mode, _T("Auto"));
        }
        fcgCXFunctionMode->Enabled = true;
    } else if (codecFFAvail) {
        fcgCXFunctionMode->Enabled = false;
        fcgCXFunctionMode->SelectedIndex = get_cx_index(list_qsv_function_mode, _T("FF"));
    } else if (codecPGAvail) {
        fcgCXFunctionMode->Enabled = false;
        fcgCXFunctionMode->SelectedIndex = get_cx_index(list_qsv_function_mode, _T("PG"));
    }
}

System::Boolean frmConfig::fcgCheckRCModeLibVersion(int rc_mode_target, int rc_mode_replace, bool mode_supported) {
    System::Boolean selected_idx_changed = false;
    int encmode_idx = get_cx_index(list_encmode, rc_mode_target);
    if (encmode_idx < 0)
        return false;
    if (mode_supported) {
        String^ encomodeName = nullptr;
        if (list_encmode[encmode_idx].mes != AUO_MES_UNKNOWN) {
            encomodeName = LOAD_CLI_STRING(list_encmode[encmode_idx].mes);
        }
        if (encomodeName == nullptr || encomodeName->Length == 0) {
            encomodeName = String(list_encmode[encmode_idx].desc).ToString();
        }
        fcgCXEncMode->Items[encmode_idx] = encomodeName;
    } else {
        fcgCXEncMode->Items[encmode_idx] = L"-----------------";
        if (fcgCXEncMode->SelectedIndex == encmode_idx) {
            fcgCXEncMode->SelectedIndex = get_cx_index(list_encmode, rc_mode_replace);
            selected_idx_changed = true;
        }
    }
    return selected_idx_changed;
}

System::Boolean frmConfig::fcgCheckLibRateControl() {
    const auto codec = get_out_enc_codec_by_index(fcgCXEncCodec->SelectedIndex);

    const QSVFunctionMode funcMode = (QSVFunctionMode)list_qsv_function_mode[fcgCXFunctionMode->SelectedIndex].value;

    System::Boolean result = false;
    if (fcgCheckRCModeLibVersion(MFX_RATECONTROL_AVBR,   MFX_RATECONTROL_VBR, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_AVBR),   codec, funcMode))) result = true;
    if (fcgCheckRCModeLibVersion(MFX_RATECONTROL_QVBR,   MFX_RATECONTROL_VBR, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_QVBR),   codec, funcMode))) result = true;
    if (fcgCheckRCModeLibVersion(MFX_RATECONTROL_LA,     MFX_RATECONTROL_VBR, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_LA),     codec, funcMode))) result = true;
        //if (fcgCheckRCModeLibVersion(MFX_RATECONTROL_LA_EXT, MFX_RATECONTROL_VBR, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_LA_EXT), codec, funcMode))) result = true;
    if (fcgCheckRCModeLibVersion(MFX_RATECONTROL_LA_HRD, MFX_RATECONTROL_VBR, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_LA_HRD), codec, funcMode))) result = true;
    if (fcgCheckRCModeLibVersion(MFX_RATECONTROL_ICQ,    MFX_RATECONTROL_CQP, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_ICQ),    codec, funcMode))) result = true;
    if (fcgCheckRCModeLibVersion(MFX_RATECONTROL_LA_ICQ, MFX_RATECONTROL_CQP, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_LA_ICQ), codec, funcMode))) result = true;
    if (fcgCheckRCModeLibVersion(MFX_RATECONTROL_VCM,    MFX_RATECONTROL_VBR, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_VCM),    codec, funcMode))) result = true;
    return result;
}

System::Void frmConfig::fcgCheckBFrameAndGopRefDsit() {
    const RGY_CODEC codec = get_out_enc_codec_by_index(fcgCXEncCodec->SelectedIndex);
    const QSVFunctionMode funcMode = (QSVFunctionMode)list_qsv_function_mode[fcgCXFunctionMode->SelectedIndex].value;
    const auto available_features = featuresHW->getFeatureOfRC(fcgCXDevice->SelectedIndex, fcgCXEncMode->SelectedIndex, codec, funcMode);
    const int currentNUBframes = (int)fcgNUBframes->Value;
    if (gopRefDistAsBframe(codec)) {
        if (fcgLBBframes->Text == L"GopRefDist") {
            LOAD_CLI_TEXT(fcgLBBframes);
            LOAD_CLI_TEXT(fcgLBBframesAuto);
            fcgNUBframes->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { -1, 0, 0, System::Int32::MinValue });;
            fcgNUBframes->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 16, 0, 0, 0 });
            SetNUValue(fcgNUBframes, currentNUBframes - 1);
        }
    } else {
        if (fcgLBBframes->Text != L"GopRefDist") {
            fcgLBBframes->Text = L"GopRefDist";
            String^ autoStr = String(g_auo_mes.get(AuofcgLBRefAuto)).ToString();
            if (autoStr->Length > 0) {
                fcgLBBframesAuto->Text = autoStr;
            }
            fcgNUBframes->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 0, 0, 0, 0 });
            fcgNUBframes->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 17, 0, 0, 0 });
            SetNUValue(fcgNUBframes, currentNUBframes + 1);
        }
    }
    if (available_features & ENC_FEATURE_GOPREFDIST) {
        if (!fcgNUBframes->Enabled) {
            fcgNUBframes->Enabled = true;
            SetNUValue(fcgNUBframes, -1);
        }
    } else {
        fcgNUBframes->Enabled = false;
        SetNUValue(fcgNUBframes, 0);
    }
}

System::Void frmConfig::fcgCheckLibVersion() {

    fcgCheckBFrameAndGopRefDsit();

    const auto codec = get_out_enc_codec_by_index(fcgCXEncCodec->SelectedIndex);
    const QSVFunctionMode funcMode = (QSVFunctionMode)list_qsv_function_mode[fcgCXFunctionMode->SelectedIndex].value;

    //API v1.3 features
    fcgCheckRCModeLibVersion(MFX_RATECONTROL_AVBR, MFX_RATECONTROL_VBR, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_AVBR), codec, funcMode));
    auto available_features = featuresHW->getFeatureOfRC(fcgCXDevice->SelectedIndex, fcgCXEncMode->SelectedIndex, codec, funcMode); // fcgCXEncMode->SelectedIndex が変わっている可能性があるので再取得
    fcgLBVideoFormat->Enabled = 0 != (available_features & ENC_FEATURE_VUI_INFO);
    fcgCXVideoFormat->Enabled = 0 != (available_features & ENC_FEATURE_VUI_INFO);
    fcgLBFullrange->Enabled   = 0 != (available_features & ENC_FEATURE_VUI_INFO);
    fcgCBFullrange->Enabled   = 0 != (available_features & ENC_FEATURE_VUI_INFO);
    fcggroupBoxColorMatrix->Enabled = 0 != (available_features & ENC_FEATURE_VUI_INFO);
    if (!fcgCXVideoFormat->Enabled) fcgCXVideoFormat->SelectedIndex = 0;
    if (!fcgCBFullrange->Enabled)   fcgCBFullrange->Checked = false;
    if (!fcggroupBoxColorMatrix->Enabled) {
        fcgCXColorMatrix->SelectedIndex = 0;
        fcgCXColorPrim->SelectedIndex = 0;
        fcgCXTransfer->SelectedIndex = 0;
    }

    //API v1.6 features
    fcgCBExtBRC->Enabled = 0 != (available_features & ENC_FEATURE_EXT_BRC);
    fcgCBMBBRC->Enabled  = 0 != (available_features & ENC_FEATURE_MBBRC);
    if (!fcgCBExtBRC->Enabled) fcgCBExtBRC->Checked = false;
    if (!fcgCBMBBRC->Enabled)  fcgCBMBBRC->Checked = false;

    //API v1.7 features
    fcgCheckRCModeLibVersion(MFX_RATECONTROL_LA, MFX_RATECONTROL_VBR, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_LA), codec, funcMode));
    available_features = featuresHW->getFeatureOfRC(fcgCXDevice->SelectedIndex, fcgCXEncMode->SelectedIndex, codec, funcMode); // fcgCXEncMode->SelectedIndex が変わっている可能性があるので再取得
    fcgLBTrellis->Enabled = 0 != (available_features & ENC_FEATURE_TRELLIS);
    fcgCXTrellis->Enabled = 0 != (available_features & ENC_FEATURE_TRELLIS);
    if (!fcgCXTrellis->Enabled) fcgCXTrellis->SelectedIndex = 0;

    //API v1.8 features
    fcgCheckRCModeLibVersion(MFX_RATECONTROL_ICQ,    MFX_RATECONTROL_CQP, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_ICQ),    codec, funcMode));
    fcgCheckRCModeLibVersion(MFX_RATECONTROL_LA_ICQ, MFX_RATECONTROL_CQP, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_LA_ICQ), codec, funcMode));
    fcgCheckRCModeLibVersion(MFX_RATECONTROL_VCM,    MFX_RATECONTROL_VBR, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_VCM),    codec, funcMode));
    available_features = featuresHW->getFeatureOfRC(fcgCXDevice->SelectedIndex, fcgCXEncMode->SelectedIndex, codec, funcMode); // fcgCXEncMode->SelectedIndex が変わっている可能性があるので再取得
    fcgCBAdaptiveB->Enabled   = 0 != (available_features & ENC_FEATURE_ADAPTIVE_B);
    fcgCBAdaptiveI->Enabled   = 0 != (available_features & ENC_FEATURE_ADAPTIVE_I);
    fcgCBBPyramid->Enabled    = 0 != (available_features & ENC_FEATURE_B_PYRAMID);
    fcgLBLookaheadDS->Enabled = 0 != (available_features & ENC_FEATURE_LA_DS);
    fcgCXLookaheadDS->Enabled = 0 != (available_features & ENC_FEATURE_LA_DS);
    if (!fcgCBAdaptiveB->Enabled)   fcgCBAdaptiveB->Checked = false;
    if (!fcgCBAdaptiveI->Enabled)   fcgCBAdaptiveI->Checked = false;
    if (!fcgCBBPyramid->Enabled)    fcgCBBPyramid->Checked  = false;
    if (!fcgCXLookaheadDS->Enabled) fcgCXLookaheadDS->SelectedIndex = 0;

    //API v1.9 features
    fcgNUQPMin->Enabled        = 0 != (available_features & ENC_FEATURE_QP_MINMAX);
    fcgNUQPMax->Enabled        = 0 != (available_features & ENC_FEATURE_QP_MINMAX);
    fcgLBIntraRefreshCycle->Enabled = 0 != (available_features & ENC_FEATURE_INTRA_REFRESH);
    fcgNUIntraRefreshCycle->Enabled = 0 != (available_features & ENC_FEATURE_INTRA_REFRESH);
    fcgCBDeblock->Enabled      = 0 != (available_features & ENC_FEATURE_NO_DEBLOCK);
    if (!fcgNUQPMin->Enabled)        fcgNUQPMin->Value = 0;
    if (!fcgNUQPMax->Enabled)        fcgNUQPMax->Value = 0;
    if (!fcgNUIntraRefreshCycle->Enabled) fcgNUIntraRefreshCycle->Value = 0;
    if (!fcgCBDeblock->Enabled)      fcgCBDeblock->Checked = true;

    //API v1.11 features
    fcgCheckRCModeLibVersion(MFX_RATECONTROL_LA_HRD, MFX_RATECONTROL_VBR, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_LA_HRD), codec, funcMode));
    //fcgCheckRCModeLibVersion(MFX_RATECONTROL_LA_EXT, MFX_RATECONTROL_VBR, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_LA_EXT), codec, funcMode));
    fcgCheckRCModeLibVersion(MFX_RATECONTROL_QVBR,   MFX_RATECONTROL_VBR, featuresHW->getRCAvail(fcgCXDevice->SelectedIndex, get_cx_index(list_encmode, MFX_RATECONTROL_QVBR),   codec, funcMode));
    available_features = featuresHW->getFeatureOfRC(fcgCXDevice->SelectedIndex, fcgCXEncMode->SelectedIndex, codec, funcMode); // fcgCXEncMode->SelectedIndex が変わっている可能性があるので再取得
    fcgLBWinBRCSize->Enabled     = 0 != (available_features & ENC_FEATURE_WINBRC);
    fcgLBWinBRCSizeAuto->Enabled = 0 != (available_features & ENC_FEATURE_WINBRC);
    fcgNUWinBRCSize->Enabled     = 0 != (available_features & ENC_FEATURE_WINBRC);
    if (!fcgNUWinBRCSize->Enabled) fcgNUWinBRCSize->Value = 0;

    //API v1.13 features
    fcgLBMVCostScaling->Enabled    = 0 != (available_features & ENC_FEATURE_GLOBAL_MOTION_ADJUST);
    fcgCXMVCostScaling->Enabled    = 0 != (available_features & ENC_FEATURE_GLOBAL_MOTION_ADJUST);
    fcgCBDirectBiasAdjust->Enabled = 0 != (available_features & ENC_FEATURE_DIRECT_BIAS_ADJUST);
    if (!fcgCXMVCostScaling->Enabled)    fcgCXMVCostScaling->SelectedIndex = 0;
    if (!fcgCBDirectBiasAdjust->Enabled) fcgCBDirectBiasAdjust->Checked = false;

    //API v1.16 features
    fcgCBWeightP->Enabled          = 0 != (available_features & ENC_FEATURE_WEIGHT_P);
    fcgCBWeightB->Enabled          = 0 != (available_features & ENC_FEATURE_WEIGHT_B);
    if (!fcgCBWeightP->Enabled) fcgCBWeightP->Checked = false;
    if (!fcgCBWeightB->Enabled) fcgCBWeightB->Checked = false;

    fcgCXBitDepth->Enabled         = 0 != (available_features & ENC_FEATURE_10BIT_DEPTH);
    if (!fcgCXBitDepth->Enabled)   fcgCXBitDepth->SelectedIndex = 0;

    fcgCXEncMode->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
    fcgCXEncMode->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::CheckOtherChanges);
}

System::Void frmConfig::fcgChangeEnabled(System::Object^  sender, System::EventArgs^  e) {
    //もしfeatureListが作成できていなければ、チェックを行わない
    if (featuresHW == nullptr || fcgCXDevice->SelectedIndex < 0 || !featuresHW->checkIfGetFeaturesFinished()) {
        return;
    }

    this->SuspendLayout();
    const auto codec = get_out_enc_codec_by_index(fcgCXEncCodec->SelectedIndex);
    const QSVFunctionMode funcMode = (QSVFunctionMode)list_qsv_function_mode[fcgCXFunctionMode->SelectedIndex].value;
    const mfxU64 available_features = featuresHW->getFeatureOfRC(fcgCXDevice->SelectedIndex, fcgCXEncMode->SelectedIndex, codec, funcMode);

    //つぎに全体のチェックを行う
    int enc_mode = list_encmode[fcgCXEncMode->SelectedIndex].value;
    bool cqp_mode =     (enc_mode == MFX_RATECONTROL_CQP);
    bool avbr_mode =    (enc_mode == MFX_RATECONTROL_AVBR);
    bool cbr_vbr_mode = (enc_mode == MFX_RATECONTROL_VBR    || enc_mode == MFX_RATECONTROL_CBR);
    bool la_mode =      (rc_is_type_lookahead(enc_mode));
    bool icq_mode =     (enc_mode == MFX_RATECONTROL_LA_ICQ || enc_mode == MFX_RATECONTROL_ICQ);
    bool qvbr_mode =    (enc_mode == MFX_RATECONTROL_QVBR);
    //bool vcm_mode =     (enc_mode == MFX_RATECONTROL_VCM);

    fcgPNQP->Visible = cqp_mode;
    fcgNUQPI->Enabled = cqp_mode;
    fcgNUQPP->Enabled = cqp_mode;
    fcgNUQPB->Enabled = cqp_mode;
    fcgLBQPI->Enabled = cqp_mode;
    fcgLBQPP->Enabled = cqp_mode;
    fcgLBQPB->Enabled = cqp_mode;

    fcgPNBitrate->Visible = !cqp_mode;
    fcgNUBitrate->Enabled = !cqp_mode;
    fcgLBBitrate->Enabled = !cqp_mode;
    bool enableMaxKbps = cbr_vbr_mode || qvbr_mode || (available_features & ENC_FEATURE_WINBRC);
    fcgNUMaxkbps->Enabled = enableMaxKbps;
    fcgLBMaxkbps->Enabled = enableMaxKbps;
    fcgLBMaxBitrate2->Enabled = enableMaxKbps;

    fcgPNAVBR->Visible = avbr_mode;
    fcgLBAVBRAccuarcy->Enabled = avbr_mode;
    fcgLBAVBRAccuarcy2->Enabled = avbr_mode;
    fcgNUAVBRAccuarcy->Enabled = avbr_mode;
    fcgLBAVBRConvergence->Enabled = avbr_mode;
    fcgLBAVBRConvergence2->Enabled = avbr_mode;
    fcgNUAVBRConvergence->Enabled = avbr_mode;

    fcgPNLookahead->Visible = la_mode;
    fcgLBLookaheadDepth->Enabled = la_mode;
    fcgNULookaheadDepth->Enabled = la_mode;
    fcgLBLookaheadDS->Visible = la_mode;
    fcgCXLookaheadDS->Visible = la_mode;
    fcgLBWinBRCSize->Visible = la_mode;
    fcgLBWinBRCSizeAuto->Visible = la_mode;
    fcgNUWinBRCSize->Visible = la_mode;

    fcgPNExtSettings->Visible = false;

    fcgPNICQ->Visible = icq_mode;
    fcgPNQVBR->Visible = qvbr_mode;

    fcggroupBoxResize->Enabled = fcgCBVppResize->Checked;
    fcgPNVppDenoiseMFX->Visible = (fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("denoise")));
    fcgPNVppDenoiseKnn->Visible = (fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("knn")));
    fcgPNVppDenoiseNLMeans->Visible = (fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("nlmeans")));
    fcgPNVppDenoisePmd->Visible = (fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("pmd")));
    fcgPNVppDenoiseSmooth->Visible = (fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("smooth")));
    fcgPNVppDenoiseDct->Visible = (fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("denoise-dct")));
    fcgPNVppDenoiseFFT3D->Visible = (fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("fft3d")));
    fcgPNVppDenoiseConv3D->Visible = (fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("convolution3d")));
    fcgPNVppDetailEnhanceMFX->Visible = (fcgCXVppDetailEnhance->SelectedIndex == get_cx_index(list_vpp_detail_enahance, _T("detail-enhance")));
    fcgPNVppUnsharp->Visible = (fcgCXVppDetailEnhance->SelectedIndex == get_cx_index(list_vpp_detail_enahance, _T("unsharp")));
    fcgPNVppEdgelevel->Visible = (fcgCXVppDetailEnhance->SelectedIndex == get_cx_index(list_vpp_detail_enahance, _T("edgelevel")));
    fcgPNVppWarpsharp->Visible = (fcgCXVppDetailEnhance->SelectedIndex == get_cx_index(list_vpp_detail_enahance, _T("warpsharp")));
    fcgPNVppAfs->Visible = (fcgCXVppDeinterlace->SelectedIndex == get_cx_index(list_deinterlace_gui, L"自動フィールドシフト"));
    fcgPNVppNnedi->Visible = (fcgCXVppDeinterlace->SelectedIndex == get_cx_index(list_deinterlace_gui, L"nnedi"));
    fcgPNVppYadif->Visible = (fcgCXVppDeinterlace->SelectedIndex == get_cx_index(list_deinterlace_gui, L"yadif"));
    fcgPNVppDecomb->Visible = (fcgCXVppDeinterlace->SelectedIndex == get_cx_index(list_deinterlace_gui, L"decomb"));
    fcgPNVppDeband->Visible = (fcgCXVppDeband->SelectedIndex == get_cx_index(list_vpp_deband_names, _T("deband")));
    fcgPNVppLibplaceboDeband->Visible = (fcgCXVppDeband->SelectedIndex == get_cx_index(list_vpp_deband_names, _T("libplacebo-deband")));

    this->ResumeLayout();
    this->PerformLayout();
}

System::Void frmConfig::fcgCheckVppFeatures() {
#if 0
    //swモードは使用しない
    fcgCBHWEncode->Checked = true;
    UInt64 available_features = (fcgCBHWEncode->Checked) ? featuresHW->getVppFeatures() : featuresSW->getVppFeatures();
    fcgCBVppResize->Enabled = 0 != (available_features & VPP_FEATURE_RESIZE);
    if (!fcgCBVppResize->Enabled) fcgCBVppResize->Checked;

    fcgCBVppDenoise->Enabled = 0 != (available_features & VPP_FEATURE_DENOISE);
    if (!fcgCBVppDenoise->Enabled) fcgCBVppDenoise->Checked;

    fcgCBVppDetail->Enabled = 0 != (available_features & VPP_FEATURE_DETAIL_ENHANCEMENT);
    if (!fcgCBVppDetail->Enabled) fcgCBVppDetail->Checked;

#if ENABLE_FPS_CONVERSION
    fcgCXFPSConversion->Enabled = 0 != (available_features & VPP_FEATURE_FPS_CONVERSION_ADV);
    fcgLBFPSConversion->Enabled = fcgCXFPSConversion->Enabled;
    if (!fcgCXFPSConversion->Enabled) fcgCXFPSConversion->SelectedIndex = 0;
#else
    //うまくうごいてなさそうなので無効化
    fcgCXFPSConversion->Visible = false;
    fcgLBFPSConversion->Visible = false;
    fcgCXFPSConversion->SelectedIndex = 0;
#endif

    fcgCXImageStabilizer->Enabled = 0 != (available_features & VPP_FEATURE_IMAGE_STABILIZATION);
    fcgLBImageStabilizer->Enabled = fcgCXImageStabilizer->Enabled;
    if (!fcgCXImageStabilizer->Enabled) fcgCXImageStabilizer->SelectedIndex = 0;
#endif
}

System::Void frmConfig::fcgDevOutputTypeFFPGChanged(System::Object^  sender, System::EventArgs^  e) {
    if (updateFeatureTableFlag) {
        return;
    }
    if (featuresHW == nullptr || fcgCXDevice->SelectedIndex < 0 || !featuresHW->checkIfGetFeaturesFinished()) {
        return;
    }

    updateFeatureTableFlag = true;

    this->SuspendLayout();

    if (fcgCheckCodec() || sender == fcgCXEncCodec) {
        setComboBox(fcgCXCodecLevel, get_level_list(get_out_enc_codec_by_index(fcgCXEncCodec->SelectedIndex)));
        setComboBox(fcgCXCodecProfile, get_profile_list(get_out_enc_codec_by_index(fcgCXEncCodec->SelectedIndex)));
        fcgCXCodecLevel->SelectedIndex = 0;
        fcgCXCodecProfile->SelectedIndex = 0;
    }
    fcgCheckFixedFunc();

    //まず、レート制御モードのみのチェックを行う
    fcgCheckLibRateControl();

    //つぎに全体のチェックを行う
    fcgCheckLibVersion();

    UpdateFeatures(false);
    fcgChangeEnabled(sender, e);

    this->ResumeLayout();
    this->PerformLayout();

    updateFeatureTableFlag = false;
}

System::Void frmConfig::fcgCXFunctionMode_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
    fcgDevOutputTypeFFPGChanged(sender, e);
}

System::Void frmConfig::fcgCXDevice_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
    fcgDevOutputTypeFFPGChanged(sender, e);
}

System::Void frmConfig::frmConfig::fcgCXOutputType_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
    fcgDevOutputTypeFFPGChanged(sender, e);
}

System::Void frmConfig::fcgChangeMuxerVisible(System::Object^  sender, System::EventArgs^  e) {
    //tc2mp4のチェック
    const bool enable_tc2mp4_muxer = (0 != str_has_char(sys_dat->exstg->s_mux[MUXER_TC2MP4].base_cmd));
    fcgTXTC2MP4Path->Visible = enable_tc2mp4_muxer;
    fcgLBTC2MP4Path->Visible = enable_tc2mp4_muxer;
    fcgBTTC2MP4Path->Visible = enable_tc2mp4_muxer;
    fcgCBAFS->Enabled        = enable_tc2mp4_muxer;
    fcgCBAuoTcfileout->Enabled = enable_tc2mp4_muxer;
    //mp4 rawのチェック
    const bool enable_mp4raw_muxer = (0 != str_has_char(sys_dat->exstg->s_mux[MUXER_MP4_RAW].base_cmd));
    fcgTXMP4RawPath->Visible = enable_mp4raw_muxer;
    fcgLBMP4RawPath->Visible = enable_mp4raw_muxer;
    fcgBTMP4RawPath->Visible = enable_mp4raw_muxer;
    //一時フォルダのチェック
    const bool enable_mp4_tmp = (0 != str_has_char(sys_dat->exstg->s_mux[MUXER_MP4].tmp_cmd));
    fcgCXMP4BoxTempDir->Visible = enable_mp4_tmp;
    fcgLBMP4BoxTempDir->Visible = enable_mp4_tmp;
    fcgTXMP4BoxTempDir->Visible = enable_mp4_tmp;
    fcgBTMP4BoxTempDir->Visible = enable_mp4_tmp;
    //Apple Chapterのチェック
    bool enable_mp4_apple_cmdex = false;
    for (int i = 0; i < sys_dat->exstg->s_mux[MUXER_MP4].ex_count; i++)
        enable_mp4_apple_cmdex |= (0 != str_has_char(sys_dat->exstg->s_mux[MUXER_MP4].ex_cmd[i].cmd_apple));
    fcgCBMP4MuxApple->Visible = enable_mp4_apple_cmdex;

    //位置の調整
    static const int HEIGHT = 31;
    fcgLBTC2MP4Path->Location = Point(fcgLBTC2MP4Path->Location.X, fcgLBMP4MuxerPath->Location.Y + HEIGHT * enable_tc2mp4_muxer);
    fcgTXTC2MP4Path->Location = Point(fcgTXTC2MP4Path->Location.X, fcgTXMP4MuxerPath->Location.Y + HEIGHT * enable_tc2mp4_muxer);
    fcgBTTC2MP4Path->Location = Point(fcgBTTC2MP4Path->Location.X, fcgBTMP4MuxerPath->Location.Y + HEIGHT * enable_tc2mp4_muxer);
    fcgLBMP4RawPath->Location = Point(fcgLBMP4RawPath->Location.X, fcgLBTC2MP4Path->Location.Y   + HEIGHT * enable_mp4raw_muxer);
    fcgTXMP4RawPath->Location = Point(fcgTXMP4RawPath->Location.X, fcgTXTC2MP4Path->Location.Y   + HEIGHT * enable_mp4raw_muxer);
    fcgBTMP4RawPath->Location = Point(fcgBTMP4RawPath->Location.X, fcgBTTC2MP4Path->Location.Y   + HEIGHT * enable_mp4raw_muxer);
}

System::Void frmConfig::SetStgEscKey(bool Enable) {
    if (this->KeyPreview == Enable)
        return;
    this->KeyPreview = Enable;
    if (Enable)
        this->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &frmConfig::frmConfig_KeyDown);
    else
        this->KeyDown -= gcnew System::Windows::Forms::KeyEventHandler(this, &frmConfig::frmConfig_KeyDown);
}

System::Void frmConfig::AdjustLocation() {
    //デスクトップ領域(タスクバー等除く)
    System::Drawing::Rectangle screen = System::Windows::Forms::Screen::GetWorkingArea(this);
    //現在のデスクトップ領域の座標
    Point CurrentDesktopLocation = this->DesktopLocation::get();
    //チェック開始
    bool ChangeLocation = false;
    if (CurrentDesktopLocation.X + this->Size.Width > screen.Width) {
        ChangeLocation = true;
        CurrentDesktopLocation.X = clamp(screen.X - this->Size.Width, 4, CurrentDesktopLocation.X);
    }
    if (CurrentDesktopLocation.Y + this->Size.Height > screen.Height) {
        ChangeLocation = true;
        CurrentDesktopLocation.Y = clamp(screen.Y - this->Size.Height, 4, CurrentDesktopLocation.Y);
    }
    if (ChangeLocation) {
        this->StartPosition = FormStartPosition::Manual;
        this->DesktopLocation::set(CurrentDesktopLocation);
    }
}

System::Void frmConfig::SetInputBufRange() {
    fcgNUInputBufSize->Minimum = QSV_INPUT_BUF_MIN;
    fcgNUInputBufSize->Maximum = QSV_INPUT_BUF_MAX;
}

System::Void frmConfig::UpdateMfxLibDetection() {
    if (featuresHW != nullptr) {
        UInt32 mfxlib_hw = featuresHW->GetmfxLibVer();
        fcgLBMFXLibDetectionHwValue->Text = (check_lib_version(mfxlib_hw, MFX_LIB_VERSION_1_1.Version)) ?
            L"v" + ((mfxlib_hw>>16).ToString() + L"." + (mfxlib_hw & 0x0000ffff).ToString()) : L"-----";

        array<String^>^ deviceNames = featuresHW->GetDeviceNames();
        fcgCXDevice->BeginUpdate();
        fcgCXDevice->Items->Clear();
        if (deviceNames) {
            for (int i = 0; i < deviceNames->Length; i++)
                fcgCXDevice->Items->Add(deviceNames[i]);
        }
        fcgCXDevice->EndUpdate();
    }
}

System::Void frmConfig::InitForm() {
    //UIテーマ切り替え
    CheckTheme();
    //言語設定ファイルのロード
    InitLangList();
    //CPU情報の取得
    getCPUInfoDelegate = gcnew SetCPUInfoDelegate(this, &frmConfig::SetCPUInfo);
    getCPUInfoDelegate->BeginInvoke(nullptr, nullptr);
    //設定ファイル集の初期化
    InitStgFileList();
    //言語表示
    LoadLangText();
    //バッファサイズの最大最少をセット
    SetInputBufRange();
    //HWエンコードの可否
    UpdateMfxLibDetection();
    //パラメータセット
    ConfToFrm(conf);
    //イベントセット
    SetTXMaxLenAll(); //テキストボックスの最大文字数
    SetAllCheckChangedEvents(this); //変更の確認,ついでにNUのEnterEvent
    //表示位置の調整
    AdjustLocation();
    //キー設定
    SetStgEscKey(sys_dat->exstg->s_local.enable_stg_esc_key != 0);
    //フォントの設定
    if (str_has_char(sys_dat->exstg->s_local.conf_font.name))
        SetFontFamilyToForm(this, gcnew FontFamily(String(sys_dat->exstg->s_local.conf_font.name).ToString()), this->Font->FontFamily);
    //フォームの変更可不可を更新
    fcgChangeMuxerVisible(nullptr, nullptr);
    EnableSettingsNoteChange(false);
    UpdateFeatures(false);
    fcgChangeEnabled(nullptr, nullptr);
    fcgCBAudioUseExt_CheckedChanged(nullptr, nullptr);
    fcgRebuildCmd(nullptr, nullptr);
}

System::Void frmConfig::LoadLangText() {
    //一度ウィンドウの再描画を完全に抑止する
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 0, 0);
    //空白時にグレーで入れる文字列を言語変更のため一度空白に戻す
    ExeTXPathEnter();
    //言語更新開始
    LOAD_CLI_TEXT(fcgtoolStripSettings);
    LOAD_CLI_TEXT(fcgTSBSave);
    LOAD_CLI_TEXT(fcgTSBSaveNew);
    LOAD_CLI_TEXT(fcgTSBDelete);
    LOAD_CLI_TEXT(fcgTSSettings);
    LOAD_CLI_TEXT(fcgTSBBitrateCalc);
    LOAD_CLI_TEXT(fcgTSBOtherSettings);
    LOAD_CLI_TEXT(fcgTSLSettingsNotes);
    LOAD_CLI_TEXT(fcgTSTSettingsNotes);
    LOAD_CLI_TEXT(fcgtabPageMP4);
    LOAD_CLI_TEXT(fcgBTMP4RawPath);
    LOAD_CLI_TEXT(fcgLBMP4RawPath);
    LOAD_CLI_TEXT(fcgCBMP4MuxApple);
    LOAD_CLI_TEXT(fcgBTMP4BoxTempDir);
    LOAD_CLI_TEXT(fcgLBMP4BoxTempDir);
    LOAD_CLI_TEXT(fcgBTTC2MP4Path);
    LOAD_CLI_TEXT(fcgBTMP4MuxerPath);
    LOAD_CLI_TEXT(fcgLBTC2MP4Path);
    LOAD_CLI_TEXT(fcgLBMP4MuxerPath);
    LOAD_CLI_TEXT(fcgLBMP4CmdEx);
    LOAD_CLI_TEXT(fcgCBMP4MuxerExt);
    LOAD_CLI_TEXT(fcgtabPageMKV);
    LOAD_CLI_TEXT(fcgBTMKVMuxerPath);
    LOAD_CLI_TEXT(fcgLBMKVMuxerPath);
    LOAD_CLI_TEXT(fcgLBMKVMuxerCmdEx);
    LOAD_CLI_TEXT(fcgCBMKVMuxerExt);
    LOAD_CLI_TEXT(fcgtabPageMux);
    LOAD_CLI_TEXT(fcgLBMuxPriority);
    LOAD_CLI_TEXT(fcgCBMuxMinimize);
    LOAD_CLI_TEXT(fcgtabPageBat);
    LOAD_CLI_TEXT(fcgLBBatAfterString);
    LOAD_CLI_TEXT(fcgLBBatBeforeString);
    LOAD_CLI_TEXT(fcgBTBatBeforePath);
    LOAD_CLI_TEXT(fcgLBBatBeforePath);
    LOAD_CLI_TEXT(fcgCBWaitForBatBefore);
    LOAD_CLI_TEXT(fcgCBRunBatBefore);
    LOAD_CLI_TEXT(fcgBTBatAfterPath);
    LOAD_CLI_TEXT(fcgLBBatAfterPath);
    LOAD_CLI_TEXT(fcgCBWaitForBatAfter);
    LOAD_CLI_TEXT(fcgCBRunBatAfter);
    LOAD_CLI_TEXT(fcgtabPageInternal);
    LOAD_CLI_TEXT(fcgLBInternalCmdEx);
    LOAD_CLI_TEXT(fcgBTCancel);
    LOAD_CLI_TEXT(fcgBTOK);
    LOAD_CLI_TEXT(fcgBTDefault);
    LOAD_CLI_TEXT(fcgLBVersionDate);
    LOAD_CLI_TEXT(fcgLBVersion);
    LOAD_CLI_TEXT(tabPageVideoEnc);
    LOAD_CLI_TEXT(fcgLBVideoFormat);
    LOAD_CLI_TEXT(fcggroupBoxColorMatrix);
    LOAD_CLI_TEXT(fcgLBFullrange);
    LOAD_CLI_TEXT(fcgLBTransfer);
    LOAD_CLI_TEXT(fcgLBColorPrim);
    LOAD_CLI_TEXT(fcgLBColorMatrix);
    LOAD_CLI_TEXT(fcgGroupBoxAspectRatio);
    LOAD_CLI_TEXT(fcgLBAspectRatio);
    LOAD_CLI_TEXT(fcgLBInterlaced);
    LOAD_CLI_TEXT(fcgBTVideoEncoderPath);
    LOAD_CLI_TEXT(fcgLBVideoEncoderPath);
    LOAD_CLI_TEXT(fcgCBAFS);
    LOAD_CLI_TEXT(fcgLBBitrate);
    LOAD_CLI_TEXT(fcgLBBitrate2);
    LOAD_CLI_TEXT(fcgLBMaxkbps);
    LOAD_CLI_TEXT(fcgLBMaxBitrate2);
    LOAD_CLI_TEXT(fcgLBQPI);
    LOAD_CLI_TEXT(fcgLBQPP);
    LOAD_CLI_TEXT(fcgLBQPB);
    LOAD_CLI_TEXT(fcgLBCodecLevel);
    LOAD_CLI_TEXT(fcgLBCodecProfile);
    LOAD_CLI_TEXT(fcgLBBframes);
    LOAD_CLI_TEXT(fcgLBEncMode);
    LOAD_CLI_TEXT(fcgLBRefFrames);
    LOAD_CLI_TEXT(fcgLBGOPLengthAuto);
    LOAD_CLI_TEXT(fcgLBGOPLength);
    LOAD_CLI_TEXT(fcgLBQualityPreset);
    LOAD_CLI_TEXT(fcgLBEncCodec);
    LOAD_CLI_TEXT(fcgLBSlices);
    LOAD_CLI_TEXT(fcgLBSlices2);
    LOAD_CLI_TEXT(fcgLBBitDepth);
    LOAD_CLI_TEXT(fcgLBHyperMode);
    LOAD_CLI_TEXT(fcgLBDevice);
    LOAD_CLI_TEXT(fcgLBOutputCsp);
    LOAD_CLI_TEXT(fcgLBAVBRConvergence);
    LOAD_CLI_TEXT(fcgLBAVBRAccuarcy);
    LOAD_CLI_TEXT(fcgLBAVBRAccuarcy2);
    LOAD_CLI_TEXT(fcgLBAVBRConvergence2);
    //LOAD_CLI_TEXT(fcgLBMFXLibDetectionHwValue);
    LOAD_CLI_TEXT(fcgLBMFXLibDetectionHwStatus);
    LOAD_CLI_TEXT(fcgCBFadeDetect);
    LOAD_CLI_TEXT(fcgCBWeightB);
    LOAD_CLI_TEXT(fcgCBWeightP);
    LOAD_CLI_TEXT(fcgLBWinBRCSizeAuto);
    LOAD_CLI_TEXT(fcgLBWinBRCSize);
    LOAD_CLI_TEXT(fcgLBQPMinMaxAuto);
    LOAD_CLI_TEXT(fcgLBQPMax);
    LOAD_CLI_TEXT(fcgLBQPMinMAX);
    LOAD_CLI_TEXT(fcgLBICQQuality);
    LOAD_CLI_TEXT(fcgLBLookaheadDS);
    LOAD_CLI_TEXT(fcgCBBPyramid);
    LOAD_CLI_TEXT(fcgCBAdaptiveB);
    LOAD_CLI_TEXT(fcgCBAdaptiveI);
    LOAD_CLI_TEXT(fcgLBBlurayCompat);
    LOAD_CLI_TEXT(fcgLBMFXLibDetection);
    LOAD_CLI_TEXT(fcgLBRefAuto);
    LOAD_CLI_TEXT(fcgLBBframesAuto);
    LOAD_CLI_TEXT(fcgCBOpenGOP);
    LOAD_CLI_TEXT(fcgCBOutputPicStruct);
    LOAD_CLI_TEXT(fcgCBOutputAud);
    LOAD_CLI_TEXT(fcggroupBoxDetail);
    LOAD_CLI_TEXT(fcgCBDirectBiasAdjust);
    LOAD_CLI_TEXT(fcgLBMVCostScaling);
    LOAD_CLI_TEXT(fcgCBExtBRC);
    LOAD_CLI_TEXT(fcgCBMBBRC);
    LOAD_CLI_TEXT(fcgLBTrellis);
    LOAD_CLI_TEXT(fcgLBIntraRefreshCycle);
    LOAD_CLI_TEXT(fcgCBDeblock);
    LOAD_CLI_TEXT(fcgLBInterPred);
    LOAD_CLI_TEXT(fcgLBIntraPred);
    LOAD_CLI_TEXT(fcgLBMVPred);
    LOAD_CLI_TEXT(fcgLBMVWindowSize);
    LOAD_CLI_TEXT(fcgLBMVSearch);
    LOAD_CLI_TEXT(fcgCBRDO);
    LOAD_CLI_TEXT(fcgCBCABAC);
    LOAD_CLI_TEXT(fcgCBD3DMemAlloc);
    LOAD_CLI_TEXT(fcgLBLookaheadDepth2);
    LOAD_CLI_TEXT(fcgLBLookaheadDepth);
    LOAD_CLI_TEXT(fcgLBQVBR);
    LOAD_CLI_TEXT(tabPageVpp);
    LOAD_CLI_TEXT(fcgLBVppMctf);
    LOAD_CLI_TEXT(fcgLBVppDetailEnhanceMFX);
    LOAD_CLI_TEXT(fcgLBVppDenoiseMFX);
    LOAD_CLI_TEXT(fcgLBImageStabilizer);
    LOAD_CLI_TEXT(fcgLBRotate);
    LOAD_CLI_TEXT(fcggroupBoxVppDeband);
    LOAD_CLI_TEXT(fcgCBVppDebandRandEachFrame);
    LOAD_CLI_TEXT(fcgCBVppDebandBlurFirst);
    LOAD_CLI_TEXT(fcgLBVppDebandSample);
    LOAD_CLI_TEXT(fcgLBVppDebandDitherC);
    LOAD_CLI_TEXT(fcgLBVppDebandDitherY);
    LOAD_CLI_TEXT(fcgLBVppDebandDither);
    LOAD_CLI_TEXT(fcgLBVppDebandThreCr);
    LOAD_CLI_TEXT(fcgLBVppDebandThreCb);
    LOAD_CLI_TEXT(fcgLBVppDebandThreY);
    LOAD_CLI_TEXT(fcgLBVppDebandThreshold);
    LOAD_CLI_TEXT(fcgLBVppDebandRange);
    LOAD_CLI_TEXT(fcgLBVppLibplaceboDebandIteration);
    LOAD_CLI_TEXT(fcgLBVppLibplaceboDebandRadius);
    LOAD_CLI_TEXT(fcgLBVppLibplaceboDebandThreshold);
    LOAD_CLI_TEXT(fcgLBVppLibplaceboDebandGrain);
    LOAD_CLI_TEXT(fcgLBVppLibplaceboDebandDither);
    LOAD_CLI_TEXT(fcgLBVppLibplaceboDebandLUTSize);
    LOAD_CLI_TEXT(fcggroupBoxVppDetailEnahance);
    LOAD_CLI_TEXT(fcgLBVppWarpsharpDepth);
    LOAD_CLI_TEXT(fcgLBVppWarpsharpThreshold);
    LOAD_CLI_TEXT(fcgLBVppWarpsharpType);
    LOAD_CLI_TEXT(fcgLBVppWarpsharpBlur);
    LOAD_CLI_TEXT(fcgLBVppEdgelevelWhite);
    LOAD_CLI_TEXT(fcgLBVppEdgelevelThreshold);
    LOAD_CLI_TEXT(fcgLBVppEdgelevelBlack);
    LOAD_CLI_TEXT(fcgLBVppEdgelevelStrength);
    LOAD_CLI_TEXT(fcgLBVppUnsharpThreshold);
    LOAD_CLI_TEXT(fcgLBVppUnsharpWeight);
    LOAD_CLI_TEXT(fcgLBVppUnsharpRadius);
    LOAD_CLI_TEXT(fcggroupBoxVppDenoise);
    LOAD_CLI_TEXT(fcgLBVppDenoiseConv3DMatrix);
    LOAD_CLI_TEXT(fcgLBVppDenoiseConv3DThreshTemporal);
    LOAD_CLI_TEXT(fcgLBVppDenoiseConv3DThreshSpatial);
    LOAD_CLI_TEXT(fcgLBVppDenoiseConv3DThreshCTemporal);
    LOAD_CLI_TEXT(fcgLBVppDenoiseConv3DThreshCSpatial);
    LOAD_CLI_TEXT(fcgLBVppDenoiseConv3DThreshYTemporal);
    LOAD_CLI_TEXT(fcgLBVppDenoiseConv3DThreshYSpatial);
    LOAD_CLI_TEXT(fcgLBVppDenoiseSmoothQP);
    LOAD_CLI_TEXT(fcgLBVppDenoiseSmoothQuality);
    LOAD_CLI_TEXT(fcgLBVppDenoiseDctStep);
    LOAD_CLI_TEXT(fcgLBVppDenoiseDctSigma);
    LOAD_CLI_TEXT(fcgLBVppDenoiseDctBlockSize);
    LOAD_CLI_TEXT(fcgLBVppDenoiseFFT3DSigma);
    LOAD_CLI_TEXT(fcgLBVppDenoiseFFT3DAmount);
    LOAD_CLI_TEXT(fcgLBVppDenoiseFFT3DBlockSize);
    LOAD_CLI_TEXT(fcgLBVppDenoiseFFT3DOverlap);
    LOAD_CLI_TEXT(fcgLBVppDenoiseFFT3DTemporal);
    LOAD_CLI_TEXT(fcgLBVppDenoiseFFT3DPrecision);
    LOAD_CLI_TEXT(fcgLBVppDenoiseKnnThreshold);
    LOAD_CLI_TEXT(fcgLBVppDenoiseKnnStrength);
    LOAD_CLI_TEXT(fcgLBVppDenoiseKnnRadius);
    LOAD_CLI_TEXT(fcgLBVppDenoiseNLMeansPatch);
    LOAD_CLI_TEXT(fcgLBVppDenoiseNLMeansSearch);
    LOAD_CLI_TEXT(fcgLBVppDenoiseNLMeansSigma);
    LOAD_CLI_TEXT(fcgLBVppDenoiseNLMeansH);
    LOAD_CLI_TEXT(fcgLBVppDenoisePmdThreshold);
    LOAD_CLI_TEXT(fcgLBVppDenoisePmdStrength);
    LOAD_CLI_TEXT(fcgLBVppDenoisePmdApplyCount);
    LOAD_CLI_TEXT(fcggroupBoxVppDeinterlace);
    LOAD_CLI_TEXT(fcgLBVppDeinterlace);
    LOAD_CLI_TEXT(fcgLBVppAfsThreCMotion);
    LOAD_CLI_TEXT(fcgLBVppAfsThreYmotion);
    LOAD_CLI_TEXT(fcgLBVppAfsThreDeint);
    LOAD_CLI_TEXT(fcgLBVppAfsThreShift);
    LOAD_CLI_TEXT(fcgLBVppAfsCoeffShift);
    LOAD_CLI_TEXT(fcgLBVppAfsRight);
    LOAD_CLI_TEXT(fcgLBVppAfsLeft);
    LOAD_CLI_TEXT(fcgLBVppAfsBottom);
    LOAD_CLI_TEXT(fcgLBVppAfsUp);
    LOAD_CLI_TEXT(fcgCBVppAfs24fps);
    LOAD_CLI_TEXT(fcgCBVppAfsTune);
    LOAD_CLI_TEXT(fcgCBVppAfsSmooth);
    LOAD_CLI_TEXT(fcgCBVppAfsDrop);
    LOAD_CLI_TEXT(fcgCBVppAfsShift);
    LOAD_CLI_TEXT(fcgLBVppAfsAnalyze);
    LOAD_CLI_TEXT(fcgLBVppAfsMethodSwitch);
    LOAD_CLI_TEXT(fcgLBVppYadifMode);
    LOAD_CLI_TEXT(fcgCBVppDecombFull);
    LOAD_CLI_TEXT(fcgCBVppDecombBlend);
    LOAD_CLI_TEXT(fcgLBVppDecombThreshold);
    LOAD_CLI_TEXT(fcgLBVppDecombDthreshold);
    LOAD_CLI_TEXT(fcgLBVppNnediErrorType);
    LOAD_CLI_TEXT(fcgLBVppNnediPrescreen);
    LOAD_CLI_TEXT(fcgLBVppNnediPrec);
    LOAD_CLI_TEXT(fcgLBVppNnediQual);
    LOAD_CLI_TEXT(fcgLBVppNnediNsize);
    LOAD_CLI_TEXT(fcgLBVppNnediNns);
    LOAD_CLI_TEXT(fcgCBVppResize);
    LOAD_CLI_TEXT(fcgLBVppResize);
    LOAD_CLI_TEXT(fcgCBPsnr);
    LOAD_CLI_TEXT(fcgCBSsim);
    LOAD_CLI_TEXT(fcgCBAvoidIdleClock);
    LOAD_CLI_TEXT(tabPageExOpt);
    LOAD_CLI_TEXT(fcgCBAuoTcfileout);
    LOAD_CLI_TEXT(fcgLBInputBufSize);
    LOAD_CLI_TEXT(fcgLBTempDir);
    LOAD_CLI_TEXT(fcgBTCustomTempDir);
    LOAD_CLI_TEXT(tabPageFeatures);
    LOAD_CLI_TEXT(fcgLBGPUInfoLabelOnFeatureTab);
    //LOAD_CLI_TEXT(fcgLBCPUInfoOnFeatureTab);
    //LOAD_CLI_TEXT(fcgLBGPUInfoOnFeatureTab);
    LOAD_CLI_TEXT(fcgLBCPUInfoLabelOnFeatureTab);
    LOAD_CLI_TEXT(fcgBTSaveFeatureList);
    //LOAD_CLI_TEXT(fcgLBFeaturesCurrentAPIVer);
    LOAD_CLI_TEXT(fcgLBFeaturesShowCurrentAPI);
    LOAD_CLI_TEXT(fcgTSExeFileshelp);
    LOAD_CLI_TEXT(fcgLBguiExBlog);
    LOAD_CLI_TEXT(fcgtabPageAudioMain);
    LOAD_CLI_TEXT(fcgLBAudioBitrateInternal);
    LOAD_CLI_TEXT(fcgLBAudioEncModeInternal);
    LOAD_CLI_TEXT(fcgCBAudioUseExt);
    LOAD_CLI_TEXT(fcgCBFAWCheck);
    LOAD_CLI_TEXT(fcgLBAudioDelayCut);
    LOAD_CLI_TEXT(fcgCBAudioEncTiming);
    LOAD_CLI_TEXT(fcgBTCustomAudioTempDir);
    LOAD_CLI_TEXT(fcgCBAudioUsePipe);
    LOAD_CLI_TEXT(fcgCBAudio2pass);
    LOAD_CLI_TEXT(fcgLBAudioEncMode);
    LOAD_CLI_TEXT(fcgBTAudioEncoderPath);
    LOAD_CLI_TEXT(fcgLBAudioEncoderPath);
    LOAD_CLI_TEXT(fcgCBAudioOnly);
    LOAD_CLI_TEXT(fcgLBAudioTemp);
    LOAD_CLI_TEXT(fcgLBAudioBitrate);
    LOAD_CLI_TEXT(fcgtabPageAudioOther);
    LOAD_CLI_TEXT(fcgLBBatAfterAudioString);
    LOAD_CLI_TEXT(fcgLBBatBeforeAudioString);
    LOAD_CLI_TEXT(fcgBTBatAfterAudioPath);
    LOAD_CLI_TEXT(fcgLBBatAfterAudioPath);
    LOAD_CLI_TEXT(fcgCBRunBatAfterAudio);
    LOAD_CLI_TEXT(fcgBTBatBeforeAudioPath);
    LOAD_CLI_TEXT(fcgLBBatBeforeAudioPath);
    LOAD_CLI_TEXT(fcgCBRunBatBeforeAudio);
    LOAD_CLI_TEXT(fcgLBAudioPriority);

    //ローカル設定のロード(ini変更を反映)
    LoadLocalStg();
    //ローカル設定の反映
    SetLocalStg();
    //コンボボックスの値を設定
    InitComboBox();
    //ツールチップ
    SetHelpToolTips();
    ActivateToolTip(sys_dat->exstg->s_local.disable_tooltip_help == FALSE);
    //タイムコードのappendix(後付修飾子)を反映
    fcgCBAuoTcfileout->Text = LOAD_CLI_STRING(AUO_CONFIG_TC_FILE_OUT) + L" (" + String(sys_dat->exstg->s_append.tc).ToString() + L")";
    { //タイトル表示,バージョン情報,コンパイル日時
        auto auo_full_name = g_auo_mes.get(AUO_GUIEX_FULL_NAME);
        if (auo_full_name == nullptr || wcslen(auo_full_name) == 0) auo_full_name = AUO_FULL_NAME_W;
        this->Text = String(auo_full_name).ToString();
        fcgLBVersion->Text = String(auo_full_name).ToString() + L" " + String(AUO_VERSION_STR).ToString();
        fcgLBVersionDate->Text = L"build " + String(__DATE__).ToString() + L" " + String(__TIME__).ToString();
    }
    UpdateFeatures(true);
    //空白時にグレーで入れる文字列を言語に即して復活させる
    ExeTXPathLeave();
    //一度ウィンドウの再描画を再開し、強制的に再描画させる
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 1, 0);
    this->Refresh();
}

/////////////         データ <-> GUI     /////////////
System::Void frmConfig::ConfToFrm(CONF_GUIEX *cnf) {
    this->SuspendLayout();

    sInputParams prm_qsv;
    parse_cmd(&prm_qsv, cnf->enc.cmd);

    SetCXIndex(fcgCXEncCodec,     get_cx_index(list_out_enc_codec, prm_qsv.codec));
    SetCXIndex(fcgCXEncMode,      get_cx_index(list_encmode, prm_qsv.rcParam.encMode));
    SetCXIndex(fcgCXQualityPreset,get_cx_index(list_quality, prm_qsv.nTargetUsage));
    SetCXIndex(fcgCXDevice,       (featuresHW) ? featuresHW->getDevIndex(prm_qsv.device) : 0);
    SetCXIndex(fcgCXHyperMode,    get_cx_index(list_hyper_mode, prm_qsv.hyperMode));
    SetNUValue(fcgNUBitrate,      prm_qsv.rcParam.bitrate);
    SetNUValue(fcgNUMaxkbps,      prm_qsv.rcParam.maxBitrate);
    SetNUValue(fcgNUQPI,          prm_qsv.rcParam.qp.qpI);
    SetNUValue(fcgNUQPP,          prm_qsv.rcParam.qp.qpP);
    SetNUValue(fcgNUQPB,          prm_qsv.rcParam.qp.qpB);
    SetNUValue(fcgNUICQQuality,   prm_qsv.rcParam.icqQuality);
    SetNUValue(fcgNUQVBR,         prm_qsv.rcParam.qvbrQuality);
    SetNUValue(fcgNUGopLength,    Convert::ToDecimal(prm_qsv.nGOPLength));
    SetNUValue(fcgNURef,          prm_qsv.nRef);
    if (gopRefDistAsBframe(prm_qsv.codec)) {
        SetNUValue(fcgNUBframes, prm_qsv.GopRefDist-1);
    } else {
        SetNUValue(fcgNUBframes, prm_qsv.GopRefDist);
    }
    SetCXIndex(fcgCXTrellis,      get_cx_index(list_avc_trellis, prm_qsv.nTrellis));
    SetCXIndex(fcgCXCodecLevel,   get_cx_index(get_level_list(prm_qsv.codec),   prm_qsv.CodecLevel));
    SetCXIndex(fcgCXCodecProfile, get_cx_index(get_profile_list(prm_qsv.codec), prm_qsv.CodecProfile));
    SetCXIndex(fcgCXFunctionMode, get_cx_index(list_qsv_function_mode, (int)prm_qsv.functionMode));
    if (fcgCBD3DMemAlloc->Enabled)
        fcgCBD3DMemAlloc->Checked = prm_qsv.memType != SYSTEM_MEMORY;
    SetNUValue(fcgNUAVBRAccuarcy, prm_qsv.rcParam.avbrAccuarcy / Convert::ToDecimal(10.0));
    SetNUValue(fcgNUAVBRConvergence, prm_qsv.rcParam.avbrConvergence);
    SetNUValue(fcgNULookaheadDepth, prm_qsv.nLookaheadDepth);
    fcgCBAdaptiveI->Checked     = prm_qsv.bAdaptiveI.value_or(false);
    fcgCBAdaptiveB->Checked     = prm_qsv.bAdaptiveB.value_or(false);
    fcgCBWeightP->Checked       = prm_qsv.nWeightP != MFX_WEIGHTED_PRED_UNKNOWN;
    fcgCBWeightB->Checked       = prm_qsv.nWeightB != MFX_WEIGHTED_PRED_UNKNOWN;
    fcgCBFadeDetect->Checked    = prm_qsv.nFadeDetect.value_or(false);
    fcgCBBPyramid->Checked      = prm_qsv.bBPyramid.value_or(false);
    SetCXIndex(fcgCXLookaheadDS,  get_cx_index(list_lookahead_ds, prm_qsv.nLookaheadDS));
    fcgCBMBBRC->Checked         = prm_qsv.bMBBRC.value_or(false);
    //fcgCBExtBRC->Checked        = prm_qsv.bExtBRC != 0;
    SetNUValue(fcgNUWinBRCSize,       prm_qsv.nWinBRCSize);
    SetCXIndex(fcgCXInterlaced,   get_cx_index(list_interlaced, prm_qsv.input.picstruct));
    if (prm_qsv.nPAR[0] * prm_qsv.nPAR[1] <= 0)
        prm_qsv.nPAR[0] = prm_qsv.nPAR[1] = 0;
    SetCXIndex(fcgCXAspectRatio, (prm_qsv.nPAR[0] < 0));
    SetNUValue(fcgNUAspectRatioX, abs(prm_qsv.nPAR[0]));
    SetNUValue(fcgNUAspectRatioY, abs(prm_qsv.nPAR[1]));
    fcgCBOpenGOP->Checked        = prm_qsv.bopenGOP;
    SetCXIndex(fcgCXOutputCsp,    get_cx_index(list_output_csp, prm_qsv.outputCsp));
    SetCXIndex(fcgCXBitDepth,     get_bit_depth_idx(prm_qsv.outputDepth));

    SetCXIndex(fcgCXScenarioInfo, get_cx_index(list_scenario_info, prm_qsv.scenarioInfo));
    SetNUValue(fcgNUSlices,       prm_qsv.nSlices);

    fcgCBBlurayCompat->Checked   = prm_qsv.nBluray != 0;

    SetNUValue(fcgNUQPMin,         prm_qsv.qpMin.qpI);
    SetNUValue(fcgNUQPMax,         prm_qsv.qpMax.qpI);

    fcgCBCABAC->Checked          = !prm_qsv.bCAVLC;
    fcgCBRDO->Checked            = prm_qsv.bRDO;
    SetNUValue(fcgNUMVSearchWindow, prm_qsv.MVSearchWindow.first);
    SetCXIndex(fcgCXMVPred,      get_cx_index(list_mv_presicion,    prm_qsv.nMVPrecision));
    SetCXIndex(fcgCXInterPred,   get_cx_index(list_pred_block_size, prm_qsv.nInterPred));
    SetCXIndex(fcgCXIntraPred,   get_cx_index(list_pred_block_size, prm_qsv.nIntraPred));

    fcgCBDirectBiasAdjust->Checked = prm_qsv.bDirectBiasAdjust.value_or(false);
    SetCXIndex(fcgCXMVCostScaling, (prm_qsv.bGlobalMotionAdjust) ? get_cx_index(list_mv_cost_scaling, prm_qsv.nMVCostScaling) : 0);

    fcgCBDeblock->Checked        = prm_qsv.bNoDeblock == 0;
    SetNUValue(fcgNUIntraRefreshCycle, prm_qsv.intraRefreshCycle);

    SetCXIndex(fcgCXTransfer,    get_cx_index(list_transfer,    prm_qsv.common.out_vui.transfer));
    SetCXIndex(fcgCXColorMatrix, get_cx_index(list_colormatrix, prm_qsv.common.out_vui.matrix));
    SetCXIndex(fcgCXColorPrim,   get_cx_index(list_colorprim,   prm_qsv.common.out_vui.colorprim));
    SetCXIndex(fcgCXVideoFormat, get_cx_index(list_videoformat, prm_qsv.common.out_vui.format));
    fcgCBFullrange->Checked      = prm_qsv.common.out_vui.colorrange == RGY_COLORRANGE_FULL;

    fcgCBOutputAud->Checked       = prm_qsv.bOutputAud != 0;
    fcgCBOutputPicStruct->Checked = prm_qsv.bOutputPicStruct != 0;

    //Vpp
        int denoise_idx = 0;
        if (prm_qsv.vpp.knn.enable) {
            denoise_idx = get_cx_index(list_vpp_denoise, _T("knn"));
        } else if (prm_qsv.vpp.pmd.enable) {
            denoise_idx = get_cx_index(list_vpp_denoise, _T("pmd"));
        } else if (prm_qsv.vpp.smooth.enable) {
            denoise_idx = get_cx_index(list_vpp_denoise, _T("smooth"));
        } else if (prm_qsv.vpp.dct.enable) {
            denoise_idx = get_cx_index(list_vpp_denoise, _T("denoise-dct"));
        } else if (prm_qsv.vpp.fft3d.enable) {
            denoise_idx = get_cx_index(list_vpp_denoise, _T("fft3d"));
        } else if (prm_qsv.vppmfx.denoise.enable) {
            denoise_idx = get_cx_index(list_vpp_denoise, _T("denoise"));
        } else if (prm_qsv.vpp.convolution3d.enable) {
            denoise_idx = get_cx_index(list_vpp_denoise, _T("convolution3d"));
        } else if (prm_qsv.vpp.nlmeans.enable) {
            denoise_idx = get_cx_index(list_vpp_denoise, _T("nlmeans"));
        }
        SetCXIndex(fcgCXVppDenoiseMethod, denoise_idx);

        int detail_enahance_idx = 0;
        if (prm_qsv.vpp.unsharp.enable) {
            detail_enahance_idx = get_cx_index(list_vpp_detail_enahance, _T("unsharp"));
        } else if (prm_qsv.vpp.edgelevel.enable) {
            detail_enahance_idx = get_cx_index(list_vpp_detail_enahance, _T("edgelevel"));
        } else if (prm_qsv.vpp.warpsharp.enable) {
            detail_enahance_idx = get_cx_index(list_vpp_detail_enahance, _T("warpsharp"));
        } else if (prm_qsv.vppmfx.detail.enable) {
            detail_enahance_idx = get_cx_index(list_vpp_detail_enahance, _T("detail-enhance"));
        }
        SetCXIndex(fcgCXVppDetailEnhance, detail_enahance_idx);

        int deinterlacer_idx = 0;
        if (prm_qsv.vpp.afs.enable) {
            deinterlacer_idx = get_cx_index(list_deinterlace_gui, L"自動フィールドシフト");
        } else if (prm_qsv.vpp.nnedi.enable) {
            deinterlacer_idx = get_cx_index(list_deinterlace_gui, L"nnedi");
        } else if (prm_qsv.vpp.yadif.enable) {
            deinterlacer_idx = get_cx_index(list_deinterlace_gui, L"yadif");
        } else if (prm_qsv.vpp.decomb.enable) {
            deinterlacer_idx = get_cx_index(list_deinterlace_gui, L"decomb");
        } else if (prm_qsv.vppmfx.deinterlace > 0) {
            deinterlacer_idx = get_cx_index(list_deinterlace_gui, prm_qsv.vppmfx.deinterlace);
        }
        SetCXIndex(fcgCXVppDeinterlace, deinterlacer_idx);
 
        int deband_idx = 0;
        if (prm_qsv.vpp.deband.enable) {
            deband_idx = get_cx_index(list_vpp_deband_names, _T("deband"));
        } else if (prm_qsv.vpp.libplacebo_deband.enable) {
            deband_idx = get_cx_index(list_vpp_deband_names, _T("libplacebo-deband"));
        }
        SetCXIndex(fcgCXVppDeband, deband_idx);

        SetNUValue(fcgNUVppDenoiseMFX, prm_qsv.vppmfx.denoise.strength);
        SetNUValue(fcgNUVppDenoiseKnnRadius, prm_qsv.vpp.knn.radius);
        SetNUValue(fcgNUVppDenoiseKnnStrength, prm_qsv.vpp.knn.strength);
        SetNUValue(fcgNUVppDenoiseKnnThreshold, prm_qsv.vpp.knn.lerp_threshold);
        SetCXIndex(fcgCXVppDenoiseNLMeansPatch, get_cx_index(list_vpp_nlmeans_block_size, (int)prm_qsv.vpp.nlmeans.patchSize));
        SetCXIndex(fcgCXVppDenoiseNLMeansSearch, get_cx_index(list_vpp_nlmeans_block_size, (int)prm_qsv.vpp.nlmeans.searchSize));
        SetNUValue(fcgNUVppDenoiseNLMeansSigma, prm_qsv.vpp.nlmeans.sigma);
        SetNUValue(fcgNUVppDenoiseNLMeansH, prm_qsv.vpp.nlmeans.h);
        SetNUValue(fcgNUVppDenoisePmdApplyCount, prm_qsv.vpp.pmd.applyCount);
        SetNUValue(fcgNUVppDenoisePmdStrength, prm_qsv.vpp.pmd.strength);
        SetNUValue(fcgNUVppDenoisePmdThreshold, prm_qsv.vpp.pmd.threshold);
        SetNUValue(fcgNUVppDenoiseSmoothQuality, prm_qsv.vpp.smooth.quality);
        SetNUValue(fcgNUVppDenoiseSmoothQP, prm_qsv.vpp.smooth.qp);
        SetCXIndex(fcgCXVppDenoiseDctStep, get_cx_index(list_vpp_denoise_dct_step_gui, (int)prm_qsv.vpp.dct.step));
        SetNUValue(fcgNUVppDenoiseDctSigma, prm_qsv.vpp.dct.sigma);
        SetCXIndex(fcgCXVppDenoiseDctBlockSize, get_cx_index(list_vpp_denoise_dct_block_size, (int)prm_qsv.vpp.dct.block_size));
        SetNUValue(fcgNUVppDenoiseFFT3DSigma, prm_qsv.vpp.fft3d.sigma);
        SetNUValue(fcgNUVppDenoiseFFT3DAmount, prm_qsv.vpp.fft3d.amount);
        SetCXIndex(fcgCXVppDenoiseFFT3DBlockSize, get_cx_index(list_vpp_fft3d_block_size, (int)prm_qsv.vpp.fft3d.block_size));
        SetNUValue(fcgNUVppDenoiseFFT3DOverlap, prm_qsv.vpp.fft3d.overlap);
        SetCXIndex(fcgCXVppDenoiseFFT3DTemporal, get_cx_index(list_vpp_fft3d_temporal_gui, (int)prm_qsv.vpp.fft3d.temporal));
        SetCXIndex(fcgCXVppDenoiseFFT3DPrecision, get_cx_index(list_vpp_fp_prec, (int)prm_qsv.vpp.fft3d.precision));

        SetCXIndex(fcgCXVppDenoiseConv3DMatrix, get_cx_index(list_vpp_convolution3d_matrix, (int)prm_qsv.vpp.convolution3d.matrix));
        SetNUValue(fcgNUVppDenoiseConv3DThreshYSpatial, prm_qsv.vpp.convolution3d.threshYspatial);
        SetNUValue(fcgNUVppDenoiseConv3DThreshCSpatial, prm_qsv.vpp.convolution3d.threshCspatial);
        SetNUValue(fcgNUVppDenoiseConv3DThreshYTemporal, prm_qsv.vpp.convolution3d.threshYtemporal);
        SetNUValue(fcgNUVppDenoiseConv3DThreshCTemporal, prm_qsv.vpp.convolution3d.threshCtemporal);
        SetNUValue(fcgNUVppDebandRange, prm_qsv.vpp.deband.range);
        SetNUValue(fcgNUVppDebandThreY, prm_qsv.vpp.deband.threY);
        SetNUValue(fcgNUVppDebandThreCb, prm_qsv.vpp.deband.threCb);
        SetNUValue(fcgNUVppDebandThreCr, prm_qsv.vpp.deband.threCr);
        SetNUValue(fcgNUVppDebandDitherY, prm_qsv.vpp.deband.ditherY);
        SetNUValue(fcgNUVppDebandDitherC, prm_qsv.vpp.deband.ditherC);
        SetCXIndex(fcgCXVppDebandSample, get_cx_index(list_vpp_deband_gui, prm_qsv.vpp.deband.sample));
        fcgCBVppDebandBlurFirst->Checked = prm_qsv.vpp.deband.blurFirst;
        fcgCBVppDebandRandEachFrame->Checked = prm_qsv.vpp.deband.randEachFrame;
        SetNUValue(fcgNUVppLibplaceboDebandIteration, prm_qsv.vpp.libplacebo_deband.iterations);
        SetNUValue(fcgNUVppLibplaceboDebandRadius,    prm_qsv.vpp.libplacebo_deband.radius);
        SetNUValue(fcgNUVppLibplaceboDebandThreshold, prm_qsv.vpp.libplacebo_deband.threshold);
        SetNUValue(fcgNUVppLibplaceboDebandGrainY,    prm_qsv.vpp.libplacebo_deband.grainY);
        SetNUValue(fcgNUVppLibplaceboDebandGrainC,    prm_qsv.vpp.libplacebo_deband.grainC >= 0.0f ? prm_qsv.vpp.libplacebo_deband.grainC : prm_qsv.vpp.libplacebo_deband.grainY);
        SetCXIndex(fcgCXVppLibplaceboDebandDither,    get_cx_index(list_vpp_libplacebo_deband_dither_mode, (int)prm_qsv.vpp.libplacebo_deband.dither));
        SetCXIndex(fcgCXVppLibplaceboDebandLUTSize,   get_cx_index(list_vpp_libplacebo_deband_lut_size, prm_qsv.vpp.libplacebo_deband.lut_size));
        SetNUValue(fcgNUVppUnsharpRadius, prm_qsv.vpp.unsharp.radius);
        SetNUValue(fcgNUVppUnsharpWeight, prm_qsv.vpp.unsharp.weight);
        SetNUValue(fcgNUVppUnsharpThreshold, prm_qsv.vpp.unsharp.threshold);
        SetNUValue(fcgNUVppEdgelevelStrength, prm_qsv.vpp.edgelevel.strength);
        SetNUValue(fcgNUVppEdgelevelThreshold, prm_qsv.vpp.edgelevel.threshold);
        SetNUValue(fcgNUVppEdgelevelBlack, prm_qsv.vpp.edgelevel.black);
        SetNUValue(fcgNUVppEdgelevelWhite, prm_qsv.vpp.edgelevel.white);
        SetNUValue(fcgNUVppWarpsharpBlur, prm_qsv.vpp.warpsharp.blur);
        SetNUValue(fcgNUVppWarpsharpThreshold, prm_qsv.vpp.warpsharp.threshold);
        SetNUValue(fcgNUVppWarpsharpType, prm_qsv.vpp.warpsharp.type);
        SetNUValue(fcgNUVppWarpsharpDepth, prm_qsv.vpp.warpsharp.depth);
        SetNUValue(fcgNUVppDetailEnhanceMFX, prm_qsv.vppmfx.detail.strength);

        SetNUValue(fcgNUVppAfsUp, prm_qsv.vpp.afs.clip.top);
        SetNUValue(fcgNUVppAfsBottom, prm_qsv.vpp.afs.clip.bottom);
        SetNUValue(fcgNUVppAfsLeft, prm_qsv.vpp.afs.clip.left);
        SetNUValue(fcgNUVppAfsRight, prm_qsv.vpp.afs.clip.right);
        SetNUValue(fcgNUVppAfsMethodSwitch, prm_qsv.vpp.afs.method_switch);
        SetNUValue(fcgNUVppAfsCoeffShift, prm_qsv.vpp.afs.coeff_shift);
        SetNUValue(fcgNUVppAfsThreShift, prm_qsv.vpp.afs.thre_shift);
        SetNUValue(fcgNUVppAfsThreDeint, prm_qsv.vpp.afs.thre_deint);
        SetNUValue(fcgNUVppAfsThreYMotion, prm_qsv.vpp.afs.thre_Ymotion);
        SetNUValue(fcgNUVppAfsThreCMotion, prm_qsv.vpp.afs.thre_Cmotion);
        SetCXIndex(fcgCXVppAfsAnalyze, prm_qsv.vpp.afs.analyze);
        fcgCBVppAfsShift->Checked = prm_qsv.vpp.afs.shift != 0;
        fcgCBVppAfsDrop->Checked = prm_qsv.vpp.afs.drop != 0;
        fcgCBVppAfsSmooth->Checked = prm_qsv.vpp.afs.smooth != 0;
        fcgCBVppAfs24fps->Checked = prm_qsv.vpp.afs.force24 != 0;
        fcgCBVppAfsTune->Checked = prm_qsv.vpp.afs.tune != 0;
        SetCXIndex(fcgCXVppNnediNsize, get_cx_index(list_vpp_nnedi_nsize, prm_qsv.vpp.nnedi.nsize));
        SetCXIndex(fcgCXVppNnediNns, get_cx_index(list_vpp_nnedi_nns, prm_qsv.vpp.nnedi.nns));
        SetCXIndex(fcgCXVppNnediPrec, get_cx_index(list_vpp_fp_prec, prm_qsv.vpp.nnedi.precision));
        SetCXIndex(fcgCXVppNnediPrescreen, get_cx_index(list_vpp_nnedi_pre_screen_gui, prm_qsv.vpp.nnedi.pre_screen));
        SetCXIndex(fcgCXVppNnediQual, get_cx_index(list_vpp_nnedi_quality, prm_qsv.vpp.nnedi.quality));
        SetCXIndex(fcgCXVppNnediErrorType, get_cx_index(list_vpp_nnedi_error_type, prm_qsv.vpp.nnedi.errortype));
        SetCXIndex(fcgCXVppYadifMode, get_cx_index(list_vpp_yadif_mode_gui, prm_qsv.vpp.yadif.mode));
        fcgCBVppDecombFull->Checked = prm_qsv.vpp.decomb.full != 0;
        fcgCBVppDecombBlend->Checked = prm_qsv.vpp.decomb.blend != 0;
        SetNUValue(fcgNUVppDecombThreshold, prm_qsv.vpp.decomb.threshold);
        SetNUValue(fcgNUVppDecombDthreshold, prm_qsv.vpp.decomb.dthreshold);

        //fcgCBSSIM->Checked = prm_qsv.ssim;
        //fcgCBPSNR->Checked = prm_qsv.psnr;

        SetCXIndex(fcgCXImageStabilizer, prm_qsv.vppmfx.imageStabilizer);
        SetCXIndex(fcgCXRotate, get_cx_index(list_rotate_angle_ja, prm_qsv.vpp.transform.rotate()));

        SetNUValue(fcgNUVppMctf, (prm_qsv.vppmfx.mctf.enable) ? prm_qsv.vppmfx.mctf.strength : 0);
        SetCXIndex(fcgCXVppResizeAlg, get_cx_index(list_vpp_resize, prm_qsv.vpp.resize_algo));
        fcgCBVppResize->Checked = cnf->vid.resize_enable;
        SetNUValue(fcgNUResizeW, cnf->vid.resize_width);
        SetNUValue(fcgNUResizeH, cnf->vid.resize_height);

        fcgCBSsim->Checked = prm_qsv.common.metric.ssim;
        fcgCBPsnr->Checked = prm_qsv.common.metric.psnr;
        fcgCBAvoidIdleClock->Checked = prm_qsv.ctrl.avoidIdleClock.mode != RGYParamAvoidIdleClockMode::Disabled;

        //SetCXIndex(fcgCXX264Priority,        cnf->vid.priority);
        const bool enable_tc2mp4_muxer = (0 != str_has_char(sys_dat->exstg->s_mux[MUXER_TC2MP4].base_cmd));
        SetCXIndex(fcgCXTempDir,             cnf->oth.temp_dir);
        SetNUValue(fcgNUInputBufSize,        prm_qsv.nInputBufSize);
        fcgCBAFS->Checked                  = (enable_tc2mp4_muxer) ? cnf->vid.afs != 0 : false;
        fcgCBAuoTcfileout->Checked         = (enable_tc2mp4_muxer) ? cnf->vid.auo_tcfile_out != 0 : false;

        //音声
        fcgCBAudioUseExt->Checked          = cnf->aud.use_internal == 0;
        //外部音声エンコーダ
        fcgCBAudioOnly->Checked            = cnf->oth.out_audio_only != 0;
        fcgCBFAWCheck->Checked             = cnf->aud.ext.faw_check != 0;
        SetCXIndex(fcgCXAudioEncoder,        cnf->aud.ext.encoder);
        fcgCBAudio2pass->Checked           = cnf->aud.ext.use_2pass != 0;
        fcgCBAudioUsePipe->Checked = (CurrentPipeEnabled && !cnf->aud.ext.use_wav);
        SetCXIndex(fcgCXAudioDelayCut,       cnf->aud.ext.delay_cut);
        SetCXIndex(fcgCXAudioEncMode,        cnf->aud.ext.enc_mode);
        SetNUValue(fcgNUAudioBitrate,       (cnf->aud.ext.bitrate != 0) ? cnf->aud.ext.bitrate : GetCurrentAudioDefaultBitrate());
        SetCXIndex(fcgCXAudioPriority,       cnf->aud.ext.priority);
        SetCXIndex(fcgCXAudioTempDir,        cnf->aud.ext.aud_temp_dir);
        SetCXIndex(fcgCXAudioEncTiming,      cnf->aud.ext.audio_encode_timing);
        fcgCBRunBatBeforeAudio->Checked    =(cnf->oth.run_bat & RUN_BAT_BEFORE_AUDIO) != 0;
        fcgCBRunBatAfterAudio->Checked     =(cnf->oth.run_bat & RUN_BAT_AFTER_AUDIO) != 0;
        fcgTXBatBeforeAudioPath->Text      = String(cnf->oth.batfile.before_audio).ToString();
        fcgTXBatAfterAudioPath->Text       = String(cnf->oth.batfile.after_audio).ToString();
        //内蔵音声エンコーダ
        SetCXIndex(fcgCXAudioEncoderInternal, cnf->aud.in.encoder);
        SetCXIndex(fcgCXAudioEncModeInternal, cnf->aud.in.enc_mode);
        SetNUValue(fcgNUAudioBitrateInternal, (cnf->aud.in.bitrate != 0) ? cnf->aud.in.bitrate : GetCurrentAudioDefaultBitrate());

        //mux
        fcgCBMP4MuxerExt->Checked          = cnf->mux.disable_mp4ext == 0;
        fcgCBMP4MuxApple->Checked          = cnf->mux.apple_mode != 0;
        SetCXIndex(fcgCXMP4CmdEx,            cnf->mux.mp4_mode);
        SetCXIndex(fcgCXMP4BoxTempDir,       cnf->mux.mp4_temp_dir);
        fcgCBMKVMuxerExt->Checked          = cnf->mux.disable_mkvext == 0;
        SetCXIndex(fcgCXMKVCmdEx,            cnf->mux.mkv_mode);
        fcgCBMuxMinimize->Checked          = cnf->mux.minimized != 0;
        SetCXIndex(fcgCXMuxPriority,         cnf->mux.priority);
        SetCXIndex(fcgCXInternalCmdEx,       cnf->mux.internal_mode);

        fcgCBRunBatBefore->Checked         =(cnf->oth.run_bat & RUN_BAT_BEFORE_PROCESS) != 0;
        fcgCBRunBatAfter->Checked          =(cnf->oth.run_bat & RUN_BAT_AFTER_PROCESS)  != 0;
        fcgCBWaitForBatBefore->Checked     =(cnf->oth.dont_wait_bat_fin & RUN_BAT_BEFORE_PROCESS) == 0;
        fcgCBWaitForBatAfter->Checked      =(cnf->oth.dont_wait_bat_fin & RUN_BAT_AFTER_PROCESS)  == 0;
        fcgTXBatBeforePath->Text           = String(cnf->oth.batfile.before_process).ToString();
        fcgTXBatAfterPath->Text            = String(cnf->oth.batfile.after_process).ToString();

        SetfcgTSLSettingsNotes(cnf->oth.notes);

    this->ResumeLayout();
    this->PerformLayout();
}

System::String^ frmConfig::FrmToConf(CONF_GUIEX *cnf) {
    //これもひたすら書くだけ。めんどい
    sInputParams prm_qsv;

    prm_qsv.codec                  = get_out_enc_codec_by_index(fcgCXEncCodec->SelectedIndex);
    cnf->enc.codec_rgy             = prm_qsv.codec;
    prm_qsv.device                 = (featuresHW) ? (featuresHW->devCount() > 1 ? featuresHW->getDevID(fcgCXDevice->SelectedIndex) : QSVDeviceNum::AUTO) : QSVDeviceNum::AUTO;
    prm_qsv.rcParam.encMode        = (int)list_encmode[fcgCXEncMode->SelectedIndex].value;
    prm_qsv.hyperMode              = (mfxHyperMode)list_hyper_mode[fcgCXHyperMode->SelectedIndex].value;
    prm_qsv.nTargetUsage           = (int)list_quality[fcgCXQualityPreset->SelectedIndex].value;
    prm_qsv.CodecProfile           = (int)get_profile_list(prm_qsv.codec)[fcgCXCodecProfile->SelectedIndex].value;
    prm_qsv.CodecLevel             = (int)get_level_list(prm_qsv.codec)[fcgCXCodecLevel->SelectedIndex].value;
    prm_qsv.outputCsp              = (RGY_CHROMAFMT)list_output_csp[fcgCXOutputCsp->SelectedIndex].value;
    prm_qsv.outputDepth            = get_bit_depth(fcgCXBitDepth->SelectedIndex);
    prm_qsv.rcParam.bitrate        = (int)fcgNUBitrate->Value;
    prm_qsv.rcParam.maxBitrate     = (int)fcgNUMaxkbps->Value;
    prm_qsv.nLookaheadDepth        = (int)fcgNULookaheadDepth->Value;
    prm_qsv.nRef                   = (int)fcgNURef->Value;
    prm_qsv.bopenGOP               = fcgCBOpenGOP->Checked;
    prm_qsv.nGOPLength             = (int)fcgNUGopLength->Value;
    prm_qsv.rcParam.qp.qpI         = (int)fcgNUQPI->Value;
    prm_qsv.rcParam.qp.qpP         = (int)fcgNUQPP->Value;
    prm_qsv.rcParam.qp.qpB         = (int)fcgNUQPB->Value;
    prm_qsv.rcParam.icqQuality     = (int)fcgNUICQQuality->Value;
    prm_qsv.rcParam.qvbrQuality    = (int)fcgNUQVBR->Value;
    if (gopRefDistAsBframe(prm_qsv.codec)) {
        prm_qsv.GopRefDist = (int)fcgNUBframes->Value + 1;
    } else {
        prm_qsv.GopRefDist = (int)fcgNUBframes->Value;
    }
    prm_qsv.nTrellis               = (int)list_avc_trellis[fcgCXTrellis->SelectedIndex].value;
    prm_qsv.input.picstruct        = (RGY_PICSTRUCT)list_interlaced[fcgCXInterlaced->SelectedIndex].value;
    prm_qsv.bAdaptiveI             = fcgCBAdaptiveI->Checked;
    prm_qsv.bAdaptiveB             = fcgCBAdaptiveB->Checked;
    prm_qsv.nWeightP               = (int)(fcgCBWeightP->Checked    ? MFX_WEIGHTED_PRED_DEFAULT : MFX_WEIGHTED_PRED_UNKNOWN);
    prm_qsv.nWeightB               = (int)(fcgCBWeightB->Checked    ? MFX_WEIGHTED_PRED_DEFAULT : MFX_WEIGHTED_PRED_UNKNOWN);
    prm_qsv.nFadeDetect            = (int)(fcgCBFadeDetect->Checked ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_UNKNOWN);
    prm_qsv.bBPyramid              = fcgCBBPyramid->Checked;
    prm_qsv.nLookaheadDS           = (int)list_lookahead_ds[fcgCXLookaheadDS->SelectedIndex].value;
    prm_qsv.bMBBRC                 = fcgCBMBBRC->Checked;
    //prm_qsv.bExtBRC                = fcgCBExtBRC->Checked;
    prm_qsv.nWinBRCSize            = (int)fcgNUWinBRCSize->Value;
    prm_qsv.functionMode           = (QSVFunctionMode)list_qsv_function_mode[fcgCXFunctionMode->SelectedIndex].value;
    prm_qsv.memType                = (fcgCBD3DMemAlloc->Checked) ? HW_MEMORY : SYSTEM_MEMORY;
    prm_qsv.rcParam.avbrAccuarcy    = (int)(fcgNUAVBRAccuarcy->Value * 10);
    prm_qsv.rcParam.avbrConvergence = (int)fcgNUAVBRConvergence->Value;
    prm_qsv.scenarioInfo           = (int)list_scenario_info[fcgCXScenarioInfo->SelectedIndex].value;
    prm_qsv.nSlices                = (int)fcgNUSlices->Value;
    prm_qsv.qpMin                  = RGYQPSet((int)fcgNUQPMin->Value, (int)fcgNUQPMin->Value, (int)fcgNUQPMin->Value);
    prm_qsv.qpMax                  = RGYQPSet((int)fcgNUQPMax->Value, (int)fcgNUQPMax->Value, (int)fcgNUQPMax->Value);

    prm_qsv.nBluray                = fcgCBBlurayCompat->Checked;

    prm_qsv.bNoDeblock             = !fcgCBDeblock->Checked;
    prm_qsv.intraRefreshCycle      = (int)fcgNUIntraRefreshCycle->Value;

    prm_qsv.bCAVLC                 = !fcgCBCABAC->Checked;
    prm_qsv.bRDO                   = fcgCBRDO->Checked;
    prm_qsv.MVSearchWindow.first   = (int)fcgNUMVSearchWindow->Value;
    prm_qsv.MVSearchWindow.second  = (int)fcgNUMVSearchWindow->Value;
    prm_qsv.nMVPrecision           = (int)list_mv_presicion[fcgCXMVPred->SelectedIndex].value;
    prm_qsv.nInterPred             = (int)list_pred_block_size[fcgCXInterPred->SelectedIndex].value;
    prm_qsv.nIntraPred             = (int)list_pred_block_size[fcgCXIntraPred->SelectedIndex].value;

    prm_qsv.bDirectBiasAdjust      = fcgCBDirectBiasAdjust->Checked;
    prm_qsv.bGlobalMotionAdjust    = list_mv_cost_scaling[fcgCXMVCostScaling->SelectedIndex].value > 0;
    prm_qsv.nMVCostScaling         = (int)((prm_qsv.bGlobalMotionAdjust) ? list_mv_cost_scaling[fcgCXMVCostScaling->SelectedIndex].value : 0);

    prm_qsv.common.out_vui.matrix    = (CspMatrix)list_colormatrix[fcgCXColorMatrix->SelectedIndex].value;
    prm_qsv.common.out_vui.colorprim = (CspColorprim)list_colorprim[fcgCXColorPrim->SelectedIndex].value;
    prm_qsv.common.out_vui.transfer  = (CspTransfer)list_transfer[fcgCXTransfer->SelectedIndex].value;
    prm_qsv.common.out_vui.format    = list_videoformat[fcgCXVideoFormat->SelectedIndex].value;
    prm_qsv.common.out_vui.colorrange = fcgCBFullrange->Checked ? RGY_COLORRANGE_FULL : RGY_COLORRANGE_UNSPECIFIED;
    prm_qsv.common.out_vui.descriptpresent = 1;

    prm_qsv.input.srcHeight        = 0;
    prm_qsv.input.srcWidth         = 0;
    prm_qsv.input.fpsN             = 0;
    prm_qsv.input.fpsD             = 0;

    prm_qsv.nPAR[0]                = (int)fcgNUAspectRatioX->Value;
    prm_qsv.nPAR[1]                = (int)fcgNUAspectRatioY->Value;
    if (fcgCXAspectRatio->SelectedIndex == 1) {
        prm_qsv.nPAR[0] *= -1;
        prm_qsv.nPAR[1] *= -1;
    }

    prm_qsv.bOutputAud              = fcgCBOutputAud->Checked;
    prm_qsv.bOutputPicStruct        = fcgCBOutputPicStruct->Checked;

    //vpp

    prm_qsv.vpp.resize_algo                 = (RGY_VPP_RESIZE_ALGO)list_vpp_resize[fcgCXVppResizeAlg->SelectedIndex].value;

    prm_qsv.vpp.knn.enable = fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("knn"));
    prm_qsv.vpp.knn.radius = (int)fcgNUVppDenoiseKnnRadius->Value;
    prm_qsv.vpp.knn.strength = (float)fcgNUVppDenoiseKnnStrength->Value;
    prm_qsv.vpp.knn.lerp_threshold = (float)fcgNUVppDenoiseKnnThreshold->Value;

    prm_qsv.vpp.nlmeans.enable = fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("nlmeans"));
    prm_qsv.vpp.nlmeans.patchSize = list_vpp_nlmeans_block_size[fcgCXVppDenoiseNLMeansPatch->SelectedIndex].value;
    prm_qsv.vpp.nlmeans.searchSize = list_vpp_nlmeans_block_size[fcgCXVppDenoiseNLMeansSearch->SelectedIndex].value;
    prm_qsv.vpp.nlmeans.sigma = (float)fcgNUVppDenoiseNLMeansSigma->Value;
    prm_qsv.vpp.nlmeans.h = (float)fcgNUVppDenoiseNLMeansH->Value;

    prm_qsv.vpp.pmd.enable = fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("pmd"));
    prm_qsv.vpp.pmd.applyCount = (int)fcgNUVppDenoisePmdApplyCount->Value;
    prm_qsv.vpp.pmd.strength = (float)fcgNUVppDenoisePmdStrength->Value;
    prm_qsv.vpp.pmd.threshold = (float)fcgNUVppDenoisePmdThreshold->Value;

    prm_qsv.vpp.smooth.enable = fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("smooth"));
    prm_qsv.vpp.smooth.quality = (int)fcgNUVppDenoiseSmoothQuality->Value;
    prm_qsv.vpp.smooth.qp = (int)fcgNUVppDenoiseSmoothQP->Value;

    prm_qsv.vpp.dct.enable = fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("denoise-dct"));
    prm_qsv.vpp.dct.step = list_vpp_denoise_dct_step[fcgCXVppDenoiseDctStep->SelectedIndex].value;
    prm_qsv.vpp.dct.sigma = (float)fcgNUVppDenoiseDctSigma->Value;
    prm_qsv.vpp.dct.block_size = list_vpp_denoise_dct_block_size[fcgCXVppDenoiseDctBlockSize->SelectedIndex].value;

    prm_qsv.vpp.fft3d.enable = fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("fft3d"));
    prm_qsv.vpp.fft3d.sigma = (float)fcgNUVppDenoiseFFT3DSigma->Value;
    prm_qsv.vpp.fft3d.amount = (float)fcgNUVppDenoiseFFT3DAmount->Value;
    prm_qsv.vpp.fft3d.block_size = list_vpp_fft3d_block_size[fcgCXVppDenoiseFFT3DBlockSize->SelectedIndex].value;
    prm_qsv.vpp.fft3d.overlap = (float)fcgNUVppDenoiseFFT3DOverlap->Value;
    prm_qsv.vpp.fft3d.temporal = list_vpp_fft3d_temporal_gui[fcgCXVppDenoiseFFT3DTemporal->SelectedIndex].value;
    prm_qsv.vpp.fft3d.precision = (VppFpPrecision)list_vpp_fp_prec[fcgCXVppDenoiseFFT3DPrecision->SelectedIndex].value;

    prm_qsv.vppmfx.denoise.enable = fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("denoise"));
    prm_qsv.vppmfx.denoise.strength = (int)fcgNUVppDenoiseMFX->Value;

    prm_qsv.vpp.unsharp.enable = fcgCXVppDetailEnhance->SelectedIndex == get_cx_index(list_vpp_detail_enahance, _T("unsharp"));
    prm_qsv.vpp.unsharp.radius = (int)fcgNUVppUnsharpRadius->Value;
    prm_qsv.vpp.unsharp.weight = (float)fcgNUVppUnsharpWeight->Value;
    prm_qsv.vpp.unsharp.threshold = (float)fcgNUVppUnsharpThreshold->Value;

    prm_qsv.vpp.edgelevel.enable = fcgCXVppDetailEnhance->SelectedIndex == get_cx_index(list_vpp_detail_enahance, _T("edgelevel"));
    prm_qsv.vpp.edgelevel.strength = (float)fcgNUVppEdgelevelStrength->Value;
    prm_qsv.vpp.edgelevel.threshold = (float)fcgNUVppEdgelevelThreshold->Value;
    prm_qsv.vpp.edgelevel.black = (float)fcgNUVppEdgelevelBlack->Value;
    prm_qsv.vpp.edgelevel.white = (float)fcgNUVppEdgelevelWhite->Value;

    prm_qsv.vpp.warpsharp.enable = fcgCXVppDetailEnhance->SelectedIndex == get_cx_index(list_vpp_detail_enahance, _T("warpsharp"));
    prm_qsv.vpp.warpsharp.blur = (int)fcgNUVppWarpsharpBlur->Value;
    prm_qsv.vpp.warpsharp.threshold = (float)fcgNUVppWarpsharpThreshold->Value;
    prm_qsv.vpp.warpsharp.type = (int)fcgNUVppWarpsharpType->Value;
    prm_qsv.vpp.warpsharp.depth = (float)fcgNUVppWarpsharpDepth->Value;

    prm_qsv.vppmfx.detail.enable = fcgCXVppDetailEnhance->SelectedIndex == get_cx_index(list_vpp_detail_enahance, _T("detail-enhance"));
    prm_qsv.vppmfx.detail.strength = (int)fcgNUVppDetailEnhanceMFX->Value;

    prm_qsv.vpp.deband.enable = fcgCXVppDeband->SelectedIndex == get_cx_index(list_vpp_deband_names, _T("deband"));
    prm_qsv.vpp.deband.range = (int)fcgNUVppDebandRange->Value;
    prm_qsv.vpp.deband.threY = (int)fcgNUVppDebandThreY->Value;
    prm_qsv.vpp.deband.threCb = (int)fcgNUVppDebandThreCb->Value;
    prm_qsv.vpp.deband.threCr = (int)fcgNUVppDebandThreCr->Value;
    prm_qsv.vpp.deband.ditherY = (int)fcgNUVppDebandDitherY->Value;
    prm_qsv.vpp.deband.ditherC = (int)fcgNUVppDebandDitherC->Value;
    prm_qsv.vpp.deband.sample = list_vpp_deband_gui[fcgCXVppDebandSample->SelectedIndex].value;
    prm_qsv.vpp.deband.blurFirst = fcgCBVppDebandBlurFirst->Checked;
    prm_qsv.vpp.deband.randEachFrame = fcgCBVppDebandRandEachFrame->Checked;
 
    prm_qsv.vpp.libplacebo_deband.enable     = fcgCXVppDeband->SelectedIndex == get_cx_index(list_vpp_deband_names, _T("libplacebo-deband"));
    prm_qsv.vpp.libplacebo_deband.iterations = (int)fcgNUVppLibplaceboDebandIteration->Value;
    prm_qsv.vpp.libplacebo_deband.radius     = (float)fcgNUVppLibplaceboDebandRadius->Value;
    prm_qsv.vpp.libplacebo_deband.threshold  = (float)fcgNUVppLibplaceboDebandThreshold->Value;
    prm_qsv.vpp.libplacebo_deband.grainY     = (float)fcgNUVppLibplaceboDebandGrainY->Value;
    prm_qsv.vpp.libplacebo_deband.grainC     = (float)fcgNUVppLibplaceboDebandGrainC->Value;
    prm_qsv.vpp.libplacebo_deband.dither     = (VppLibplaceboDebandDitherMode)list_vpp_libplacebo_deband_dither_mode[fcgCXVppLibplaceboDebandDither->SelectedIndex].value;
    prm_qsv.vpp.libplacebo_deband.lut_size   = list_vpp_libplacebo_deband_lut_size[fcgCXVppLibplaceboDebandLUTSize->SelectedIndex].value;

    prm_qsv.vpp.afs.enable             = (fcgCXVppDeinterlace->SelectedIndex == get_cx_index(list_deinterlace_gui, L"自動フィールドシフト"));
    prm_qsv.vpp.afs.timecode           = false;
    prm_qsv.vpp.afs.clip.top           = (int)fcgNUVppAfsUp->Value;
    prm_qsv.vpp.afs.clip.bottom        = (int)fcgNUVppAfsBottom->Value;
    prm_qsv.vpp.afs.clip.left          = (int)fcgNUVppAfsLeft->Value;
    prm_qsv.vpp.afs.clip.right         = (int)fcgNUVppAfsRight->Value;
    prm_qsv.vpp.afs.method_switch      = (int)fcgNUVppAfsMethodSwitch->Value;
    prm_qsv.vpp.afs.coeff_shift        = (int)fcgNUVppAfsCoeffShift->Value;
    prm_qsv.vpp.afs.thre_shift         = (int)fcgNUVppAfsThreShift->Value;
    prm_qsv.vpp.afs.thre_deint         = (int)fcgNUVppAfsThreDeint->Value;
    prm_qsv.vpp.afs.thre_Ymotion       = (int)fcgNUVppAfsThreYMotion->Value;
    prm_qsv.vpp.afs.thre_Cmotion       = (int)fcgNUVppAfsThreCMotion->Value;
    prm_qsv.vpp.afs.analyze            = fcgCXVppAfsAnalyze->SelectedIndex;
    prm_qsv.vpp.afs.shift              = fcgCBVppAfsShift->Checked;
    prm_qsv.vpp.afs.drop               = fcgCBVppAfsDrop->Checked;
    prm_qsv.vpp.afs.smooth             = fcgCBVppAfsSmooth->Checked;
    prm_qsv.vpp.afs.force24            = fcgCBVppAfs24fps->Checked;
    prm_qsv.vpp.afs.tune               = fcgCBVppAfsTune->Checked;

    prm_qsv.vpp.nnedi.enable           = (fcgCXVppDeinterlace->SelectedIndex == get_cx_index(list_deinterlace_gui, L"nnedi"));
    prm_qsv.vpp.nnedi.nsize            = (VppNnediNSize)list_vpp_nnedi_nsize[fcgCXVppNnediNsize->SelectedIndex].value;
    prm_qsv.vpp.nnedi.nns              = list_vpp_nnedi_nns[fcgCXVppNnediNns->SelectedIndex].value;
    prm_qsv.vpp.nnedi.quality          = (VppNnediQuality)list_vpp_nnedi_quality[fcgCXVppNnediQual->SelectedIndex].value;
    prm_qsv.vpp.nnedi.precision        = (VppFpPrecision)list_vpp_fp_prec[fcgCXVppNnediPrec->SelectedIndex].value;
    prm_qsv.vpp.nnedi.pre_screen       = (VppNnediPreScreen)list_vpp_nnedi_pre_screen_gui[fcgCXVppNnediPrescreen->SelectedIndex].value;
    prm_qsv.vpp.nnedi.errortype        = (VppNnediErrorType)list_vpp_nnedi_error_type[fcgCXVppNnediErrorType->SelectedIndex].value;

    prm_qsv.vpp.yadif.enable           = (fcgCXVppDeinterlace->SelectedIndex == get_cx_index(list_deinterlace_gui, L"yadif"));
    prm_qsv.vpp.yadif.mode             = (VppYadifMode)list_vpp_yadif_mode_gui[fcgCXVppYadifMode->SelectedIndex].value;

    prm_qsv.vpp.decomb.enable          = (fcgCXVppDeinterlace->SelectedIndex == get_cx_index(list_deinterlace_gui, L"decomb"));
    prm_qsv.vpp.decomb.full            = fcgCBVppDecombFull->Checked;
    prm_qsv.vpp.decomb.blend           = fcgCBVppDecombBlend->Checked;
    prm_qsv.vpp.decomb.threshold       = (int)fcgNUVppDecombThreshold->Value;
    prm_qsv.vpp.decomb.dthreshold      = (int)fcgNUVppDecombDthreshold->Value;

    if (!prm_qsv.vpp.afs.enable
        && !prm_qsv.vpp.nnedi.enable
        && !prm_qsv.vpp.yadif.enable
        && !prm_qsv.vpp.decomb.enable
    ) {
        prm_qsv.vppmfx.deinterlace = list_deinterlace_gui[fcgCXVppDeinterlace->SelectedIndex].value;
    }

    //prm_qsv.ssim                       = fcgCBSSIM->Checked;
    //prm_qsv.psnr                       = fcgCBPSNR->Checked;

    prm_qsv.vppmfx.mctf.enable = (int)fcgNUVppMctf->Value > 0;
    prm_qsv.vppmfx.mctf.strength = (int)fcgNUVppMctf->Value;
    prm_qsv.vppmfx.imageStabilizer = (int)list_vpp_image_stabilizer[fcgCXImageStabilizer->SelectedIndex].value;
    prm_qsv.vpp.transform.setRotate((int)list_rotate_angle_ja[fcgCXRotate->SelectedIndex].value);

    prm_qsv.vpp.resize_algo = (RGY_VPP_RESIZE_ALGO)list_vpp_resize[fcgCXVppResizeAlg->SelectedIndex].value;
    cnf->vid.resize_enable = fcgCBVppResize->Checked;
    cnf->vid.resize_width = (int)fcgNUResizeW->Value;
    cnf->vid.resize_height = (int)fcgNUResizeH->Value;
    if (cnf->vid.resize_enable) {
        prm_qsv.input.dstWidth = cnf->vid.resize_width;
        prm_qsv.input.dstHeight = cnf->vid.resize_height;
    } else {
        prm_qsv.input.dstWidth = 0;
        prm_qsv.input.dstHeight = 0;
    }

    prm_qsv.common.metric.ssim     = fcgCBSsim->Checked;
    prm_qsv.common.metric.psnr     = fcgCBPsnr->Checked;
    prm_qsv.ctrl.avoidIdleClock.mode = fcgCBAvoidIdleClock->Checked ? RGYParamAvoidIdleClockMode::Auto : RGYParamAvoidIdleClockMode::Disabled;

    //拡張部
    const bool enable_tc2mp4_muxer = (0 != str_has_char(sys_dat->exstg->s_mux[MUXER_TC2MP4].base_cmd));
    cnf->oth.temp_dir               = fcgCXTempDir->SelectedIndex;
    prm_qsv.nInputBufSize          = (int)fcgNUInputBufSize->Value;
    cnf->vid.auo_tcfile_out         = (enable_tc2mp4_muxer) ? fcgCBAuoTcfileout->Checked : false;
    cnf->vid.afs                    = (enable_tc2mp4_muxer) ? fcgCBAFS->Checked : false;

    //音声部
    cnf->oth.out_audio_only             = fcgCBAudioOnly->Checked;
    cnf->aud.use_internal               = !fcgCBAudioUseExt->Checked;
    cnf->aud.ext.encoder                = fcgCXAudioEncoder->SelectedIndex;
    cnf->aud.ext.faw_check              = fcgCBFAWCheck->Checked;
    cnf->aud.ext.enc_mode               = fcgCXAudioEncMode->SelectedIndex;
    cnf->aud.ext.bitrate                = (int)fcgNUAudioBitrate->Value;
    cnf->aud.ext.use_2pass              = fcgCBAudio2pass->Checked;
    cnf->aud.ext.use_wav                = !fcgCBAudioUsePipe->Checked;
    cnf->aud.ext.delay_cut              = fcgCXAudioDelayCut->SelectedIndex;
    cnf->aud.ext.priority               = fcgCXAudioPriority->SelectedIndex;
    cnf->aud.ext.audio_encode_timing    = fcgCXAudioEncTiming->SelectedIndex;
    cnf->aud.ext.aud_temp_dir           = fcgCXAudioTempDir->SelectedIndex;
    cnf->aud.in.encoder                 = fcgCXAudioEncoderInternal->SelectedIndex;
    cnf->aud.in.faw_check               = fcgCBFAWCheck->Checked;
    cnf->aud.in.enc_mode                = fcgCXAudioEncModeInternal->SelectedIndex;
    cnf->aud.in.bitrate                 = (int)fcgNUAudioBitrateInternal->Value;

    //mux部
    cnf->mux.use_internal           = !fcgCBAudioUseExt->Checked;
    cnf->mux.disable_mp4ext         = !fcgCBMP4MuxerExt->Checked;
    cnf->mux.apple_mode             = fcgCBMP4MuxApple->Checked;
    cnf->mux.mp4_mode               = fcgCXMP4CmdEx->SelectedIndex;
    cnf->mux.mp4_temp_dir           = fcgCXMP4BoxTempDir->SelectedIndex;
    cnf->mux.disable_mkvext         = !fcgCBMKVMuxerExt->Checked;
    cnf->mux.mkv_mode               = fcgCXMKVCmdEx->SelectedIndex;
    cnf->mux.minimized              = fcgCBMuxMinimize->Checked;
    cnf->mux.priority               = fcgCXMuxPriority->SelectedIndex;
    cnf->mux.internal_mode          = fcgCXInternalCmdEx->SelectedIndex;

    cnf->oth.run_bat                = RUN_BAT_NONE;
    cnf->oth.run_bat               |= (fcgCBRunBatBeforeAudio->Checked) ? RUN_BAT_BEFORE_AUDIO   : NULL;
    cnf->oth.run_bat               |= (fcgCBRunBatAfterAudio->Checked)  ? RUN_BAT_AFTER_AUDIO    : NULL;
    cnf->oth.run_bat               |= (fcgCBRunBatBefore->Checked)      ? RUN_BAT_BEFORE_PROCESS : NULL;
    cnf->oth.run_bat               |= (fcgCBRunBatAfter->Checked)       ? RUN_BAT_AFTER_PROCESS  : NULL;
    cnf->oth.dont_wait_bat_fin      = RUN_BAT_NONE;
    cnf->oth.dont_wait_bat_fin     |= (!fcgCBWaitForBatBefore->Checked) ? RUN_BAT_BEFORE_PROCESS : NULL;
    cnf->oth.dont_wait_bat_fin     |= (!fcgCBWaitForBatAfter->Checked)  ? RUN_BAT_AFTER_PROCESS  : NULL;
    GetCHARfromString(cnf->oth.batfile.before_process, sizeof(cnf->oth.batfile.before_process), fcgTXBatBeforePath->Text);
    GetCHARfromString(cnf->oth.batfile.after_process,  sizeof(cnf->oth.batfile.after_process),  fcgTXBatAfterPath->Text);
    GetCHARfromString(cnf->oth.batfile.before_audio, sizeof(cnf->oth.batfile.before_audio), fcgTXBatBeforeAudioPath->Text);
    GetCHARfromString(cnf->oth.batfile.after_audio,  sizeof(cnf->oth.batfile.after_audio),  fcgTXBatAfterAudioPath->Text);

    GetfcgTSLSettingsNotes(cnf->oth.notes, sizeof(cnf->oth.notes));
    strcpy_s(cnf->enc.cmd, gen_cmd(&prm_qsv, true).c_str());

    return String(gen_cmd(&prm_qsv, false).c_str()).ToString();
}

System::Void frmConfig::GetfcgTSLSettingsNotes(char *notes, int nSize) {
    ZeroMemory(notes, nSize);
    if (fcgTSLSettingsNotes->Overflow != ToolStripItemOverflow::Never)
        GetCHARfromString(notes, nSize, fcgTSLSettingsNotes->Text);
}

System::Void frmConfig::SetfcgTSLSettingsNotes(const char *notes) {
    if (str_has_char(notes)) {
        fcgTSLSettingsNotes->ForeColor = Color::FromArgb(StgNotesColor[0][0], StgNotesColor[0][1], StgNotesColor[0][2]);
        fcgTSLSettingsNotes->Text = String(notes).ToString();
        fcgTSLSettingsNotes->Overflow = ToolStripItemOverflow::AsNeeded;
    } else {
        fcgTSLSettingsNotes->ForeColor = Color::FromArgb(StgNotesColor[1][0], StgNotesColor[1][1], StgNotesColor[1][2]);
        fcgTSLSettingsNotes->Text = LOAD_CLI_STRING(AuofcgTSTSettingsNotes);
        fcgTSLSettingsNotes->Overflow = ToolStripItemOverflow::Never;
    }
}

System::Void frmConfig::SetfcgTSLSettingsNotes(String^ notes) {
    if (notes->Length && fcgTSLSettingsNotes->Overflow != ToolStripItemOverflow::Never) {
        fcgTSLSettingsNotes->ForeColor = Color::FromArgb(StgNotesColor[0][0], StgNotesColor[0][1], StgNotesColor[0][2]);
        fcgTSLSettingsNotes->Text = notes;
        fcgTSLSettingsNotes->Overflow = ToolStripItemOverflow::AsNeeded;
    } else {
        fcgTSLSettingsNotes->ForeColor = Color::FromArgb(StgNotesColor[1][0], StgNotesColor[1][1], StgNotesColor[1][2]);
        fcgTSLSettingsNotes->Text = LOAD_CLI_STRING(AuofcgTSTSettingsNotes);
        fcgTSLSettingsNotes->Overflow = ToolStripItemOverflow::Never;
    }
}

System::Void frmConfig::SetChangedEvent(Control^ control, System::EventHandler^ _event) {
    System::Type^ ControlType = control->GetType();
    if (ControlType == NumericUpDown::typeid)
        ((NumericUpDown^)control)->ValueChanged += _event;
    else if (ControlType == ComboBox::typeid)
        ((ComboBox^)control)->SelectedIndexChanged += _event;
    else if (ControlType == CheckBox::typeid)
        ((CheckBox^)control)->CheckedChanged += _event;
    else if (ControlType == TextBox::typeid)
        ((TextBox^)control)->TextChanged += _event;
}

System::Void frmConfig::SetToolStripEvents(ToolStrip^ TS, System::Windows::Forms::MouseEventHandler^ _event) {
    for (int i = 0; i < TS->Items->Count; i++) {
        ToolStripButton^ TSB = dynamic_cast<ToolStripButton^>(TS->Items[i]);
        if (TSB != nullptr) TSB->MouseDown += _event;
    }
}

System::Void frmConfig::TabControl_DarkDrawItem(System::Object^ sender, DrawItemEventArgs^ e) {
    //対象のTabControlを取得
    TabControl^ tab = dynamic_cast<TabControl^>(sender);
    //タブページのテキストを取得
    System::String^ txt = tab->TabPages[e->Index]->Text;

    //タブのテキストと背景を描画するためのブラシを決定する
    SolidBrush^ foreBrush = gcnew System::Drawing::SolidBrush(ColorfromInt(DEFAULT_UI_COLOR_TEXT_DARK));
    SolidBrush^ backBrush = gcnew System::Drawing::SolidBrush(ColorfromInt(DEFAULT_UI_COLOR_BASE_DARK));

    //StringFormatを作成
    StringFormat^ sf = gcnew System::Drawing::StringFormat();
    //中央に表示する
    sf->Alignment = StringAlignment::Center;
    sf->LineAlignment = StringAlignment::Center;

    //背景の描画
    e->Graphics->FillRectangle(backBrush, e->Bounds);
    //Textの描画
    e->Graphics->DrawString(txt, e->Font, foreBrush, e->Bounds, sf);
}

System::Void frmConfig::fcgMouseEnter_SetColor(System::Object^  sender, System::EventArgs^  e) {
    fcgMouseEnterLeave_SetColor(sender, themeMode, DarkenWindowState::Hot, dwStgReader);
}
System::Void frmConfig::fcgMouseLeave_SetColor(System::Object^  sender, System::EventArgs^  e) {
    fcgMouseEnterLeave_SetColor(sender, themeMode, DarkenWindowState::Normal, dwStgReader);
}

System::Void frmConfig::SetAllMouseMove(Control ^top, const AuoTheme themeTo) {
    if (themeTo == themeMode) return;
    System::Type^ type = top->GetType();
    if (type == CheckBox::typeid /* || isToolStripItem(type)*/) {
        top->MouseEnter += gcnew System::EventHandler(this, &frmConfig::fcgMouseEnter_SetColor);
        top->MouseLeave += gcnew System::EventHandler(this, &frmConfig::fcgMouseLeave_SetColor);
    } else if (type == ToolStrip::typeid) {
        ToolStrip^ TS = dynamic_cast<ToolStrip^>(top);
        for (int i = 0; i < TS->Items->Count; i++) {
            auto item = TS->Items[i];
            item->MouseEnter += gcnew System::EventHandler(this, &frmConfig::fcgMouseEnter_SetColor);
            item->MouseLeave += gcnew System::EventHandler(this, &frmConfig::fcgMouseLeave_SetColor);
        }
    }
    for (int i = 0; i < top->Controls->Count; i++) {
        SetAllMouseMove(top->Controls[i], themeTo);
    }
}

System::Void frmConfig::CheckTheme() {
    //DarkenWindowが使用されていれば設定をロードする
    if (dwStgReader != nullptr) delete dwStgReader;
    const auto [themeTo, dwStg] = check_current_theme(sys_dat->aviutl_dir);
    dwStgReader = dwStg;

    //変更の必要がなければ終了
    if (themeTo == themeMode) return;

    //一度ウィンドウの再描画を完全に抑止する
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 0, 0);
#if 0
    //tabcontrolのborderを隠す
    SwitchComboBoxBorder(fcgtabControlVideo, fcgPNHideTabControlVideo, themeMode, themeTo, dwStgReader);
    SwitchComboBoxBorder(fcgtabControlAudio, fcgPNHideTabControlAudio, themeMode, themeTo, dwStgReader);
    SwitchComboBoxBorder(fcgtabControlMux,   fcgPNHideTabControlMux,   themeMode, themeTo, dwStgReader);
#endif
    //上部のtoolstripborderを隠すためのパネル
    fcgPNHideToolStripBorder->Visible = themeTo == AuoTheme::DarkenWindowDark;
#if 0
    //TabControlをオーナードローする
    fcgtabControlVideo->DrawMode = TabDrawMode::OwnerDrawFixed;
    fcgtabControlVideo->DrawItem += gcnew DrawItemEventHandler(this, &frmConfig::TabControl_DarkDrawItem);

    fcgtabControlAudio->DrawMode = TabDrawMode::OwnerDrawFixed;
    fcgtabControlAudio->DrawItem += gcnew DrawItemEventHandler(this, &frmConfig::TabControl_DarkDrawItem);

    fcgtabControlMux->DrawMode = TabDrawMode::OwnerDrawFixed;
    fcgtabControlMux->DrawItem += gcnew DrawItemEventHandler(this, &frmConfig::TabControl_DarkDrawItem);
#endif
    if (themeTo != themeMode) {
        SetAllColor(this, themeTo, this->GetType(), dwStgReader);
        SetAllMouseMove(this, themeTo);
    }
    fcgSetDataGridViewCellStyleHeader(fcgDGVFeatures, themeMode, dwStgReader);
    //一度ウィンドウの再描画を再開し、強制的に再描画させる
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 1, 0);
    this->Refresh();
    themeMode = themeTo;
}

System::Void frmConfig::SetAllCheckChangedEvents(Control ^top) {
    //再帰を使用してすべてのコントロールのtagを調べ、イベントをセットする
    for (int i = 0; i < top->Controls->Count; i++) {
        System::Type^ type = top->Controls[i]->GetType();
        if (type == NumericUpDown::typeid)
            top->Controls[i]->Enter += gcnew System::EventHandler(this, &frmConfig::NUSelectAll);

        if (type == Label::typeid || type == Button::typeid)
            ;
        else if (type == ToolStrip::typeid)
            SetToolStripEvents((ToolStrip^)(top->Controls[i]), gcnew System::Windows::Forms::MouseEventHandler(this, &frmConfig::fcgTSItem_MouseDown));
        else if (top->Controls[i]->Tag == nullptr)
            SetAllCheckChangedEvents(top->Controls[i]);
        else if (String::Equals(top->Controls[i]->Tag->ToString(), L"reCmd"))
            SetChangedEvent(top->Controls[i], gcnew System::EventHandler(this, &frmConfig::fcgRebuildCmd));
        else if (top->Controls[i]->Tag->ToString()->Contains(L"chValue"))
            SetChangedEvent(top->Controls[i], gcnew System::EventHandler(this, &frmConfig::CheckOtherChanges));
        else
            SetAllCheckChangedEvents(top->Controls[i]);
    }
}

System::Void frmConfig::SetHelpToolTipsColorMatrix(Control^ control, const CX_DESC *list, const wchar_t *type) {
    fcgTTEx->SetToolTip(control, L"--" + String(type).ToString() + L"\n"
        + LOAD_CLI_STRING(AuofrmTTColorMatrix1) + L"\n"
        + LOAD_CLI_STRING(AuofrmTTColorMatrix2) + L"\n"
        + LOAD_CLI_STRING(AuofrmTTColorMatrix3) + L" " + HD_HEIGHT_THRESHOLD + L" " + LOAD_CLI_STRING(AuofrmTTColorMatrix4) + L" … " + String(COLOR_VALUE_AUTO_HD_NAME).ToString() + L"\n"
        + LOAD_CLI_STRING(AuofrmTTColorMatrix3) + L" " + HD_HEIGHT_THRESHOLD + L" " + LOAD_CLI_STRING(AuofrmTTColorMatrix5) + L" … " + String(COLOR_VALUE_AUTO_SD_NAME).ToString() + L"\n"
        + LOAD_CLI_STRING(AuofrmTTColorMatrix6)
    );
}

System::Void frmConfig::SetHelpToolTips() {

#define SET_TOOL_TIP_EX2(target, x) { fcgTTEx->SetToolTip(target, LOAD_CLI_STRING(AuofrmTT ## x)); }
#define SET_TOOL_TIP_EX(target) { fcgTTEx->SetToolTip(target, LOAD_CLI_STRING(AuofrmTT ## target)); }
#define SET_TOOL_TIP_EX_AUD_INTERNAL(target) { fcgTTEx->SetToolTip(target ## Internal, LOAD_CLI_STRING(AuofrmTT ## target)); }

    SET_TOOL_TIP_EX(fcgTXVideoEncoderPath);
    SET_TOOL_TIP_EX(fcgCXDevice);
    SET_TOOL_TIP_EX(fcgCXEncCodec);
    SET_TOOL_TIP_EX(fcgCXQualityPreset);
    SET_TOOL_TIP_EX(fcgCXHyperMode);
    SET_TOOL_TIP_EX(fcgCXEncMode);
    SET_TOOL_TIP_EX(fcgNUQPI);
    SET_TOOL_TIP_EX(fcgNUQPP);
    SET_TOOL_TIP_EX(fcgNUQPB);
    SET_TOOL_TIP_EX(fcgNUICQQuality);
    SET_TOOL_TIP_EX(fcgNUAVBRConvergence);
    SET_TOOL_TIP_EX(fcgNUAVBRAccuarcy);
    SET_TOOL_TIP_EX(fcgNUMaxkbps);
    SET_TOOL_TIP_EX(fcgNULookaheadDepth);
    SET_TOOL_TIP_EX(fcgNUGopLength);
    SET_TOOL_TIP_EX(fcgNURef);
    SET_TOOL_TIP_EX(fcgNUBframes);
    SET_TOOL_TIP_EX(fcgCBAdaptiveI);
    SET_TOOL_TIP_EX(fcgCBAdaptiveB);
    SET_TOOL_TIP_EX(fcgCBWeightP);
    SET_TOOL_TIP_EX(fcgCBWeightB);
    SET_TOOL_TIP_EX(fcgCBFadeDetect);
    SET_TOOL_TIP_EX(fcgCBOpenGOP);
    SET_TOOL_TIP_EX(fcgCBBPyramid);
    SET_TOOL_TIP_EX(fcgCXLookaheadDS);
    SET_TOOL_TIP_EX(fcgNUWinBRCSize);
    SET_TOOL_TIP_EX(fcgNUQPMin);
    SET_TOOL_TIP_EX(fcgNUQPMax);
    SET_TOOL_TIP_EX(fcgCBBlurayCompat);
    SET_TOOL_TIP_EX(fcgCXInterlaced);
    SET_TOOL_TIP_EX(fcgCXCodecProfile);
    SET_TOOL_TIP_EX(fcgCXCodecLevel);
    SET_TOOL_TIP_EX(fcgNUSlices);
    SET_TOOL_TIP_EX(fcgCXVideoFormat);
    SET_TOOL_TIP_EX(fcgCBFullrange);
    SET_TOOL_TIP_EX(fcgNUInputBufSize);
    SET_TOOL_TIP_EX(fcgCBD3DMemAlloc);
    SET_TOOL_TIP_EX(fcgCBSsim);
    SET_TOOL_TIP_EX(fcgCBPsnr);
    SET_TOOL_TIP_EX(fcgCBAvoidIdleClock);
    SET_TOOL_TIP_EX(fcgCBOutputAud);
    SET_TOOL_TIP_EX(fcgCBOutputPicStruct);
    SET_TOOL_TIP_EX(fcgCBDeblock);
    SET_TOOL_TIP_EX(fcgNUIntraRefreshCycle);
    SET_TOOL_TIP_EX(fcgCBDirectBiasAdjust);
    SET_TOOL_TIP_EX(fcgCXMVCostScaling);
    SET_TOOL_TIP_EX(fcgCXTrellis);
    SET_TOOL_TIP_EX(fcgCBMBBRC);
    SET_TOOL_TIP_EX(fcgCBExtBRC);
    SET_TOOL_TIP_EX(fcgCXIntraPred);
    SET_TOOL_TIP_EX(fcgCXInterPred);
    SET_TOOL_TIP_EX(fcgNUMVSearchWindow);
    SET_TOOL_TIP_EX(fcgCXMVPred);
    SET_TOOL_TIP_EX(fcgCBCABAC);
    SET_TOOL_TIP_EX(fcgCBRDO);

    SET_TOOL_TIP_EX(fcgCXOutputCsp);

    SetHelpToolTipsColorMatrix(fcgCXColorMatrix, list_colormatrix, L"colormatrix");
    SetHelpToolTipsColorMatrix(fcgCXColorPrim, list_colorprim, L"colorprim");
    SetHelpToolTipsColorMatrix(fcgCXTransfer, list_transfer, L"transfer");

    fcgTTEx->SetToolTip(fcgCXAspectRatio, L""
        + LOAD_CLI_STRING(aspect_desc[0].mes) + L"\n"
        + L"   " + LOAD_CLI_STRING(AuofrmTTfcgCXAspectRatioSAR) + L"\n"
        + L"\n"
        + LOAD_CLI_STRING(aspect_desc[1].mes) + L"\n"
        + L"   " + LOAD_CLI_STRING(AuofrmTTfcgCXAspectRatioDAR) + L"\n"
    );
    SET_TOOL_TIP_EX(fcgNUAspectRatioX);
    SET_TOOL_TIP_EX(fcgNUAspectRatioY);

    //フィルタ
    SET_TOOL_TIP_EX(fcgCBVppResize);
    SET_TOOL_TIP_EX(fcgCXVppResizeAlg);
    SET_TOOL_TIP_EX(fcgCXVppDenoiseMethod);
    SET_TOOL_TIP_EX(fcgNUVppDenoisePmdApplyCount);
    SET_TOOL_TIP_EX(fcgNUVppDenoisePmdStrength);
    SET_TOOL_TIP_EX(fcgNUVppDenoisePmdThreshold);
    SET_TOOL_TIP_EX(fcgCXVppDenoiseConv3DMatrix);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseConv3DThreshYSpatial);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseConv3DThreshCSpatial);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseConv3DThreshYTemporal);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseConv3DThreshCTemporal);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseSmoothQuality);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseSmoothQP);
    SET_TOOL_TIP_EX(fcgCXVppDenoiseDctStep);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseDctSigma);
    SET_TOOL_TIP_EX(fcgCXVppDenoiseDctBlockSize);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseFFT3DSigma);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseFFT3DAmount);
    SET_TOOL_TIP_EX(fcgCXVppDenoiseFFT3DBlockSize);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseFFT3DOverlap);
    SET_TOOL_TIP_EX(fcgCXVppDenoiseFFT3DTemporal);
    SET_TOOL_TIP_EX(fcgCXVppDenoiseFFT3DPrecision);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseKnnRadius);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseKnnStrength);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseKnnThreshold);
    SET_TOOL_TIP_EX(fcgCXVppDenoiseNLMeansPatch);
    SET_TOOL_TIP_EX(fcgCXVppDenoiseNLMeansSearch);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseNLMeansSigma);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseNLMeansH);
    SET_TOOL_TIP_EX(fcgCXVppDetailEnhance);
    SET_TOOL_TIP_EX(fcgNUVppWarpsharpBlur);
    SET_TOOL_TIP_EX(fcgNUVppWarpsharpThreshold);
    SET_TOOL_TIP_EX(fcgNUVppWarpsharpType);
    SET_TOOL_TIP_EX(fcgNUVppWarpsharpDepth);
    SET_TOOL_TIP_EX(fcgNUVppEdgelevelStrength);
    SET_TOOL_TIP_EX(fcgNUVppEdgelevelThreshold);
    SET_TOOL_TIP_EX(fcgNUVppEdgelevelBlack);
    SET_TOOL_TIP_EX(fcgNUVppEdgelevelWhite);
    SET_TOOL_TIP_EX(fcgNUVppUnsharpRadius);
    SET_TOOL_TIP_EX(fcgNUVppUnsharpThreshold);
    SET_TOOL_TIP_EX(fcgNUVppUnsharpWeight);
    SET_TOOL_TIP_EX(fcggroupBoxVppDeband);
    SET_TOOL_TIP_EX(fcgNUVppDebandRange);
    SET_TOOL_TIP_EX(fcgNUVppDebandThreY);
    SET_TOOL_TIP_EX(fcgNUVppDebandThreCb);
    SET_TOOL_TIP_EX(fcgNUVppDebandThreCr);
    SET_TOOL_TIP_EX(fcgNUVppDebandDitherY);
    SET_TOOL_TIP_EX(fcgNUVppDebandDitherC);
    SET_TOOL_TIP_EX(fcgCXVppDebandSample);
    SET_TOOL_TIP_EX(fcgCBVppDebandBlurFirst);
    SET_TOOL_TIP_EX(fcgCBVppDebandRandEachFrame);
    SET_TOOL_TIP_EX(fcgNUVppLibplaceboDebandIteration);
    SET_TOOL_TIP_EX(fcgNUVppLibplaceboDebandRadius);
    SET_TOOL_TIP_EX(fcgNUVppLibplaceboDebandThreshold);
    SET_TOOL_TIP_EX(fcgNUVppLibplaceboDebandGrainY);
    SET_TOOL_TIP_EX(fcgNUVppLibplaceboDebandGrainC);
    SET_TOOL_TIP_EX(fcgCXVppLibplaceboDebandDither);
    SET_TOOL_TIP_EX(fcgCXVppLibplaceboDebandLUTSize);
    SET_TOOL_TIP_EX(fcgCXVppDeinterlace);
    SET_TOOL_TIP_EX(fcgNUVppAfsUp);
    SET_TOOL_TIP_EX(fcgNUVppAfsBottom);
    SET_TOOL_TIP_EX(fcgNUVppAfsLeft);
    SET_TOOL_TIP_EX(fcgNUVppAfsRight);
    SET_TOOL_TIP_EX(fcgNUVppAfsMethodSwitch);
    SET_TOOL_TIP_EX(fcgTBVppAfsMethodSwitch);
    SET_TOOL_TIP_EX(fcgNUVppAfsCoeffShift);
    SET_TOOL_TIP_EX(fcgTBVppAfsCoeffShift);
    SET_TOOL_TIP_EX(fcgNUVppAfsThreShift);
    SET_TOOL_TIP_EX(fcgTBVppAfsThreShift);
    SET_TOOL_TIP_EX(fcgNUVppAfsThreDeint);
    SET_TOOL_TIP_EX(fcgTBVppAfsThreDeint);
    SET_TOOL_TIP_EX(fcgNUVppAfsThreYMotion);
    SET_TOOL_TIP_EX(fcgTBVppAfsThreYMotion);
    SET_TOOL_TIP_EX(fcgNUVppAfsThreCMotion);
    SET_TOOL_TIP_EX(fcgTBVppAfsThreCMotion);
    SET_TOOL_TIP_EX(fcgCXVppAfsAnalyze);
    SET_TOOL_TIP_EX(fcgCBVppAfsShift);
    SET_TOOL_TIP_EX(fcgCBVppAfs24fps);
    SET_TOOL_TIP_EX(fcgCBVppAfsDrop);
    SET_TOOL_TIP_EX(fcgCBVppAfsSmooth);
    SET_TOOL_TIP_EX(fcgCBVppAfsTune);
    SET_TOOL_TIP_EX(fcgCXVppYadifMode);
    SET_TOOL_TIP_EX(fcgCBVppDecombFull);
    SET_TOOL_TIP_EX(fcgCBVppDecombBlend);
    SET_TOOL_TIP_EX(fcgNUVppDecombThreshold);
    SET_TOOL_TIP_EX(fcgNUVppDecombDthreshold);
    SET_TOOL_TIP_EX(fcgCXVppNnediNns);
    SET_TOOL_TIP_EX(fcgCXVppNnediNsize);
    SET_TOOL_TIP_EX(fcgCXVppNnediQual);
    SET_TOOL_TIP_EX(fcgCXVppNnediPrec);
    SET_TOOL_TIP_EX(fcgCXVppNnediPrescreen);
    SET_TOOL_TIP_EX(fcgCXVppNnediErrorType);
    SET_TOOL_TIP_EX(fcgCXRotate);
    SET_TOOL_TIP_EX(fcgNUVppDetailEnhanceMFX);
    SET_TOOL_TIP_EX(fcgNUVppMctf);
    SET_TOOL_TIP_EX(fcgCXImageStabilizer);
    SET_TOOL_TIP_EX(fcgPNVppDenoiseMFX);

    //拡張
    SET_TOOL_TIP_EX(fcgCBAFS);
    SET_TOOL_TIP_EX(fcgCBAuoTcfileout);
    SET_TOOL_TIP_EX(fcgCXTempDir);
    SET_TOOL_TIP_EX(fcgBTCustomTempDir);

    //音声
    SET_TOOL_TIP_EX(fcgCBAudioUseExt);
    SET_TOOL_TIP_EX_AUD_INTERNAL(fcgCXAudioEncoder);
    SET_TOOL_TIP_EX_AUD_INTERNAL(fcgCXAudioEncMode);
    SET_TOOL_TIP_EX_AUD_INTERNAL(fcgNUAudioBitrate);
    SET_TOOL_TIP_EX(fcgCXAudioEncoder);
    SET_TOOL_TIP_EX(fcgCBAudioOnly);
    SET_TOOL_TIP_EX(fcgCBFAWCheck);
    SET_TOOL_TIP_EX(fcgBTAudioEncoderPath);
    SET_TOOL_TIP_EX(fcgCXAudioEncMode);
    SET_TOOL_TIP_EX(fcgCBAudio2pass);
    SET_TOOL_TIP_EX(fcgCBAudioUsePipe);
    SET_TOOL_TIP_EX(fcgNUAudioBitrate);
    SET_TOOL_TIP_EX(fcgCXAudioPriority);
    SET_TOOL_TIP_EX(fcgCXAudioEncTiming);
    SET_TOOL_TIP_EX(fcgCXAudioTempDir);
    SET_TOOL_TIP_EX(fcgBTCustomAudioTempDir);
    //音声バッチファイル実行
    SET_TOOL_TIP_EX(fcgCBRunBatBeforeAudio);
    SET_TOOL_TIP_EX(fcgCBRunBatAfterAudio);
    SET_TOOL_TIP_EX(fcgBTBatBeforeAudioPath);
    SET_TOOL_TIP_EX(fcgBTBatAfterAudioPath);
    //muxer
    SET_TOOL_TIP_EX(fcgCBMP4MuxerExt);
    SET_TOOL_TIP_EX(fcgCXMP4CmdEx);
    SET_TOOL_TIP_EX(fcgBTMP4MuxerPath);
    SET_TOOL_TIP_EX(fcgBTTC2MP4Path);
    SET_TOOL_TIP_EX(fcgBTMP4RawPath);
    SET_TOOL_TIP_EX(fcgCXMP4BoxTempDir);
    SET_TOOL_TIP_EX(fcgBTMP4BoxTempDir);
    SET_TOOL_TIP_EX(fcgCBMKVMuxerExt);
    SET_TOOL_TIP_EX(fcgCXMKVCmdEx);
    SET_TOOL_TIP_EX(fcgBTMKVMuxerPath);
    SET_TOOL_TIP_EX(fcgCXMuxPriority);
    //バッチファイル実行
    SET_TOOL_TIP_EX(fcgCBRunBatBefore);
    SET_TOOL_TIP_EX(fcgCBRunBatAfter);
    SET_TOOL_TIP_EX(fcgCBWaitForBatBefore);
    SET_TOOL_TIP_EX(fcgCBWaitForBatAfter);
    SET_TOOL_TIP_EX(fcgBTBatBeforePath);
    SET_TOOL_TIP_EX(fcgBTBatAfterPath);
    //上部ツールストリップ
    fcgTSBDelete->ToolTipText = LOAD_CLI_STRING(AuofrmTTfcgTSBDelete);
    fcgTSBOtherSettings->ToolTipText = LOAD_CLI_STRING(AuofrmTTfcgTSBOtherSettings);
    fcgTSBSave->ToolTipText = LOAD_CLI_STRING(AuofrmTTfcgTSBSave);
    fcgTSBSaveNew->ToolTipText = LOAD_CLI_STRING(AuofrmTTfcgTSBSaveNew);

    //他
    SET_TOOL_TIP_EX(fcgTXCmd);
    SET_TOOL_TIP_EX(fcgBTDefault);
}
System::Void frmConfig::ShowExehelp(String^ ExePath, String^ args) {
    if (!File::Exists(ExePath)) {
        MessageBox::Show(L"指定された実行ファイルが存在しません。", L"エラー", MessageBoxButtons::OK, MessageBoxIcon::Error);
    } else {
        char exe_path[MAX_PATH_LEN];
        char file_path[MAX_PATH_LEN];
        char cmd[MAX_CMD_LEN];
        GetCHARfromString(exe_path, sizeof(exe_path), ExePath);
        apply_appendix(file_path, _countof(file_path), exe_path, "_fullhelp.txt");
        File::Delete(String(file_path).ToString());
        array<String^>^ arg_list = args->Split(L';');
        for (int i = 0; i < arg_list->Length; i++) {
            if (i) {
                System::IO::StreamWriter^ sw;
                try {
                    sw = gcnew System::IO::StreamWriter(String(file_path).ToString(), true, System::Text::Encoding::GetEncoding("shift_jis"));
                    sw->WriteLine();
                    sw->WriteLine();
                } catch (...) {
                    //ファイルオープンに失敗…初回のget_exe_message_to_fileでエラーとなるため、おそらく起こらない
                } finally {
                    if (sw != nullptr) { sw->Close(); }
                }
            }
            GetCHARfromString(cmd, sizeof(cmd), arg_list[i]);
            if (get_exe_message_to_file(exe_path, cmd, file_path, AUO_PIPE_MUXED, 5) != RP_SUCCESS) {
                File::Delete(String(file_path).ToString());
                MessageBox::Show(L"helpの取得に失敗しました。", L"エラー", MessageBoxButtons::OK, MessageBoxIcon::Error);
                return;
            }
        }
        try {
            System::Diagnostics::Process::Start(String(file_path).ToString());
        } catch (...) {
            MessageBox::Show(L"helpを開く際に不明なエラーが発生しました。", L"エラー", MessageBoxButtons::OK, MessageBoxIcon::Error);
        }
    }
}

System::Void frmConfig::UpdateFeatures(bool reGenerateTable) {
    if (fcgCXEncCodec->SelectedIndex < 0 || _countof(list_out_enc_codec) == fcgCXEncCodec->SelectedIndex) {
        return;
    }
    if (featuresHW == nullptr) {
        return;
    }

    //表示更新
    const auto codec = get_out_enc_codec_by_index(fcgCXEncCodec->SelectedIndex);
    QSVFunctionMode funcMode = (QSVFunctionMode)list_qsv_function_mode[fcgCXFunctionMode->SelectedIndex].value;
    if (funcMode == QSVFunctionMode::Auto) {
        funcMode = featuresHW->getAutoSelectFunctionMode(fcgCXDevice->SelectedIndex, fcgCXEncMode->SelectedIndex, codec);
    }
    const mfxU32 currentLib = featuresHW->GetmfxLibVer();
    String^ gpuname = featuresHW->GetGPUName();
    const bool currentLibValid = 0 != check_lib_version(currentLib, MFX_LIB_VERSION_1_1.Version);
    String^ currentAPI = L"hw: ";
    currentAPI += (currentLibValid) ? L"API v" + ((currentLib>>16).ToString() + L"." + (currentLib & 0x0000ffff).ToString()) : L"-------";
    fcgLBFeaturesCurrentAPIVer->Text = currentAPI + L" / codec: " + String(list_out_enc_codec[fcgCXEncCodec->SelectedIndex].desc).ToString() + L" " + String(funcMode == QSVFunctionMode::FF ? L"FF" : L"PG").ToString();
    fcgLBGPUInfoOnFeatureTab->Text = gpuname;

    auto dataGridViewFont = gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, FontStyle::Regular, GraphicsUnit::Point, static_cast<Byte>(128));

    fcgDGVFeatures->ReadOnly = true;
    fcgDGVFeatures->AllowUserToAddRows = false;
    fcgDGVFeatures->AllowUserToResizeRows = false;
    fcgDGVFeatures->AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode::Fill;

    fcgDGVFeatures->DataSource = featuresHW->getFeatureTable(fcgCXDevice->SelectedIndex, codec, funcMode == QSVFunctionMode::FF, reGenerateTable);

    if (fcgDGVFeatures->Columns->Count > 0) {
        fcgDGVFeatures->Columns[0]->FillWeight = 240;
    }
    fcgDGVFeatures->DefaultCellStyle->Font = dataGridViewFont;
    fcgDGVFeatures->ColumnHeadersDefaultCellStyle->Font = dataGridViewFont;
    fcgDGVFeatures->RowHeadersDefaultCellStyle->Font = dataGridViewFont;
    fcgSetDataGridViewCellStyleHeader(fcgDGVFeatures, themeMode, dwStgReader);
    fcgCheckVppFeatures();
}

System::Void frmConfig::SaveQSVFeature() {
    //WinXPにおいて、OpenFileDialogはCurrentDirctoryを勝手に変更しやがるので、
    //一度保存し、あとから再適用する
    String^ CurrentDir = Directory::GetCurrentDirectory();

    if (nullptr == saveFileQSVFeautures) {
        saveFileQSVFeautures = gcnew SaveFileDialog();
        WCHAR aviutl_dir[MAX_PATH_LEN];
        get_aviutl_dir(aviutl_dir, _countof(aviutl_dir));
        saveFileQSVFeautures->InitialDirectory = String(aviutl_dir).ToString();
    }
    saveFileQSVFeautures->FileName = L"";

    //ofd->Filter = L"pngファイル(*.png)|*.png|txtファイル(*.txt)|*.txt|csvファイル(*.csv)|*.csv";
    saveFileQSVFeautures->Filter = L"pngファイル(*.png)|*.png";

    saveFileQSVFeautures->Title = L"保存するファイル名を入力してください";
    if (System::Windows::Forms::DialogResult::OK == saveFileQSVFeautures->ShowDialog()) {
        String^ SavePath = saveFileQSVFeautures->FileName;
        saveFileQSVFeautures->InitialDirectory = Path::GetDirectoryName(saveFileQSVFeautures->FileName);

        bool isImage = 0 == String::Compare(".png", Path::GetExtension(SavePath), true);

        if (isImage) {
            SaveQSVFeatureAsImg(SavePath);
        } else {
            //SaveQSVFeatureAsTxt(SavePath);
        }
    }

    Directory::SetCurrentDirectory(CurrentDir);
}

System::Void frmConfig::SaveQSVFeatureAsImg(String^ SavePath) {
    bool SaveAsBmp = 0 == String::Compare(".bmp", Path::GetExtension(SavePath), true);
    Bitmap^ bmp = gcnew Bitmap(tabPageFeatures->Width, tabPageFeatures->Height);
    try {
        tabPageFeatures->DrawToBitmap(bmp, System::Drawing::Rectangle(0, 0, tabPageFeatures->Width, tabPageFeatures->Height));
        bmp->Save(SavePath, (SaveAsBmp) ? System::Drawing::Imaging::ImageFormat::Bmp : System::Drawing::Imaging::ImageFormat::Png);
    } catch (...) {
        MessageBox::Show(L"画像の保存中にエラーが発生しました。", L"エラー", MessageBoxButtons::OK, MessageBoxIcon::Error);
    } finally {
        delete bmp;
    }
}
System::Void frmConfig::SaveQSVFeatureAsTxt(String^ SavePath) {

}
System::Void frmConfig::SetCPUInfo() {
    //CPU名
    if (nullptr == StrCPUInfo || StrCPUInfo->Length <= 0) {
        TCHAR cpu_info[256];
        getCPUInfo(cpu_info, _countof(cpu_info));
        StrCPUInfo = String(cpu_info).ToString()->Replace(L" CPU ", L" ");
    }
    if (this->InvokeRequired) {
        SetCPUInfoDelegate^ sl = gcnew SetCPUInfoDelegate(this, &frmConfig::SetCPUInfo);
        this->Invoke(sl);
    } else {
        fcgLBCPUInfoOnFeatureTab->Text = StrCPUInfo;
    }
}

#pragma warning( pop )
