// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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

#pragma once

#include <Windows.h>

#include "auo.h"
#include "auo_pipe.h"
#include "auo_conf.h"
#include "auo_settings.h"
#include "auo_system.h"
#include "auo_util.h"
#include "auo_clrutil.h"

#include "frmConfig_helper.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;
using namespace System::IO;


namespace QSVEnc {

    /// <summary>
    /// frmConfig の概要
    ///
    /// 警告: このクラスの名前を変更する場合、このクラスが依存するすべての .resx ファイルに関連付けられた
    ///          マネージ リソース コンパイラ ツールに対して 'Resource File Name' プロパティを
    ///          変更する必要があります。この変更を行わないと、
    ///          デザイナと、このフォームに関連付けられたローカライズ済みリソースとが、
    ///          正しく相互に利用できなくなります。
    /// </summary>
    public ref class frmConfig : public System::Windows::Forms::Form
    {
    public:
        frmConfig(CONF_GUIEX *_conf, const SYSTEM_DATA *_sys_dat)
        {
            //ライブラリのチェック
            InitData(_conf, _sys_dat);
            cnf_stgSelected = (CONF_GUIEX *)calloc(1, sizeof(CONF_GUIEX));
            conf_link_prm = (AUO_LINK_PARAM *)calloc(1, sizeof(AUO_LINK_PARAM));
            InitializeComponent();
            //
            //TODO: ここにコンストラクタ コードを追加します
            //
        }

    protected:
        /// <summary>
        /// 使用中のリソースをすべてクリーンアップします。
        /// </summary>
        ~frmConfig()
        {
            if (components)
            {
                delete components;
            }
            CloseBitrateCalc();
            if (cnf_stgSelected) free(cnf_stgSelected); cnf_stgSelected = nullptr;
            if (conf_link_prm) free(conf_link_prm); conf_link_prm = nullptr;
            if (featuresHW != nullptr) delete featuresHW;
            if (featuresSW != nullptr) delete featuresSW;
            if (nullptr != saveFileQSVFeautures)
                delete saveFileQSVFeautures;
        }




    private: System::Windows::Forms::ToolStrip^  fcgtoolStripSettings;

    private: System::Windows::Forms::TabControl^  fcgtabControlMux;
    private: System::Windows::Forms::TabPage^  fcgtabPageMP4;
    private: System::Windows::Forms::TabPage^  fcgtabPageMKV;
    private: System::Windows::Forms::TextBox^  fcgTXCmd;

    private: System::Windows::Forms::Button^  fcgBTCancel;

    private: System::Windows::Forms::Button^  fcgBTOK;
    private: System::Windows::Forms::Button^  fcgBTDefault;




    private: System::Windows::Forms::ToolStripButton^  fcgTSBSave;

    private: System::Windows::Forms::ToolStripButton^  fcgTSBSaveNew;

    private: System::Windows::Forms::ToolStripButton^  fcgTSBDelete;

    private: System::Windows::Forms::ToolStripSeparator^  fcgtoolStripSeparator1;
    private: System::Windows::Forms::ToolStripDropDownButton^  fcgTSSettings;





























































































































































































































































    private: System::Windows::Forms::TabPage^  fcgtabPageMux;












    private: System::Windows::Forms::ComboBox^  fcgCXMP4CmdEx;

    private: System::Windows::Forms::Label^  fcgLBMP4CmdEx;
    private: System::Windows::Forms::CheckBox^  fcgCBMP4MuxerExt;
    private: System::Windows::Forms::Button^  fcgBTMP4BoxTempDir;
    private: System::Windows::Forms::TextBox^  fcgTXMP4BoxTempDir;


    private: System::Windows::Forms::ComboBox^  fcgCXMP4BoxTempDir;
    private: System::Windows::Forms::Label^  fcgLBMP4BoxTempDir;
    private: System::Windows::Forms::Button^  fcgBTTC2MP4Path;
    private: System::Windows::Forms::TextBox^  fcgTXTC2MP4Path;
    private: System::Windows::Forms::Button^  fcgBTMP4MuxerPath;
    private: System::Windows::Forms::TextBox^  fcgTXMP4MuxerPath;

    private: System::Windows::Forms::Label^  fcgLBTC2MP4Path;
    private: System::Windows::Forms::Label^  fcgLBMP4MuxerPath;


    private: System::Windows::Forms::Button^  fcgBTMKVMuxerPath;

    private: System::Windows::Forms::TextBox^  fcgTXMKVMuxerPath;

    private: System::Windows::Forms::Label^  fcgLBMKVMuxerPath;
    private: System::Windows::Forms::ComboBox^  fcgCXMKVCmdEx;
    private: System::Windows::Forms::Label^  fcgLBMKVMuxerCmdEx;
    private: System::Windows::Forms::CheckBox^  fcgCBMKVMuxerExt;
    private: System::Windows::Forms::ComboBox^  fcgCXMuxPriority;
    private: System::Windows::Forms::Label^  fcgLBMuxPriority;

    private: System::Windows::Forms::Label^  fcgLBVersionDate;


    private: System::Windows::Forms::Label^  fcgLBVersion;
    private: System::Windows::Forms::FolderBrowserDialog^  fcgfolderBrowserTemp;
    private: System::Windows::Forms::OpenFileDialog^  fcgOpenFileDialog;










private: System::Windows::Forms::ToolTip^  fcgTTEx;






private: System::Windows::Forms::ToolStripSeparator^  toolStripSeparator2;
private: System::Windows::Forms::ToolStripButton^  fcgTSBOtherSettings;















































































































private: System::Windows::Forms::ToolStripButton^  fcgTSBBitrateCalc;
private: System::Windows::Forms::TabControl^  fcgtabControlQSV;
private: System::Windows::Forms::TabPage^  tabPageVideoEnc;

private: System::Windows::Forms::Label^  fcgLBAVBRConvergence;


private: System::Windows::Forms::NumericUpDown^  fcgNUAVBRConvergence;
private: System::Windows::Forms::Label^  fcgLBAVBRAccuarcy;



private: System::Windows::Forms::NumericUpDown^  fcgNUAVBRAccuarcy;


private: System::Windows::Forms::GroupBox^  fcgGroupBoxAspectRatio;
private: System::Windows::Forms::Label^  fcgLBAspectRatio;
private: System::Windows::Forms::NumericUpDown^  fcgNUAspectRatioY;
private: System::Windows::Forms::NumericUpDown^  fcgNUAspectRatioX;
private: System::Windows::Forms::ComboBox^  fcgCXAspectRatio;
private: System::Windows::Forms::ComboBox^  fcgCXInterlaced;
private: System::Windows::Forms::Label^  fcgLBInterlaced;
private: System::Windows::Forms::NumericUpDown^  fcgNUGopLength;
private: System::Windows::Forms::Label^  fcgLBGOPLength;
private: System::Windows::Forms::ComboBox^  fcgCXCodecLevel;
private: System::Windows::Forms::ComboBox^  fcgCXCodecProfile;
private: System::Windows::Forms::Label^  fcgLBCodecLevel;
private: System::Windows::Forms::Label^  fcgLBCodecProfile;
private: System::Windows::Forms::NumericUpDown^  fcgNUBframes;
private: System::Windows::Forms::NumericUpDown^  fcgNURef;
private: System::Windows::Forms::Label^  fcgLBBframes;
private: System::Windows::Forms::Label^  fcgLBRef;
private: System::Windows::Forms::Label^  fcgLBMaxBitrate2;
private: System::Windows::Forms::Label^  fcgLBEncMode;
private: System::Windows::Forms::ComboBox^  fcgCXEncMode;
private: System::Windows::Forms::Label^  fcgLBMaxkbps;
private: System::Windows::Forms::NumericUpDown^  fcgNUMaxkbps;
private: System::Windows::Forms::Label^  fcgLBQPB;
private: System::Windows::Forms::Label^  fcgLBQPP;
private: System::Windows::Forms::Label^  fcgLBQPI;

private: System::Windows::Forms::NumericUpDown^  fcgNUQPB;
private: System::Windows::Forms::NumericUpDown^  fcgNUQPP;
private: System::Windows::Forms::NumericUpDown^  fcgNUQPI;
private: System::Windows::Forms::Label^  fcgLBQuality;
private: System::Windows::Forms::Label^  fcgLBOutputType;
private: System::Windows::Forms::ComboBox^  fcgCXQuality;
private: System::Windows::Forms::Label^  fcgLBBitrate2;
private: System::Windows::Forms::NumericUpDown^  fcgNUBitrate;
private: System::Windows::Forms::Label^  fcgLBBitrate;
private: System::Windows::Forms::ComboBox^  fcgCXOutputType;
private: System::Windows::Forms::CheckBox^  fcgCBHWEncode;







private: System::Windows::Forms::Label^  fcgLBAVBRAccuarcy2;
private: System::Windows::Forms::Label^  fcgLBAVBRConvergence2;
private: System::Windows::Forms::CheckBox^  fcgCBSceneChange;
private: System::Windows::Forms::CheckBox^  fcgCBOpenGOP;
private: System::Windows::Forms::GroupBox^  fcggroupBoxColor;
private: System::Windows::Forms::CheckBox^  fcgCBFullrange;
private: System::Windows::Forms::ComboBox^  fcgCXTransfer;


private: System::Windows::Forms::ComboBox^  fcgCXColorPrim;

private: System::Windows::Forms::ComboBox^  fcgCXColorMatrix;

private: System::Windows::Forms::Label^  fcgLBTransfer;
private: System::Windows::Forms::Label^  fcgLBColorPrim;
private: System::Windows::Forms::Label^  fcgLBColorMatrix;
private: System::Windows::Forms::TabPage^  tabPageExOpt;
private: System::Windows::Forms::Label^  fcgLBTempDir;

private: System::Windows::Forms::Button^  fcgBTCustomTempDir;
private: System::Windows::Forms::TextBox^  fcgTXCustomTempDir;

private: System::Windows::Forms::ComboBox^  fcgCXTempDir;
private: System::Windows::Forms::ComboBox^  fcgCXVideoFormat;
private: System::Windows::Forms::Label^  fcgLBVideoFormat;
private: System::Windows::Forms::Label^  fcgLBGOPLengthAuto;
private: System::Windows::Forms::Label^  fcgLBBframesAuto;
private: System::Windows::Forms::Label^  fcgLBRefAuto;
private: System::Windows::Forms::TabPage^  tabPageVpp;
private: System::Windows::Forms::CheckBox^  fcgCBUseVpp;
private: System::Windows::Forms::GroupBox^  fcggroupBoxVpp;
private: System::Windows::Forms::Label^  fcgLBVppResize;

private: System::Windows::Forms::NumericUpDown^  fcgNUVppResizeW;
private: System::Windows::Forms::CheckBox^  fcgCBVppResize;
private: System::Windows::Forms::GroupBox^  fcggroupBoxVppResize;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppResizeH;
private: System::Windows::Forms::CheckBox^  fcgCBVppDenoise;
private: System::Windows::Forms::GroupBox^  fcggroupBoxVppDenoise;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDenoise;
private: System::Windows::Forms::Label^  fcgLBVppDenoise;
private: System::Windows::Forms::CheckBox^  fcgCBVppDetail;
private: System::Windows::Forms::GroupBox^  fcggroupBoxVppDetail;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDetail;
private: System::Windows::Forms::Label^  fcgLBDetail;








private: System::Windows::Forms::ComboBox^  fcgCXDeinterlace;
private: System::Windows::Forms::Label^  fcgLBDeinterlace;
private: System::Windows::Forms::Label^  fcgLBDeinterlaceDesc;
private: System::Windows::Forms::NumericUpDown^  fcgNUSlices;

private: System::Windows::Forms::Label^  fcgLBSlices;
private: System::Windows::Forms::Label^  fcgLBSlices2;
private: System::Windows::Forms::ToolStripLabel^  fcgTSLSettingsNotes;
private: System::Windows::Forms::ToolStripTextBox^  fcgTSTSettingsNotes;
private: System::Windows::Forms::TabPage^  fcgtabPageBat;
private: System::Windows::Forms::Button^  fcgBTBatAfterPath;

private: System::Windows::Forms::TextBox^  fcgTXBatAfterPath;

private: System::Windows::Forms::Label^  fcgLBBatAfterPath;

private: System::Windows::Forms::CheckBox^  fcgCBWaitForBatAfter;

private: System::Windows::Forms::CheckBox^  fcgCBRunBatAfter;

private: System::Windows::Forms::CheckBox^  fcgCBMP4MuxApple;
private: System::Windows::Forms::Label^  fcgLBInputBufSize;
private: System::Windows::Forms::NumericUpDown^  fcgNUInputBufSize;
private: System::Windows::Forms::Label^  fcgLBMFXLibDetection;
private: System::Windows::Forms::Label^  fcgLBMFXLibDetectionSwValue;
private: System::Windows::Forms::Label^  fcgLBMFXLibDetectionHwValue;
private: System::Windows::Forms::Label^  fcgLBMFXLibDetectionSwStatus;
private: System::Windows::Forms::Label^  fcgLBMFXLibDetectionHwStatus;


private: System::Windows::Forms::Panel^  fcgPNAVBR;

private: System::Windows::Forms::Panel^  fcgPNBitrate;
private: System::Windows::Forms::Panel^  fcgPNQP;













private: System::Windows::Forms::Button^  fcgBTMP4RawPath;

private: System::Windows::Forms::TextBox^  fcgTXMP4RawPath;
private: System::Windows::Forms::Label^  fcgLBMP4RawPath;





private: System::Windows::Forms::TabPage^  fcgtabPageMPG;
private: System::Windows::Forms::Button^  fcgBTMPGMuxerPath;

private: System::Windows::Forms::TextBox^  fcgTXMPGMuxerPath;

private: System::Windows::Forms::Label^  fcgLBMPGMuxerPath;

private: System::Windows::Forms::ComboBox^  fcgCXMPGCmdEx;

private: System::Windows::Forms::Label^  label3;
private: System::Windows::Forms::CheckBox^  fcgCBMPGMuxerExt;

private: System::Windows::Forms::ContextMenuStrip^  fcgCSExeFiles;
private: System::Windows::Forms::ToolStripMenuItem^  fcgTSExeFileshelp;


private: System::Windows::Forms::Label^  fcgLBBlurayCompat;
private: System::Windows::Forms::CheckBox^  fcgCBBlurayCompat;
private: System::Windows::Forms::Label^  fcgLBBatAfterString;

private: System::Windows::Forms::Label^  fcgLBBatBeforeString;
private: System::Windows::Forms::Panel^  fcgPNSeparator;
private: System::Windows::Forms::Button^  fcgBTBatBeforePath;
private: System::Windows::Forms::TextBox^  fcgTXBatBeforePath;
private: System::Windows::Forms::Label^  fcgLBBatBeforePath;
private: System::Windows::Forms::CheckBox^  fcgCBWaitForBatBefore;
private: System::Windows::Forms::CheckBox^  fcgCBRunBatBefore;
private: System::Windows::Forms::LinkLabel^  fcgLBguiExBlog;
private: System::Windows::Forms::Label^  fcgLBFullrange;
private: System::Windows::Forms::CheckBox^  fcgCBAuoTcfileout;
private: System::Windows::Forms::CheckBox^  fcgCBAFS;


private: System::Windows::Forms::Panel^  fcgPNExtSettings;
private: System::Windows::Forms::Label^  fcgLBInterPred;
private: System::Windows::Forms::Label^  fcgLBIntraPred;
private: System::Windows::Forms::ComboBox^  fcgCXIntraPred;
private: System::Windows::Forms::ComboBox^  fcgCXInterPred;
private: System::Windows::Forms::ComboBox^  fcgCXMVPred;
private: System::Windows::Forms::Label^  fcgLBMVPred;
private: System::Windows::Forms::Label^  fcgLBMVWindowSize;
private: System::Windows::Forms::NumericUpDown^  fcgNUMVSearchWindow;
private: System::Windows::Forms::Label^  fcgLBMVSearch;
private: System::Windows::Forms::CheckBox^  fcgCBRDO;
private: System::Windows::Forms::CheckBox^  fcgCBCABAC;
private: System::Windows::Forms::Panel^  fcgPNLookahead;


private: System::Windows::Forms::NumericUpDown^  fcgNULookaheadDepth;
private: System::Windows::Forms::Label^  fcgLBLookaheadDepth;


private: System::Windows::Forms::Label^  label1;



private: System::Windows::Forms::CheckBox^  fcgCBBPyramid;
private: System::Windows::Forms::CheckBox^  fcgCBAdaptiveB;
private: System::Windows::Forms::CheckBox^  fcgCBAdaptiveI;
private: System::Windows::Forms::ComboBox^  fcgCXLookaheadDS;
private: System::Windows::Forms::Label^  fcgLBLookaheadDS;
private: System::Windows::Forms::Panel^  fcgPNICQ;

private: System::Windows::Forms::NumericUpDown^  fcgNUICQQuality;
private: System::Windows::Forms::Label^  fcgLBICQQuality;
private: System::Windows::Forms::TabPage^  tabPageFeatures;
private: System::Windows::Forms::DataGridView^  fcgDGVFeatures;
private: System::Windows::Forms::Label^  fcgLBFeaturesCurrentAPIVer;
private: System::Windows::Forms::Label^  fcgLBFeaturesShowCurrentAPI;

private: System::Windows::Forms::Button^  fcgBTSaveFeatureList;
private: System::Windows::Forms::Label^  fcgLBCPUInfoOnFeatureTab;
private: System::Windows::Forms::Label^  fcgLBCPUInfoLabelOnFeatureTab;


private: System::Windows::Forms::Label^  fcgLBGPUInfoOnFeatureTab;
private: System::Windows::Forms::Label^  fcgLBGPUInfoLabelOnFeatureTab;
private: System::Windows::Forms::Panel^  fcgPNQVBR;
private: System::Windows::Forms::NumericUpDown^  fcgNUQVBR;
private: System::Windows::Forms::Label^  fcgLBQVBR;
private: System::Windows::Forms::Label^  fcgLBQPMinMaxAuto;

private: System::Windows::Forms::NumericUpDown^  fcgNUQPMax;
private: System::Windows::Forms::Label^  fcgLBQPMax;
private: System::Windows::Forms::Label^  fcgLBQPMinMAX;
private: System::Windows::Forms::NumericUpDown^  fcgNUQPMin;
private: System::Windows::Forms::CheckBox^  fcgCBD3DMemAlloc;
private: System::Windows::Forms::GroupBox^  fcggroupBoxDetail;
private: System::Windows::Forms::CheckBox^  fcgCBIntraRefresh;
private: System::Windows::Forms::CheckBox^  fcgCBDeblock;
private: System::Windows::Forms::CheckBox^  fcgCBExtBRC;
private: System::Windows::Forms::CheckBox^  fcgCBMBBRC;
private: System::Windows::Forms::ComboBox^  fcgCXTrellis;
private: System::Windows::Forms::Label^  fcgLBTrellis;
private: System::Windows::Forms::Label^  fcgLBWinBRCSizeAuto;

private: System::Windows::Forms::NumericUpDown^  fcgNUWinBRCSize;

private: System::Windows::Forms::Label^  fcgLBWinBRCSize;


private: System::Windows::Forms::ComboBox^  fcgCXFPSConversion;
private: System::Windows::Forms::Label^  fcgLBFPSConversion;
private: System::Windows::Forms::Label^  fcgLBImageStabilizer;
private: System::Windows::Forms::ComboBox^  fcgCXImageStabilizer;
private: System::Windows::Forms::CheckBox^  fcgCBDirectBiasAdjust;
private: System::Windows::Forms::ComboBox^  fcgCXMVCostScaling;
private: System::Windows::Forms::Label^  fcgLBMVCostScaling;
private: System::Windows::Forms::ComboBox^  fcgCXTelecinePatterns;
private: System::Windows::Forms::TabControl^  fcgtabControlAudio;
private: System::Windows::Forms::TabPage^  fcgtabPageAudioMain;
private: System::Windows::Forms::ComboBox^  fcgCXAudioDelayCut;
private: System::Windows::Forms::Label^  fcgLBAudioDelayCut;
private: System::Windows::Forms::Label^  fcgCBAudioEncTiming;
private: System::Windows::Forms::ComboBox^  fcgCXAudioEncTiming;
private: System::Windows::Forms::ComboBox^  fcgCXAudioTempDir;
private: System::Windows::Forms::TextBox^  fcgTXCustomAudioTempDir;
private: System::Windows::Forms::Button^  fcgBTCustomAudioTempDir;
private: System::Windows::Forms::CheckBox^  fcgCBAudioUsePipe;
private: System::Windows::Forms::Label^  fcgLBAudioBitrate;
private: System::Windows::Forms::NumericUpDown^  fcgNUAudioBitrate;
private: System::Windows::Forms::CheckBox^  fcgCBAudio2pass;
private: System::Windows::Forms::ComboBox^  fcgCXAudioEncMode;
private: System::Windows::Forms::Label^  fcgLBAudioEncMode;
private: System::Windows::Forms::Button^  fcgBTAudioEncoderPath;
private: System::Windows::Forms::TextBox^  fcgTXAudioEncoderPath;
private: System::Windows::Forms::Label^  fcgLBAudioEncoderPath;
private: System::Windows::Forms::CheckBox^  fcgCBAudioOnly;
private: System::Windows::Forms::CheckBox^  fcgCBFAWCheck;
private: System::Windows::Forms::ComboBox^  fcgCXAudioEncoder;
private: System::Windows::Forms::Label^  fcgLBAudioEncoder;
private: System::Windows::Forms::Label^  fcgLBAudioTemp;
private: System::Windows::Forms::TabPage^  fcgtabPageAudioOther;
private: System::Windows::Forms::Panel^  panel2;
private: System::Windows::Forms::Label^  fcgLBBatAfterAudioString;
private: System::Windows::Forms::Label^  fcgLBBatBeforeAudioString;
private: System::Windows::Forms::Button^  fcgBTBatAfterAudioPath;
private: System::Windows::Forms::TextBox^  fcgTXBatAfterAudioPath;
private: System::Windows::Forms::Label^  fcgLBBatAfterAudioPath;
private: System::Windows::Forms::CheckBox^  fcgCBRunBatAfterAudio;
private: System::Windows::Forms::Panel^  panel1;
private: System::Windows::Forms::Button^  fcgBTBatBeforeAudioPath;
private: System::Windows::Forms::TextBox^  fcgTXBatBeforeAudioPath;
private: System::Windows::Forms::Label^  fcgLBBatBeforeAudioPath;
private: System::Windows::Forms::CheckBox^  fcgCBRunBatBeforeAudio;
private: System::Windows::Forms::ComboBox^  fcgCXAudioPriority;
private: System::Windows::Forms::Label^  fcgLBAudioPriority;
private: System::Windows::Forms::CheckBox^  fcgCBFixedFunc;
private: System::Windows::Forms::CheckBox^  fcgCBWeightB;
private: System::Windows::Forms::CheckBox^  fcgCBWeightP;
private: System::Windows::Forms::CheckBox^  fcgCBFadeDetect;
private: System::Windows::Forms::ComboBox^  fcgCXRotate;
private: System::Windows::Forms::Label^  fcgLBRotate;
private: System::Windows::Forms::GroupBox^  fcggroupBoxAvqsv;

private: System::Windows::Forms::CheckBox^  fcgCBAvqsv;
private: System::Windows::Forms::Button^  fcgBTAvqsvInputFile;










private: System::Windows::Forms::TextBox^  fcgTXAvqsvInputFile;

private: System::Windows::Forms::Label^  fcgLBAvqsvInputFile;
private: System::Windows::Forms::Label^  fcgLBAvqsvEncWarn;


private: System::Windows::Forms::Label^  fcgLBTrimInfo;
private: System::Windows::Forms::Label^  fcgLBTrim;
private: System::Windows::Forms::CheckBox^  fcgCBTrim;
private: System::Windows::Forms::TabPage^  fcgtabPageAvqsvAudio;

private: System::Windows::Forms::Label^  fcgLBAvqsvAudioBitrate2;

private: System::Windows::Forms::NumericUpDown^  fcgNUAvqsvAudioBitrate;


private: System::Windows::Forms::Label^  fcgLBAvqsvAudioBitrate;


private: System::Windows::Forms::ComboBox^  fcgCXAvqsvAudioEncoder;

private: System::Windows::Forms::Label^  fcgLBAvqsvAudioEncoder;



private: System::Windows::Forms::TabPage^  fcgtabPageMuxInternal;
private: System::Windows::Forms::CheckBox^  fcgCBCopyChapter;
private: System::Windows::Forms::CheckBox^  fcgCBCopySubtitle;























































































































    private: System::ComponentModel::IContainer^  components;


    

    private:
        /// <summary>
        /// 必要なデザイナ変数です。
        /// </summary>


#pragma region Windows Form Designer generated code
        /// <summary>
        /// デザイナ サポートに必要なメソッドです。このメソッドの内容を
        /// コード エディタで変更しないでください。
        /// </summary>
        void InitializeComponent(void)
        {
            this->components = (gcnew System::ComponentModel::Container());
            System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(frmConfig::typeid));
            this->fcgtoolStripSettings = (gcnew System::Windows::Forms::ToolStrip());
            this->fcgTSBSave = (gcnew System::Windows::Forms::ToolStripButton());
            this->fcgTSBSaveNew = (gcnew System::Windows::Forms::ToolStripButton());
            this->fcgTSBDelete = (gcnew System::Windows::Forms::ToolStripButton());
            this->fcgtoolStripSeparator1 = (gcnew System::Windows::Forms::ToolStripSeparator());
            this->fcgTSSettings = (gcnew System::Windows::Forms::ToolStripDropDownButton());
            this->fcgTSBBitrateCalc = (gcnew System::Windows::Forms::ToolStripButton());
            this->toolStripSeparator2 = (gcnew System::Windows::Forms::ToolStripSeparator());
            this->fcgTSBOtherSettings = (gcnew System::Windows::Forms::ToolStripButton());
            this->fcgTSLSettingsNotes = (gcnew System::Windows::Forms::ToolStripLabel());
            this->fcgTSTSettingsNotes = (gcnew System::Windows::Forms::ToolStripTextBox());
            this->fcgtabControlMux = (gcnew System::Windows::Forms::TabControl());
            this->fcgtabPageMP4 = (gcnew System::Windows::Forms::TabPage());
            this->fcgBTMP4RawPath = (gcnew System::Windows::Forms::Button());
            this->fcgTXMP4RawPath = (gcnew System::Windows::Forms::TextBox());
            this->fcgLBMP4RawPath = (gcnew System::Windows::Forms::Label());
            this->fcgCBMP4MuxApple = (gcnew System::Windows::Forms::CheckBox());
            this->fcgBTMP4BoxTempDir = (gcnew System::Windows::Forms::Button());
            this->fcgTXMP4BoxTempDir = (gcnew System::Windows::Forms::TextBox());
            this->fcgCXMP4BoxTempDir = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBMP4BoxTempDir = (gcnew System::Windows::Forms::Label());
            this->fcgBTTC2MP4Path = (gcnew System::Windows::Forms::Button());
            this->fcgTXTC2MP4Path = (gcnew System::Windows::Forms::TextBox());
            this->fcgBTMP4MuxerPath = (gcnew System::Windows::Forms::Button());
            this->fcgTXMP4MuxerPath = (gcnew System::Windows::Forms::TextBox());
            this->fcgLBTC2MP4Path = (gcnew System::Windows::Forms::Label());
            this->fcgLBMP4MuxerPath = (gcnew System::Windows::Forms::Label());
            this->fcgCXMP4CmdEx = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBMP4CmdEx = (gcnew System::Windows::Forms::Label());
            this->fcgCBMP4MuxerExt = (gcnew System::Windows::Forms::CheckBox());
            this->fcgtabPageMKV = (gcnew System::Windows::Forms::TabPage());
            this->fcgBTMKVMuxerPath = (gcnew System::Windows::Forms::Button());
            this->fcgTXMKVMuxerPath = (gcnew System::Windows::Forms::TextBox());
            this->fcgLBMKVMuxerPath = (gcnew System::Windows::Forms::Label());
            this->fcgCXMKVCmdEx = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBMKVMuxerCmdEx = (gcnew System::Windows::Forms::Label());
            this->fcgCBMKVMuxerExt = (gcnew System::Windows::Forms::CheckBox());
            this->fcgtabPageMPG = (gcnew System::Windows::Forms::TabPage());
            this->fcgBTMPGMuxerPath = (gcnew System::Windows::Forms::Button());
            this->fcgTXMPGMuxerPath = (gcnew System::Windows::Forms::TextBox());
            this->fcgLBMPGMuxerPath = (gcnew System::Windows::Forms::Label());
            this->fcgCXMPGCmdEx = (gcnew System::Windows::Forms::ComboBox());
            this->label3 = (gcnew System::Windows::Forms::Label());
            this->fcgCBMPGMuxerExt = (gcnew System::Windows::Forms::CheckBox());
            this->fcgtabPageMux = (gcnew System::Windows::Forms::TabPage());
            this->fcgCXMuxPriority = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBMuxPriority = (gcnew System::Windows::Forms::Label());
            this->fcgtabPageBat = (gcnew System::Windows::Forms::TabPage());
            this->fcgLBBatAfterString = (gcnew System::Windows::Forms::Label());
            this->fcgLBBatBeforeString = (gcnew System::Windows::Forms::Label());
            this->fcgPNSeparator = (gcnew System::Windows::Forms::Panel());
            this->fcgBTBatBeforePath = (gcnew System::Windows::Forms::Button());
            this->fcgTXBatBeforePath = (gcnew System::Windows::Forms::TextBox());
            this->fcgLBBatBeforePath = (gcnew System::Windows::Forms::Label());
            this->fcgCBWaitForBatBefore = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBRunBatBefore = (gcnew System::Windows::Forms::CheckBox());
            this->fcgBTBatAfterPath = (gcnew System::Windows::Forms::Button());
            this->fcgTXBatAfterPath = (gcnew System::Windows::Forms::TextBox());
            this->fcgLBBatAfterPath = (gcnew System::Windows::Forms::Label());
            this->fcgCBWaitForBatAfter = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBRunBatAfter = (gcnew System::Windows::Forms::CheckBox());
            this->fcgtabPageMuxInternal = (gcnew System::Windows::Forms::TabPage());
            this->fcgCBCopyChapter = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBCopySubtitle = (gcnew System::Windows::Forms::CheckBox());
            this->fcgTXCmd = (gcnew System::Windows::Forms::TextBox());
            this->fcgBTCancel = (gcnew System::Windows::Forms::Button());
            this->fcgBTOK = (gcnew System::Windows::Forms::Button());
            this->fcgBTDefault = (gcnew System::Windows::Forms::Button());
            this->fcgLBVersionDate = (gcnew System::Windows::Forms::Label());
            this->fcgLBVersion = (gcnew System::Windows::Forms::Label());
            this->fcgfolderBrowserTemp = (gcnew System::Windows::Forms::FolderBrowserDialog());
            this->fcgOpenFileDialog = (gcnew System::Windows::Forms::OpenFileDialog());
            this->fcgTTEx = (gcnew System::Windows::Forms::ToolTip(this->components));
            this->fcgtabControlQSV = (gcnew System::Windows::Forms::TabControl());
            this->tabPageVideoEnc = (gcnew System::Windows::Forms::TabPage());
            this->fcgCBFadeDetect = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBWeightB = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBWeightP = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBFixedFunc = (gcnew System::Windows::Forms::CheckBox());
            this->fcgLBWinBRCSizeAuto = (gcnew System::Windows::Forms::Label());
            this->fcgNUWinBRCSize = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBWinBRCSize = (gcnew System::Windows::Forms::Label());
            this->fcgLBQPMinMaxAuto = (gcnew System::Windows::Forms::Label());
            this->fcgNUQPMax = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBQPMax = (gcnew System::Windows::Forms::Label());
            this->fcgLBQPMinMAX = (gcnew System::Windows::Forms::Label());
            this->fcgNUQPMin = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgPNQVBR = (gcnew System::Windows::Forms::Panel());
            this->fcgNUQVBR = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBQVBR = (gcnew System::Windows::Forms::Label());
            this->fcgPNICQ = (gcnew System::Windows::Forms::Panel());
            this->fcgNUICQQuality = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBICQQuality = (gcnew System::Windows::Forms::Label());
            this->fcgCXLookaheadDS = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBLookaheadDS = (gcnew System::Windows::Forms::Label());
            this->fcgCBBPyramid = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBAdaptiveB = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBAdaptiveI = (gcnew System::Windows::Forms::CheckBox());
            this->fcgPNLookahead = (gcnew System::Windows::Forms::Panel());
            this->label1 = (gcnew System::Windows::Forms::Label());
            this->fcgNULookaheadDepth = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBLookaheadDepth = (gcnew System::Windows::Forms::Label());
            this->fcgLBFullrange = (gcnew System::Windows::Forms::Label());
            this->fcgCBFullrange = (gcnew System::Windows::Forms::CheckBox());
            this->fcgLBBlurayCompat = (gcnew System::Windows::Forms::Label());
            this->fcgCBBlurayCompat = (gcnew System::Windows::Forms::CheckBox());
            this->fcgPNAVBR = (gcnew System::Windows::Forms::Panel());
            this->fcgLBAVBRConvergence = (gcnew System::Windows::Forms::Label());
            this->fcgNUAVBRAccuarcy = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBAVBRAccuarcy = (gcnew System::Windows::Forms::Label());
            this->fcgNUAVBRConvergence = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBAVBRAccuarcy2 = (gcnew System::Windows::Forms::Label());
            this->fcgLBAVBRConvergence2 = (gcnew System::Windows::Forms::Label());
            this->fcgPNBitrate = (gcnew System::Windows::Forms::Panel());
            this->fcgLBBitrate = (gcnew System::Windows::Forms::Label());
            this->fcgNUBitrate = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBBitrate2 = (gcnew System::Windows::Forms::Label());
            this->fcgNUMaxkbps = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBMaxkbps = (gcnew System::Windows::Forms::Label());
            this->fcgLBMaxBitrate2 = (gcnew System::Windows::Forms::Label());
            this->fcgPNQP = (gcnew System::Windows::Forms::Panel());
            this->fcgLBQPI = (gcnew System::Windows::Forms::Label());
            this->fcgNUQPI = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUQPP = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUQPB = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBQPP = (gcnew System::Windows::Forms::Label());
            this->fcgLBQPB = (gcnew System::Windows::Forms::Label());
            this->fcgLBMFXLibDetectionSwValue = (gcnew System::Windows::Forms::Label());
            this->fcgLBMFXLibDetectionHwValue = (gcnew System::Windows::Forms::Label());
            this->fcgLBMFXLibDetectionSwStatus = (gcnew System::Windows::Forms::Label());
            this->fcgLBMFXLibDetectionHwStatus = (gcnew System::Windows::Forms::Label());
            this->fcgLBMFXLibDetection = (gcnew System::Windows::Forms::Label());
            this->fcgLBSlices2 = (gcnew System::Windows::Forms::Label());
            this->fcgNUSlices = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBSlices = (gcnew System::Windows::Forms::Label());
            this->fcgLBRefAuto = (gcnew System::Windows::Forms::Label());
            this->fcgLBGOPLengthAuto = (gcnew System::Windows::Forms::Label());
            this->fcgLBBframesAuto = (gcnew System::Windows::Forms::Label());
            this->fcgCXVideoFormat = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBVideoFormat = (gcnew System::Windows::Forms::Label());
            this->fcggroupBoxColor = (gcnew System::Windows::Forms::GroupBox());
            this->fcgCXTransfer = (gcnew System::Windows::Forms::ComboBox());
            this->fcgCXColorPrim = (gcnew System::Windows::Forms::ComboBox());
            this->fcgCXColorMatrix = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBTransfer = (gcnew System::Windows::Forms::Label());
            this->fcgLBColorPrim = (gcnew System::Windows::Forms::Label());
            this->fcgLBColorMatrix = (gcnew System::Windows::Forms::Label());
            this->fcgCBOpenGOP = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBSceneChange = (gcnew System::Windows::Forms::CheckBox());
            this->fcgGroupBoxAspectRatio = (gcnew System::Windows::Forms::GroupBox());
            this->fcgLBAspectRatio = (gcnew System::Windows::Forms::Label());
            this->fcgNUAspectRatioY = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUAspectRatioX = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgCXAspectRatio = (gcnew System::Windows::Forms::ComboBox());
            this->fcgCXInterlaced = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBInterlaced = (gcnew System::Windows::Forms::Label());
            this->fcgNUGopLength = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBGOPLength = (gcnew System::Windows::Forms::Label());
            this->fcgCXCodecLevel = (gcnew System::Windows::Forms::ComboBox());
            this->fcgCXCodecProfile = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBCodecLevel = (gcnew System::Windows::Forms::Label());
            this->fcgLBCodecProfile = (gcnew System::Windows::Forms::Label());
            this->fcgNUBframes = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNURef = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBBframes = (gcnew System::Windows::Forms::Label());
            this->fcgLBRef = (gcnew System::Windows::Forms::Label());
            this->fcgLBEncMode = (gcnew System::Windows::Forms::Label());
            this->fcgCXEncMode = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBQuality = (gcnew System::Windows::Forms::Label());
            this->fcgLBOutputType = (gcnew System::Windows::Forms::Label());
            this->fcgCXQuality = (gcnew System::Windows::Forms::ComboBox());
            this->fcgCXOutputType = (gcnew System::Windows::Forms::ComboBox());
            this->fcgCBHWEncode = (gcnew System::Windows::Forms::CheckBox());
            this->tabPageVpp = (gcnew System::Windows::Forms::TabPage());
            this->fcggroupBoxDetail = (gcnew System::Windows::Forms::GroupBox());
            this->fcgCBDirectBiasAdjust = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCXMVCostScaling = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBMVCostScaling = (gcnew System::Windows::Forms::Label());
            this->fcgCBExtBRC = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBMBBRC = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCXTrellis = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBTrellis = (gcnew System::Windows::Forms::Label());
            this->fcgCBIntraRefresh = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBDeblock = (gcnew System::Windows::Forms::CheckBox());
            this->fcgPNExtSettings = (gcnew System::Windows::Forms::Panel());
            this->fcgLBInterPred = (gcnew System::Windows::Forms::Label());
            this->fcgLBIntraPred = (gcnew System::Windows::Forms::Label());
            this->fcgCXIntraPred = (gcnew System::Windows::Forms::ComboBox());
            this->fcgCXInterPred = (gcnew System::Windows::Forms::ComboBox());
            this->fcgCXMVPred = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBMVPred = (gcnew System::Windows::Forms::Label());
            this->fcgLBMVWindowSize = (gcnew System::Windows::Forms::Label());
            this->fcgNUMVSearchWindow = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBMVSearch = (gcnew System::Windows::Forms::Label());
            this->fcgCBRDO = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBCABAC = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBUseVpp = (gcnew System::Windows::Forms::CheckBox());
            this->fcggroupBoxVpp = (gcnew System::Windows::Forms::GroupBox());
            this->fcgCXRotate = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBRotate = (gcnew System::Windows::Forms::Label());
            this->fcgCXTelecinePatterns = (gcnew System::Windows::Forms::ComboBox());
            this->fcgCXImageStabilizer = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBImageStabilizer = (gcnew System::Windows::Forms::Label());
            this->fcgLBFPSConversion = (gcnew System::Windows::Forms::Label());
            this->fcgCXFPSConversion = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBDeinterlaceDesc = (gcnew System::Windows::Forms::Label());
            this->fcgCXDeinterlace = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBDeinterlace = (gcnew System::Windows::Forms::Label());
            this->fcgCBVppDetail = (gcnew System::Windows::Forms::CheckBox());
            this->fcggroupBoxVppDetail = (gcnew System::Windows::Forms::GroupBox());
            this->fcgNUVppDetail = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBDetail = (gcnew System::Windows::Forms::Label());
            this->fcgCBVppDenoise = (gcnew System::Windows::Forms::CheckBox());
            this->fcggroupBoxVppDenoise = (gcnew System::Windows::Forms::GroupBox());
            this->fcgNUVppDenoise = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBVppDenoise = (gcnew System::Windows::Forms::Label());
            this->fcgCBVppResize = (gcnew System::Windows::Forms::CheckBox());
            this->fcggroupBoxVppResize = (gcnew System::Windows::Forms::GroupBox());
            this->fcgNUVppResizeW = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppResizeH = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBVppResize = (gcnew System::Windows::Forms::Label());
            this->tabPageExOpt = (gcnew System::Windows::Forms::TabPage());
            this->fcgCBD3DMemAlloc = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBAuoTcfileout = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBAFS = (gcnew System::Windows::Forms::CheckBox());
            this->fcgNUInputBufSize = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBInputBufSize = (gcnew System::Windows::Forms::Label());
            this->fcgLBTempDir = (gcnew System::Windows::Forms::Label());
            this->fcgBTCustomTempDir = (gcnew System::Windows::Forms::Button());
            this->fcgTXCustomTempDir = (gcnew System::Windows::Forms::TextBox());
            this->fcgCXTempDir = (gcnew System::Windows::Forms::ComboBox());
            this->tabPageFeatures = (gcnew System::Windows::Forms::TabPage());
            this->fcgLBGPUInfoOnFeatureTab = (gcnew System::Windows::Forms::Label());
            this->fcgLBGPUInfoLabelOnFeatureTab = (gcnew System::Windows::Forms::Label());
            this->fcgLBCPUInfoOnFeatureTab = (gcnew System::Windows::Forms::Label());
            this->fcgLBCPUInfoLabelOnFeatureTab = (gcnew System::Windows::Forms::Label());
            this->fcgBTSaveFeatureList = (gcnew System::Windows::Forms::Button());
            this->fcgLBFeaturesCurrentAPIVer = (gcnew System::Windows::Forms::Label());
            this->fcgLBFeaturesShowCurrentAPI = (gcnew System::Windows::Forms::Label());
            this->fcgDGVFeatures = (gcnew System::Windows::Forms::DataGridView());
            this->fcgCSExeFiles = (gcnew System::Windows::Forms::ContextMenuStrip(this->components));
            this->fcgTSExeFileshelp = (gcnew System::Windows::Forms::ToolStripMenuItem());
            this->fcgLBguiExBlog = (gcnew System::Windows::Forms::LinkLabel());
            this->fcgtabControlAudio = (gcnew System::Windows::Forms::TabControl());
            this->fcgtabPageAudioMain = (gcnew System::Windows::Forms::TabPage());
            this->fcgCXAudioDelayCut = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBAudioDelayCut = (gcnew System::Windows::Forms::Label());
            this->fcgCBAudioEncTiming = (gcnew System::Windows::Forms::Label());
            this->fcgCXAudioEncTiming = (gcnew System::Windows::Forms::ComboBox());
            this->fcgCXAudioTempDir = (gcnew System::Windows::Forms::ComboBox());
            this->fcgTXCustomAudioTempDir = (gcnew System::Windows::Forms::TextBox());
            this->fcgBTCustomAudioTempDir = (gcnew System::Windows::Forms::Button());
            this->fcgCBAudioUsePipe = (gcnew System::Windows::Forms::CheckBox());
            this->fcgLBAudioBitrate = (gcnew System::Windows::Forms::Label());
            this->fcgNUAudioBitrate = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgCBAudio2pass = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCXAudioEncMode = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBAudioEncMode = (gcnew System::Windows::Forms::Label());
            this->fcgBTAudioEncoderPath = (gcnew System::Windows::Forms::Button());
            this->fcgTXAudioEncoderPath = (gcnew System::Windows::Forms::TextBox());
            this->fcgLBAudioEncoderPath = (gcnew System::Windows::Forms::Label());
            this->fcgCBAudioOnly = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBFAWCheck = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCXAudioEncoder = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBAudioEncoder = (gcnew System::Windows::Forms::Label());
            this->fcgLBAudioTemp = (gcnew System::Windows::Forms::Label());
            this->fcgtabPageAudioOther = (gcnew System::Windows::Forms::TabPage());
            this->panel2 = (gcnew System::Windows::Forms::Panel());
            this->fcgLBBatAfterAudioString = (gcnew System::Windows::Forms::Label());
            this->fcgLBBatBeforeAudioString = (gcnew System::Windows::Forms::Label());
            this->fcgBTBatAfterAudioPath = (gcnew System::Windows::Forms::Button());
            this->fcgTXBatAfterAudioPath = (gcnew System::Windows::Forms::TextBox());
            this->fcgLBBatAfterAudioPath = (gcnew System::Windows::Forms::Label());
            this->fcgCBRunBatAfterAudio = (gcnew System::Windows::Forms::CheckBox());
            this->panel1 = (gcnew System::Windows::Forms::Panel());
            this->fcgBTBatBeforeAudioPath = (gcnew System::Windows::Forms::Button());
            this->fcgTXBatBeforeAudioPath = (gcnew System::Windows::Forms::TextBox());
            this->fcgLBBatBeforeAudioPath = (gcnew System::Windows::Forms::Label());
            this->fcgCBRunBatBeforeAudio = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCXAudioPriority = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBAudioPriority = (gcnew System::Windows::Forms::Label());
            this->fcgtabPageAvqsvAudio = (gcnew System::Windows::Forms::TabPage());
            this->fcgLBAvqsvAudioBitrate2 = (gcnew System::Windows::Forms::Label());
            this->fcgNUAvqsvAudioBitrate = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBAvqsvAudioBitrate = (gcnew System::Windows::Forms::Label());
            this->fcgCXAvqsvAudioEncoder = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBAvqsvAudioEncoder = (gcnew System::Windows::Forms::Label());
            this->fcggroupBoxAvqsv = (gcnew System::Windows::Forms::GroupBox());
            this->fcgLBTrimInfo = (gcnew System::Windows::Forms::Label());
            this->fcgLBTrim = (gcnew System::Windows::Forms::Label());
            this->fcgCBTrim = (gcnew System::Windows::Forms::CheckBox());
            this->fcgLBAvqsvEncWarn = (gcnew System::Windows::Forms::Label());
            this->fcgBTAvqsvInputFile = (gcnew System::Windows::Forms::Button());
            this->fcgTXAvqsvInputFile = (gcnew System::Windows::Forms::TextBox());
            this->fcgLBAvqsvInputFile = (gcnew System::Windows::Forms::Label());
            this->fcgCBAvqsv = (gcnew System::Windows::Forms::CheckBox());
            this->fcgtoolStripSettings->SuspendLayout();
            this->fcgtabControlMux->SuspendLayout();
            this->fcgtabPageMP4->SuspendLayout();
            this->fcgtabPageMKV->SuspendLayout();
            this->fcgtabPageMPG->SuspendLayout();
            this->fcgtabPageMux->SuspendLayout();
            this->fcgtabPageBat->SuspendLayout();
            this->fcgtabPageMuxInternal->SuspendLayout();
            this->fcgtabControlQSV->SuspendLayout();
            this->tabPageVideoEnc->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUWinBRCSize))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMax))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMin))->BeginInit();
            this->fcgPNQVBR->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQVBR))->BeginInit();
            this->fcgPNICQ->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUICQQuality))->BeginInit();
            this->fcgPNLookahead->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNULookaheadDepth))->BeginInit();
            this->fcgPNAVBR->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAVBRAccuarcy))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAVBRConvergence))->BeginInit();
            this->fcgPNBitrate->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUBitrate))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUMaxkbps))->BeginInit();
            this->fcgPNQP->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPI))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPP))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPB))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUSlices))->BeginInit();
            this->fcggroupBoxColor->SuspendLayout();
            this->fcgGroupBoxAspectRatio->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAspectRatioY))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAspectRatioX))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUGopLength))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUBframes))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNURef))->BeginInit();
            this->tabPageVpp->SuspendLayout();
            this->fcggroupBoxDetail->SuspendLayout();
            this->fcgPNExtSettings->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUMVSearchWindow))->BeginInit();
            this->fcggroupBoxVpp->SuspendLayout();
            this->fcggroupBoxVppDetail->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDetail))->BeginInit();
            this->fcggroupBoxVppDenoise->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoise))->BeginInit();
            this->fcggroupBoxVppResize->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppResizeW))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppResizeH))->BeginInit();
            this->tabPageExOpt->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUInputBufSize))->BeginInit();
            this->tabPageFeatures->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgDGVFeatures))->BeginInit();
            this->fcgCSExeFiles->SuspendLayout();
            this->fcgtabControlAudio->SuspendLayout();
            this->fcgtabPageAudioMain->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAudioBitrate))->BeginInit();
            this->fcgtabPageAudioOther->SuspendLayout();
            this->fcgtabPageAvqsvAudio->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAvqsvAudioBitrate))->BeginInit();
            this->fcggroupBoxAvqsv->SuspendLayout();
            this->SuspendLayout();
            // 
            // fcgtoolStripSettings
            // 
            this->fcgtoolStripSettings->ImageScalingSize = System::Drawing::Size(18, 18);
            this->fcgtoolStripSettings->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(10) {
                this->fcgTSBSave,
                    this->fcgTSBSaveNew, this->fcgTSBDelete, this->fcgtoolStripSeparator1, this->fcgTSSettings, this->fcgTSBBitrateCalc, this->toolStripSeparator2,
                    this->fcgTSBOtherSettings, this->fcgTSLSettingsNotes, this->fcgTSTSettingsNotes
            });
            this->fcgtoolStripSettings->Location = System::Drawing::Point(0, 0);
            this->fcgtoolStripSettings->Name = L"fcgtoolStripSettings";
            this->fcgtoolStripSettings->Size = System::Drawing::Size(1008, 25);
            this->fcgtoolStripSettings->TabIndex = 1;
            this->fcgtoolStripSettings->Text = L"toolStrip1";
            // 
            // fcgTSBSave
            // 
            this->fcgTSBSave->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgTSBSave.Image")));
            this->fcgTSBSave->ImageTransparentColor = System::Drawing::Color::Magenta;
            this->fcgTSBSave->Name = L"fcgTSBSave";
            this->fcgTSBSave->Size = System::Drawing::Size(86, 22);
            this->fcgTSBSave->Text = L"上書き保存";
            this->fcgTSBSave->Click += gcnew System::EventHandler(this, &frmConfig::fcgTSBSave_Click);
            // 
            // fcgTSBSaveNew
            // 
            this->fcgTSBSaveNew->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgTSBSaveNew.Image")));
            this->fcgTSBSaveNew->ImageTransparentColor = System::Drawing::Color::Black;
            this->fcgTSBSaveNew->Name = L"fcgTSBSaveNew";
            this->fcgTSBSaveNew->Size = System::Drawing::Size(77, 22);
            this->fcgTSBSaveNew->Text = L"新規保存";
            this->fcgTSBSaveNew->Click += gcnew System::EventHandler(this, &frmConfig::fcgTSBSaveNew_Click);
            // 
            // fcgTSBDelete
            // 
            this->fcgTSBDelete->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgTSBDelete.Image")));
            this->fcgTSBDelete->ImageTransparentColor = System::Drawing::Color::Magenta;
            this->fcgTSBDelete->Name = L"fcgTSBDelete";
            this->fcgTSBDelete->Size = System::Drawing::Size(53, 22);
            this->fcgTSBDelete->Text = L"削除";
            this->fcgTSBDelete->Click += gcnew System::EventHandler(this, &frmConfig::fcgTSBDelete_Click);
            // 
            // fcgtoolStripSeparator1
            // 
            this->fcgtoolStripSeparator1->Name = L"fcgtoolStripSeparator1";
            this->fcgtoolStripSeparator1->Size = System::Drawing::Size(6, 25);
            // 
            // fcgTSSettings
            // 
            this->fcgTSSettings->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgTSSettings.Image")));
            this->fcgTSSettings->ImageTransparentColor = System::Drawing::Color::Magenta;
            this->fcgTSSettings->Name = L"fcgTSSettings";
            this->fcgTSSettings->Size = System::Drawing::Size(81, 22);
            this->fcgTSSettings->Text = L"プリセット";
            this->fcgTSSettings->DropDownItemClicked += gcnew System::Windows::Forms::ToolStripItemClickedEventHandler(this, &frmConfig::fcgTSSettings_DropDownItemClicked);
            this->fcgTSSettings->Click += gcnew System::EventHandler(this, &frmConfig::fcgTSSettings_Click);
            // 
            // fcgTSBBitrateCalc
            // 
            this->fcgTSBBitrateCalc->Alignment = System::Windows::Forms::ToolStripItemAlignment::Right;
            this->fcgTSBBitrateCalc->CheckOnClick = true;
            this->fcgTSBBitrateCalc->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Text;
            this->fcgTSBBitrateCalc->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgTSBBitrateCalc.Image")));
            this->fcgTSBBitrateCalc->ImageTransparentColor = System::Drawing::Color::Magenta;
            this->fcgTSBBitrateCalc->Name = L"fcgTSBBitrateCalc";
            this->fcgTSBBitrateCalc->Size = System::Drawing::Size(96, 22);
            this->fcgTSBBitrateCalc->Text = L"ビットレート計算機";
            this->fcgTSBBitrateCalc->CheckedChanged += gcnew System::EventHandler(this, &frmConfig::fcgTSBBitrateCalc_CheckedChanged);
            // 
            // toolStripSeparator2
            // 
            this->toolStripSeparator2->Alignment = System::Windows::Forms::ToolStripItemAlignment::Right;
            this->toolStripSeparator2->Name = L"toolStripSeparator2";
            this->toolStripSeparator2->Size = System::Drawing::Size(6, 25);
            // 
            // fcgTSBOtherSettings
            // 
            this->fcgTSBOtherSettings->Alignment = System::Windows::Forms::ToolStripItemAlignment::Right;
            this->fcgTSBOtherSettings->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Text;
            this->fcgTSBOtherSettings->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgTSBOtherSettings.Image")));
            this->fcgTSBOtherSettings->ImageTransparentColor = System::Drawing::Color::Magenta;
            this->fcgTSBOtherSettings->Name = L"fcgTSBOtherSettings";
            this->fcgTSBOtherSettings->Size = System::Drawing::Size(76, 22);
            this->fcgTSBOtherSettings->Text = L"その他の設定";
            this->fcgTSBOtherSettings->Click += gcnew System::EventHandler(this, &frmConfig::fcgTSBOtherSettings_Click);
            // 
            // fcgTSLSettingsNotes
            // 
            this->fcgTSLSettingsNotes->DoubleClickEnabled = true;
            this->fcgTSLSettingsNotes->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgTSLSettingsNotes->Margin = System::Windows::Forms::Padding(3, 1, 0, 2);
            this->fcgTSLSettingsNotes->Name = L"fcgTSLSettingsNotes";
            this->fcgTSLSettingsNotes->Overflow = System::Windows::Forms::ToolStripItemOverflow::Never;
            this->fcgTSLSettingsNotes->Size = System::Drawing::Size(45, 22);
            this->fcgTSLSettingsNotes->Text = L"メモ表示";
            this->fcgTSLSettingsNotes->DoubleClick += gcnew System::EventHandler(this, &frmConfig::fcgTSLSettingsNotes_DoubleClick);
            // 
            // fcgTSTSettingsNotes
            // 
            this->fcgTSTSettingsNotes->BackColor = System::Drawing::SystemColors::Window;
            this->fcgTSTSettingsNotes->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgTSTSettingsNotes->Margin = System::Windows::Forms::Padding(3, 0, 1, 0);
            this->fcgTSTSettingsNotes->Name = L"fcgTSTSettingsNotes";
            this->fcgTSTSettingsNotes->Size = System::Drawing::Size(200, 25);
            this->fcgTSTSettingsNotes->Text = L"メモ...";
            this->fcgTSTSettingsNotes->Visible = false;
            this->fcgTSTSettingsNotes->Leave += gcnew System::EventHandler(this, &frmConfig::fcgTSTSettingsNotes_Leave);
            this->fcgTSTSettingsNotes->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &frmConfig::fcgTSTSettingsNotes_KeyDown);
            this->fcgTSTSettingsNotes->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTSTSettingsNotes_TextChanged);
            // 
            // fcgtabControlMux
            // 
            this->fcgtabControlMux->Controls->Add(this->fcgtabPageMP4);
            this->fcgtabControlMux->Controls->Add(this->fcgtabPageMKV);
            this->fcgtabControlMux->Controls->Add(this->fcgtabPageMPG);
            this->fcgtabControlMux->Controls->Add(this->fcgtabPageMux);
            this->fcgtabControlMux->Controls->Add(this->fcgtabPageBat);
            this->fcgtabControlMux->Controls->Add(this->fcgtabPageMuxInternal);
            this->fcgtabControlMux->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgtabControlMux->Location = System::Drawing::Point(622, 339);
            this->fcgtabControlMux->Name = L"fcgtabControlMux";
            this->fcgtabControlMux->SelectedIndex = 0;
            this->fcgtabControlMux->Size = System::Drawing::Size(384, 201);
            this->fcgtabControlMux->TabIndex = 3;
            this->fcgtabControlMux->Tag = L"";
            // 
            // fcgtabPageMP4
            // 
            this->fcgtabPageMP4->Controls->Add(this->fcgBTMP4RawPath);
            this->fcgtabPageMP4->Controls->Add(this->fcgTXMP4RawPath);
            this->fcgtabPageMP4->Controls->Add(this->fcgLBMP4RawPath);
            this->fcgtabPageMP4->Controls->Add(this->fcgCBMP4MuxApple);
            this->fcgtabPageMP4->Controls->Add(this->fcgBTMP4BoxTempDir);
            this->fcgtabPageMP4->Controls->Add(this->fcgTXMP4BoxTempDir);
            this->fcgtabPageMP4->Controls->Add(this->fcgCXMP4BoxTempDir);
            this->fcgtabPageMP4->Controls->Add(this->fcgLBMP4BoxTempDir);
            this->fcgtabPageMP4->Controls->Add(this->fcgBTTC2MP4Path);
            this->fcgtabPageMP4->Controls->Add(this->fcgTXTC2MP4Path);
            this->fcgtabPageMP4->Controls->Add(this->fcgBTMP4MuxerPath);
            this->fcgtabPageMP4->Controls->Add(this->fcgTXMP4MuxerPath);
            this->fcgtabPageMP4->Controls->Add(this->fcgLBTC2MP4Path);
            this->fcgtabPageMP4->Controls->Add(this->fcgLBMP4MuxerPath);
            this->fcgtabPageMP4->Controls->Add(this->fcgCXMP4CmdEx);
            this->fcgtabPageMP4->Controls->Add(this->fcgLBMP4CmdEx);
            this->fcgtabPageMP4->Controls->Add(this->fcgCBMP4MuxerExt);
            this->fcgtabPageMP4->Location = System::Drawing::Point(4, 23);
            this->fcgtabPageMP4->Name = L"fcgtabPageMP4";
            this->fcgtabPageMP4->Padding = System::Windows::Forms::Padding(3);
            this->fcgtabPageMP4->Size = System::Drawing::Size(376, 174);
            this->fcgtabPageMP4->TabIndex = 0;
            this->fcgtabPageMP4->Text = L"mp4";
            this->fcgtabPageMP4->UseVisualStyleBackColor = true;
            // 
            // fcgBTMP4RawPath
            // 
            this->fcgBTMP4RawPath->Location = System::Drawing::Point(340, 102);
            this->fcgBTMP4RawPath->Name = L"fcgBTMP4RawPath";
            this->fcgBTMP4RawPath->Size = System::Drawing::Size(30, 23);
            this->fcgBTMP4RawPath->TabIndex = 8;
            this->fcgBTMP4RawPath->Text = L"...";
            this->fcgBTMP4RawPath->UseVisualStyleBackColor = true;
            this->fcgBTMP4RawPath->Visible = false;
            this->fcgBTMP4RawPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTMP4RawMuxerPath_Click);
            // 
            // fcgTXMP4RawPath
            // 
            this->fcgTXMP4RawPath->AllowDrop = true;
            this->fcgTXMP4RawPath->Location = System::Drawing::Point(136, 103);
            this->fcgTXMP4RawPath->Name = L"fcgTXMP4RawPath";
            this->fcgTXMP4RawPath->Size = System::Drawing::Size(202, 21);
            this->fcgTXMP4RawPath->TabIndex = 7;
            this->fcgTXMP4RawPath->Visible = false;
            this->fcgTXMP4RawPath->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXMP4RawMuxerPath_TextChanged);
            this->fcgTXMP4RawPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
            this->fcgTXMP4RawPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
            // 
            // fcgLBMP4RawPath
            // 
            this->fcgLBMP4RawPath->AutoSize = true;
            this->fcgLBMP4RawPath->Location = System::Drawing::Point(4, 106);
            this->fcgLBMP4RawPath->Name = L"fcgLBMP4RawPath";
            this->fcgLBMP4RawPath->Size = System::Drawing::Size(49, 14);
            this->fcgLBMP4RawPath->TabIndex = 21;
            this->fcgLBMP4RawPath->Text = L"～の指定";
            this->fcgLBMP4RawPath->Visible = false;
            // 
            // fcgCBMP4MuxApple
            // 
            this->fcgCBMP4MuxApple->AutoSize = true;
            this->fcgCBMP4MuxApple->Location = System::Drawing::Point(254, 34);
            this->fcgCBMP4MuxApple->Name = L"fcgCBMP4MuxApple";
            this->fcgCBMP4MuxApple->Size = System::Drawing::Size(109, 18);
            this->fcgCBMP4MuxApple->TabIndex = 2;
            this->fcgCBMP4MuxApple->Tag = L"chValue";
            this->fcgCBMP4MuxApple->Text = L"Apple形式に対応";
            this->fcgCBMP4MuxApple->UseVisualStyleBackColor = true;
            // 
            // fcgBTMP4BoxTempDir
            // 
            this->fcgBTMP4BoxTempDir->Location = System::Drawing::Point(340, 146);
            this->fcgBTMP4BoxTempDir->Name = L"fcgBTMP4BoxTempDir";
            this->fcgBTMP4BoxTempDir->Size = System::Drawing::Size(30, 23);
            this->fcgBTMP4BoxTempDir->TabIndex = 11;
            this->fcgBTMP4BoxTempDir->Text = L"...";
            this->fcgBTMP4BoxTempDir->UseVisualStyleBackColor = true;
            this->fcgBTMP4BoxTempDir->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTMP4BoxTempDir_Click);
            // 
            // fcgTXMP4BoxTempDir
            // 
            this->fcgTXMP4BoxTempDir->Location = System::Drawing::Point(107, 147);
            this->fcgTXMP4BoxTempDir->Name = L"fcgTXMP4BoxTempDir";
            this->fcgTXMP4BoxTempDir->Size = System::Drawing::Size(227, 21);
            this->fcgTXMP4BoxTempDir->TabIndex = 10;
            this->fcgTXMP4BoxTempDir->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXMP4BoxTempDir_TextChanged);
            // 
            // fcgCXMP4BoxTempDir
            // 
            this->fcgCXMP4BoxTempDir->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXMP4BoxTempDir->FormattingEnabled = true;
            this->fcgCXMP4BoxTempDir->Location = System::Drawing::Point(145, 119);
            this->fcgCXMP4BoxTempDir->Name = L"fcgCXMP4BoxTempDir";
            this->fcgCXMP4BoxTempDir->Size = System::Drawing::Size(206, 22);
            this->fcgCXMP4BoxTempDir->TabIndex = 9;
            this->fcgCXMP4BoxTempDir->Tag = L"chValue";
            // 
            // fcgLBMP4BoxTempDir
            // 
            this->fcgLBMP4BoxTempDir->AutoSize = true;
            this->fcgLBMP4BoxTempDir->Location = System::Drawing::Point(25, 122);
            this->fcgLBMP4BoxTempDir->Name = L"fcgLBMP4BoxTempDir";
            this->fcgLBMP4BoxTempDir->Size = System::Drawing::Size(105, 14);
            this->fcgLBMP4BoxTempDir->TabIndex = 18;
            this->fcgLBMP4BoxTempDir->Text = L"mp4box一時フォルダ";
            // 
            // fcgBTTC2MP4Path
            // 
            this->fcgBTTC2MP4Path->Location = System::Drawing::Point(340, 80);
            this->fcgBTTC2MP4Path->Name = L"fcgBTTC2MP4Path";
            this->fcgBTTC2MP4Path->Size = System::Drawing::Size(30, 23);
            this->fcgBTTC2MP4Path->TabIndex = 6;
            this->fcgBTTC2MP4Path->Text = L"...";
            this->fcgBTTC2MP4Path->UseVisualStyleBackColor = true;
            this->fcgBTTC2MP4Path->Visible = false;
            this->fcgBTTC2MP4Path->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTTC2MP4Path_Click);
            // 
            // fcgTXTC2MP4Path
            // 
            this->fcgTXTC2MP4Path->AllowDrop = true;
            this->fcgTXTC2MP4Path->Location = System::Drawing::Point(136, 81);
            this->fcgTXTC2MP4Path->Name = L"fcgTXTC2MP4Path";
            this->fcgTXTC2MP4Path->Size = System::Drawing::Size(202, 21);
            this->fcgTXTC2MP4Path->TabIndex = 5;
            this->fcgTXTC2MP4Path->Visible = false;
            this->fcgTXTC2MP4Path->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXTC2MP4Path_TextChanged);
            this->fcgTXTC2MP4Path->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
            this->fcgTXTC2MP4Path->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
            // 
            // fcgBTMP4MuxerPath
            // 
            this->fcgBTMP4MuxerPath->Location = System::Drawing::Point(340, 58);
            this->fcgBTMP4MuxerPath->Name = L"fcgBTMP4MuxerPath";
            this->fcgBTMP4MuxerPath->Size = System::Drawing::Size(30, 23);
            this->fcgBTMP4MuxerPath->TabIndex = 4;
            this->fcgBTMP4MuxerPath->Text = L"...";
            this->fcgBTMP4MuxerPath->UseVisualStyleBackColor = true;
            this->fcgBTMP4MuxerPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTMP4MuxerPath_Click);
            // 
            // fcgTXMP4MuxerPath
            // 
            this->fcgTXMP4MuxerPath->AllowDrop = true;
            this->fcgTXMP4MuxerPath->Location = System::Drawing::Point(136, 59);
            this->fcgTXMP4MuxerPath->Name = L"fcgTXMP4MuxerPath";
            this->fcgTXMP4MuxerPath->Size = System::Drawing::Size(202, 21);
            this->fcgTXMP4MuxerPath->TabIndex = 3;
            this->fcgTXMP4MuxerPath->Tag = L"";
            this->fcgTXMP4MuxerPath->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXMP4MuxerPath_TextChanged);
            this->fcgTXMP4MuxerPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
            this->fcgTXMP4MuxerPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
            // 
            // fcgLBTC2MP4Path
            // 
            this->fcgLBTC2MP4Path->AutoSize = true;
            this->fcgLBTC2MP4Path->Location = System::Drawing::Point(4, 84);
            this->fcgLBTC2MP4Path->Name = L"fcgLBTC2MP4Path";
            this->fcgLBTC2MP4Path->Size = System::Drawing::Size(49, 14);
            this->fcgLBTC2MP4Path->TabIndex = 4;
            this->fcgLBTC2MP4Path->Text = L"～の指定";
            this->fcgLBTC2MP4Path->Visible = false;
            // 
            // fcgLBMP4MuxerPath
            // 
            this->fcgLBMP4MuxerPath->AutoSize = true;
            this->fcgLBMP4MuxerPath->Location = System::Drawing::Point(4, 62);
            this->fcgLBMP4MuxerPath->Name = L"fcgLBMP4MuxerPath";
            this->fcgLBMP4MuxerPath->Size = System::Drawing::Size(49, 14);
            this->fcgLBMP4MuxerPath->TabIndex = 3;
            this->fcgLBMP4MuxerPath->Text = L"～の指定";
            // 
            // fcgCXMP4CmdEx
            // 
            this->fcgCXMP4CmdEx->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXMP4CmdEx->FormattingEnabled = true;
            this->fcgCXMP4CmdEx->Location = System::Drawing::Point(213, 7);
            this->fcgCXMP4CmdEx->Name = L"fcgCXMP4CmdEx";
            this->fcgCXMP4CmdEx->Size = System::Drawing::Size(157, 22);
            this->fcgCXMP4CmdEx->TabIndex = 1;
            this->fcgCXMP4CmdEx->Tag = L"chValue";
            // 
            // fcgLBMP4CmdEx
            // 
            this->fcgLBMP4CmdEx->AutoSize = true;
            this->fcgLBMP4CmdEx->Location = System::Drawing::Point(139, 10);
            this->fcgLBMP4CmdEx->Name = L"fcgLBMP4CmdEx";
            this->fcgLBMP4CmdEx->Size = System::Drawing::Size(68, 14);
            this->fcgLBMP4CmdEx->TabIndex = 1;
            this->fcgLBMP4CmdEx->Text = L"拡張オプション";
            // 
            // fcgCBMP4MuxerExt
            // 
            this->fcgCBMP4MuxerExt->AutoSize = true;
            this->fcgCBMP4MuxerExt->Location = System::Drawing::Point(10, 9);
            this->fcgCBMP4MuxerExt->Name = L"fcgCBMP4MuxerExt";
            this->fcgCBMP4MuxerExt->Size = System::Drawing::Size(113, 18);
            this->fcgCBMP4MuxerExt->TabIndex = 0;
            this->fcgCBMP4MuxerExt->Tag = L"chValue";
            this->fcgCBMP4MuxerExt->Text = L"外部muxerを使用";
            this->fcgCBMP4MuxerExt->UseVisualStyleBackColor = true;
            // 
            // fcgtabPageMKV
            // 
            this->fcgtabPageMKV->Controls->Add(this->fcgBTMKVMuxerPath);
            this->fcgtabPageMKV->Controls->Add(this->fcgTXMKVMuxerPath);
            this->fcgtabPageMKV->Controls->Add(this->fcgLBMKVMuxerPath);
            this->fcgtabPageMKV->Controls->Add(this->fcgCXMKVCmdEx);
            this->fcgtabPageMKV->Controls->Add(this->fcgLBMKVMuxerCmdEx);
            this->fcgtabPageMKV->Controls->Add(this->fcgCBMKVMuxerExt);
            this->fcgtabPageMKV->Location = System::Drawing::Point(4, 23);
            this->fcgtabPageMKV->Name = L"fcgtabPageMKV";
            this->fcgtabPageMKV->Padding = System::Windows::Forms::Padding(3);
            this->fcgtabPageMKV->Size = System::Drawing::Size(376, 174);
            this->fcgtabPageMKV->TabIndex = 1;
            this->fcgtabPageMKV->Text = L"mkv";
            this->fcgtabPageMKV->UseVisualStyleBackColor = true;
            // 
            // fcgBTMKVMuxerPath
            // 
            this->fcgBTMKVMuxerPath->Location = System::Drawing::Point(340, 76);
            this->fcgBTMKVMuxerPath->Name = L"fcgBTMKVMuxerPath";
            this->fcgBTMKVMuxerPath->Size = System::Drawing::Size(30, 23);
            this->fcgBTMKVMuxerPath->TabIndex = 3;
            this->fcgBTMKVMuxerPath->Text = L"...";
            this->fcgBTMKVMuxerPath->UseVisualStyleBackColor = true;
            this->fcgBTMKVMuxerPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTMKVMuxerPath_Click);
            // 
            // fcgTXMKVMuxerPath
            // 
            this->fcgTXMKVMuxerPath->Location = System::Drawing::Point(131, 77);
            this->fcgTXMKVMuxerPath->Name = L"fcgTXMKVMuxerPath";
            this->fcgTXMKVMuxerPath->Size = System::Drawing::Size(207, 21);
            this->fcgTXMKVMuxerPath->TabIndex = 2;
            this->fcgTXMKVMuxerPath->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXMKVMuxerPath_TextChanged);
            this->fcgTXMKVMuxerPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
            this->fcgTXMKVMuxerPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
            // 
            // fcgLBMKVMuxerPath
            // 
            this->fcgLBMKVMuxerPath->AutoSize = true;
            this->fcgLBMKVMuxerPath->Location = System::Drawing::Point(4, 80);
            this->fcgLBMKVMuxerPath->Name = L"fcgLBMKVMuxerPath";
            this->fcgLBMKVMuxerPath->Size = System::Drawing::Size(49, 14);
            this->fcgLBMKVMuxerPath->TabIndex = 19;
            this->fcgLBMKVMuxerPath->Text = L"～の指定";
            // 
            // fcgCXMKVCmdEx
            // 
            this->fcgCXMKVCmdEx->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXMKVCmdEx->FormattingEnabled = true;
            this->fcgCXMKVCmdEx->Location = System::Drawing::Point(213, 43);
            this->fcgCXMKVCmdEx->Name = L"fcgCXMKVCmdEx";
            this->fcgCXMKVCmdEx->Size = System::Drawing::Size(157, 22);
            this->fcgCXMKVCmdEx->TabIndex = 1;
            this->fcgCXMKVCmdEx->Tag = L"chValue";
            // 
            // fcgLBMKVMuxerCmdEx
            // 
            this->fcgLBMKVMuxerCmdEx->AutoSize = true;
            this->fcgLBMKVMuxerCmdEx->Location = System::Drawing::Point(139, 46);
            this->fcgLBMKVMuxerCmdEx->Name = L"fcgLBMKVMuxerCmdEx";
            this->fcgLBMKVMuxerCmdEx->Size = System::Drawing::Size(68, 14);
            this->fcgLBMKVMuxerCmdEx->TabIndex = 17;
            this->fcgLBMKVMuxerCmdEx->Text = L"拡張オプション";
            // 
            // fcgCBMKVMuxerExt
            // 
            this->fcgCBMKVMuxerExt->AutoSize = true;
            this->fcgCBMKVMuxerExt->Location = System::Drawing::Point(10, 45);
            this->fcgCBMKVMuxerExt->Name = L"fcgCBMKVMuxerExt";
            this->fcgCBMKVMuxerExt->Size = System::Drawing::Size(113, 18);
            this->fcgCBMKVMuxerExt->TabIndex = 0;
            this->fcgCBMKVMuxerExt->Tag = L"chValue";
            this->fcgCBMKVMuxerExt->Text = L"外部muxerを使用";
            this->fcgCBMKVMuxerExt->UseVisualStyleBackColor = true;
            // 
            // fcgtabPageMPG
            // 
            this->fcgtabPageMPG->Controls->Add(this->fcgBTMPGMuxerPath);
            this->fcgtabPageMPG->Controls->Add(this->fcgTXMPGMuxerPath);
            this->fcgtabPageMPG->Controls->Add(this->fcgLBMPGMuxerPath);
            this->fcgtabPageMPG->Controls->Add(this->fcgCXMPGCmdEx);
            this->fcgtabPageMPG->Controls->Add(this->label3);
            this->fcgtabPageMPG->Controls->Add(this->fcgCBMPGMuxerExt);
            this->fcgtabPageMPG->Location = System::Drawing::Point(4, 23);
            this->fcgtabPageMPG->Name = L"fcgtabPageMPG";
            this->fcgtabPageMPG->Size = System::Drawing::Size(376, 174);
            this->fcgtabPageMPG->TabIndex = 4;
            this->fcgtabPageMPG->Text = L"mpg";
            this->fcgtabPageMPG->UseVisualStyleBackColor = true;
            // 
            // fcgBTMPGMuxerPath
            // 
            this->fcgBTMPGMuxerPath->Location = System::Drawing::Point(341, 92);
            this->fcgBTMPGMuxerPath->Name = L"fcgBTMPGMuxerPath";
            this->fcgBTMPGMuxerPath->Size = System::Drawing::Size(30, 23);
            this->fcgBTMPGMuxerPath->TabIndex = 23;
            this->fcgBTMPGMuxerPath->Text = L"...";
            this->fcgBTMPGMuxerPath->UseVisualStyleBackColor = true;
            this->fcgBTMPGMuxerPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTMPGMuxerPath_Click);
            // 
            // fcgTXMPGMuxerPath
            // 
            this->fcgTXMPGMuxerPath->Location = System::Drawing::Point(132, 93);
            this->fcgTXMPGMuxerPath->Name = L"fcgTXMPGMuxerPath";
            this->fcgTXMPGMuxerPath->Size = System::Drawing::Size(207, 21);
            this->fcgTXMPGMuxerPath->TabIndex = 22;
            this->fcgTXMPGMuxerPath->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXMPGMuxerPath_TextChanged);
            this->fcgTXMPGMuxerPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
            this->fcgTXMPGMuxerPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
            // 
            // fcgLBMPGMuxerPath
            // 
            this->fcgLBMPGMuxerPath->AutoSize = true;
            this->fcgLBMPGMuxerPath->Location = System::Drawing::Point(5, 96);
            this->fcgLBMPGMuxerPath->Name = L"fcgLBMPGMuxerPath";
            this->fcgLBMPGMuxerPath->Size = System::Drawing::Size(49, 14);
            this->fcgLBMPGMuxerPath->TabIndex = 25;
            this->fcgLBMPGMuxerPath->Text = L"～の指定";
            // 
            // fcgCXMPGCmdEx
            // 
            this->fcgCXMPGCmdEx->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXMPGCmdEx->FormattingEnabled = true;
            this->fcgCXMPGCmdEx->Location = System::Drawing::Point(214, 59);
            this->fcgCXMPGCmdEx->Name = L"fcgCXMPGCmdEx";
            this->fcgCXMPGCmdEx->Size = System::Drawing::Size(157, 22);
            this->fcgCXMPGCmdEx->TabIndex = 21;
            this->fcgCXMPGCmdEx->Tag = L"chValue";
            // 
            // label3
            // 
            this->label3->AutoSize = true;
            this->label3->Location = System::Drawing::Point(140, 62);
            this->label3->Name = L"label3";
            this->label3->Size = System::Drawing::Size(68, 14);
            this->label3->TabIndex = 24;
            this->label3->Text = L"拡張オプション";
            // 
            // fcgCBMPGMuxerExt
            // 
            this->fcgCBMPGMuxerExt->AutoSize = true;
            this->fcgCBMPGMuxerExt->Location = System::Drawing::Point(11, 61);
            this->fcgCBMPGMuxerExt->Name = L"fcgCBMPGMuxerExt";
            this->fcgCBMPGMuxerExt->Size = System::Drawing::Size(113, 18);
            this->fcgCBMPGMuxerExt->TabIndex = 20;
            this->fcgCBMPGMuxerExt->Tag = L"chValue";
            this->fcgCBMPGMuxerExt->Text = L"外部muxerを使用";
            this->fcgCBMPGMuxerExt->UseVisualStyleBackColor = true;
            // 
            // fcgtabPageMux
            // 
            this->fcgtabPageMux->Controls->Add(this->fcgCXMuxPriority);
            this->fcgtabPageMux->Controls->Add(this->fcgLBMuxPriority);
            this->fcgtabPageMux->Location = System::Drawing::Point(4, 23);
            this->fcgtabPageMux->Name = L"fcgtabPageMux";
            this->fcgtabPageMux->Size = System::Drawing::Size(376, 174);
            this->fcgtabPageMux->TabIndex = 2;
            this->fcgtabPageMux->Text = L"Mux共通設定";
            this->fcgtabPageMux->UseVisualStyleBackColor = true;
            // 
            // fcgCXMuxPriority
            // 
            this->fcgCXMuxPriority->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXMuxPriority->FormattingEnabled = true;
            this->fcgCXMuxPriority->Location = System::Drawing::Point(102, 64);
            this->fcgCXMuxPriority->Name = L"fcgCXMuxPriority";
            this->fcgCXMuxPriority->Size = System::Drawing::Size(198, 22);
            this->fcgCXMuxPriority->TabIndex = 1;
            this->fcgCXMuxPriority->Tag = L"chValue";
            // 
            // fcgLBMuxPriority
            // 
            this->fcgLBMuxPriority->AutoSize = true;
            this->fcgLBMuxPriority->Location = System::Drawing::Point(15, 67);
            this->fcgLBMuxPriority->Name = L"fcgLBMuxPriority";
            this->fcgLBMuxPriority->Size = System::Drawing::Size(62, 14);
            this->fcgLBMuxPriority->TabIndex = 1;
            this->fcgLBMuxPriority->Text = L"Mux優先度";
            // 
            // fcgtabPageBat
            // 
            this->fcgtabPageBat->Controls->Add(this->fcgLBBatAfterString);
            this->fcgtabPageBat->Controls->Add(this->fcgLBBatBeforeString);
            this->fcgtabPageBat->Controls->Add(this->fcgPNSeparator);
            this->fcgtabPageBat->Controls->Add(this->fcgBTBatBeforePath);
            this->fcgtabPageBat->Controls->Add(this->fcgTXBatBeforePath);
            this->fcgtabPageBat->Controls->Add(this->fcgLBBatBeforePath);
            this->fcgtabPageBat->Controls->Add(this->fcgCBWaitForBatBefore);
            this->fcgtabPageBat->Controls->Add(this->fcgCBRunBatBefore);
            this->fcgtabPageBat->Controls->Add(this->fcgBTBatAfterPath);
            this->fcgtabPageBat->Controls->Add(this->fcgTXBatAfterPath);
            this->fcgtabPageBat->Controls->Add(this->fcgLBBatAfterPath);
            this->fcgtabPageBat->Controls->Add(this->fcgCBWaitForBatAfter);
            this->fcgtabPageBat->Controls->Add(this->fcgCBRunBatAfter);
            this->fcgtabPageBat->Location = System::Drawing::Point(4, 23);
            this->fcgtabPageBat->Name = L"fcgtabPageBat";
            this->fcgtabPageBat->Size = System::Drawing::Size(376, 174);
            this->fcgtabPageBat->TabIndex = 3;
            this->fcgtabPageBat->Text = L"エンコ前後バッチ処理";
            this->fcgtabPageBat->UseVisualStyleBackColor = true;
            // 
            // fcgLBBatAfterString
            // 
            this->fcgLBBatAfterString->AutoSize = true;
            this->fcgLBBatAfterString->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Italic | System::Drawing::FontStyle::Underline)),
                System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(128)));
            this->fcgLBBatAfterString->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
            this->fcgLBBatAfterString->Location = System::Drawing::Point(304, 112);
            this->fcgLBBatAfterString->Name = L"fcgLBBatAfterString";
            this->fcgLBBatAfterString->Size = System::Drawing::Size(27, 15);
            this->fcgLBBatAfterString->TabIndex = 20;
            this->fcgLBBatAfterString->Text = L" 後& ";
            this->fcgLBBatAfterString->TextAlign = System::Drawing::ContentAlignment::TopCenter;
            // 
            // fcgLBBatBeforeString
            // 
            this->fcgLBBatBeforeString->AutoSize = true;
            this->fcgLBBatBeforeString->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Italic | System::Drawing::FontStyle::Underline)),
                System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(128)));
            this->fcgLBBatBeforeString->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
            this->fcgLBBatBeforeString->Location = System::Drawing::Point(304, 14);
            this->fcgLBBatBeforeString->Name = L"fcgLBBatBeforeString";
            this->fcgLBBatBeforeString->Size = System::Drawing::Size(27, 15);
            this->fcgLBBatBeforeString->TabIndex = 19;
            this->fcgLBBatBeforeString->Text = L" 前& ";
            this->fcgLBBatBeforeString->TextAlign = System::Drawing::ContentAlignment::TopCenter;
            // 
            // fcgPNSeparator
            // 
            this->fcgPNSeparator->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
            this->fcgPNSeparator->Location = System::Drawing::Point(18, 88);
            this->fcgPNSeparator->Name = L"fcgPNSeparator";
            this->fcgPNSeparator->Size = System::Drawing::Size(342, 1);
            this->fcgPNSeparator->TabIndex = 18;
            // 
            // fcgBTBatBeforePath
            // 
            this->fcgBTBatBeforePath->Location = System::Drawing::Point(330, 55);
            this->fcgBTBatBeforePath->Name = L"fcgBTBatBeforePath";
            this->fcgBTBatBeforePath->Size = System::Drawing::Size(30, 23);
            this->fcgBTBatBeforePath->TabIndex = 17;
            this->fcgBTBatBeforePath->Tag = L"chValue";
            this->fcgBTBatBeforePath->Text = L"...";
            this->fcgBTBatBeforePath->UseVisualStyleBackColor = true;
            this->fcgBTBatBeforePath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTBatBeforePath_Click);
            // 
            // fcgTXBatBeforePath
            // 
            this->fcgTXBatBeforePath->AllowDrop = true;
            this->fcgTXBatBeforePath->Location = System::Drawing::Point(126, 56);
            this->fcgTXBatBeforePath->Name = L"fcgTXBatBeforePath";
            this->fcgTXBatBeforePath->Size = System::Drawing::Size(202, 21);
            this->fcgTXBatBeforePath->TabIndex = 16;
            this->fcgTXBatBeforePath->Tag = L"chValue";
            this->fcgTXBatBeforePath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
            this->fcgTXBatBeforePath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
            // 
            // fcgLBBatBeforePath
            // 
            this->fcgLBBatBeforePath->AutoSize = true;
            this->fcgLBBatBeforePath->Location = System::Drawing::Point(40, 59);
            this->fcgLBBatBeforePath->Name = L"fcgLBBatBeforePath";
            this->fcgLBBatBeforePath->Size = System::Drawing::Size(61, 14);
            this->fcgLBBatBeforePath->TabIndex = 15;
            this->fcgLBBatBeforePath->Text = L"バッチファイル";
            // 
            // fcgCBWaitForBatBefore
            // 
            this->fcgCBWaitForBatBefore->AutoSize = true;
            this->fcgCBWaitForBatBefore->Location = System::Drawing::Point(40, 30);
            this->fcgCBWaitForBatBefore->Name = L"fcgCBWaitForBatBefore";
            this->fcgCBWaitForBatBefore->Size = System::Drawing::Size(150, 18);
            this->fcgCBWaitForBatBefore->TabIndex = 14;
            this->fcgCBWaitForBatBefore->Tag = L"chValue";
            this->fcgCBWaitForBatBefore->Text = L"バッチ処理の終了を待機する";
            this->fcgCBWaitForBatBefore->UseVisualStyleBackColor = true;
            // 
            // fcgCBRunBatBefore
            // 
            this->fcgCBRunBatBefore->AutoSize = true;
            this->fcgCBRunBatBefore->Location = System::Drawing::Point(18, 6);
            this->fcgCBRunBatBefore->Name = L"fcgCBRunBatBefore";
            this->fcgCBRunBatBefore->Size = System::Drawing::Size(179, 18);
            this->fcgCBRunBatBefore->TabIndex = 13;
            this->fcgCBRunBatBefore->Tag = L"chValue";
            this->fcgCBRunBatBefore->Text = L"エンコード開始前、バッチ処理を行う";
            this->fcgCBRunBatBefore->UseVisualStyleBackColor = true;
            // 
            // fcgBTBatAfterPath
            // 
            this->fcgBTBatAfterPath->Location = System::Drawing::Point(330, 146);
            this->fcgBTBatAfterPath->Name = L"fcgBTBatAfterPath";
            this->fcgBTBatAfterPath->Size = System::Drawing::Size(30, 23);
            this->fcgBTBatAfterPath->TabIndex = 24;
            this->fcgBTBatAfterPath->Tag = L"chValue";
            this->fcgBTBatAfterPath->Text = L"...";
            this->fcgBTBatAfterPath->UseVisualStyleBackColor = true;
            this->fcgBTBatAfterPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTBatAfterPath_Click);
            // 
            // fcgTXBatAfterPath
            // 
            this->fcgTXBatAfterPath->AllowDrop = true;
            this->fcgTXBatAfterPath->Location = System::Drawing::Point(126, 147);
            this->fcgTXBatAfterPath->Name = L"fcgTXBatAfterPath";
            this->fcgTXBatAfterPath->Size = System::Drawing::Size(202, 21);
            this->fcgTXBatAfterPath->TabIndex = 23;
            this->fcgTXBatAfterPath->Tag = L"chValue";
            this->fcgTXBatAfterPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
            this->fcgTXBatAfterPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
            // 
            // fcgLBBatAfterPath
            // 
            this->fcgLBBatAfterPath->AutoSize = true;
            this->fcgLBBatAfterPath->Location = System::Drawing::Point(40, 150);
            this->fcgLBBatAfterPath->Name = L"fcgLBBatAfterPath";
            this->fcgLBBatAfterPath->Size = System::Drawing::Size(61, 14);
            this->fcgLBBatAfterPath->TabIndex = 22;
            this->fcgLBBatAfterPath->Text = L"バッチファイル";
            // 
            // fcgCBWaitForBatAfter
            // 
            this->fcgCBWaitForBatAfter->AutoSize = true;
            this->fcgCBWaitForBatAfter->Location = System::Drawing::Point(40, 122);
            this->fcgCBWaitForBatAfter->Name = L"fcgCBWaitForBatAfter";
            this->fcgCBWaitForBatAfter->Size = System::Drawing::Size(150, 18);
            this->fcgCBWaitForBatAfter->TabIndex = 21;
            this->fcgCBWaitForBatAfter->Tag = L"chValue";
            this->fcgCBWaitForBatAfter->Text = L"バッチ処理の終了を待機する";
            this->fcgCBWaitForBatAfter->UseVisualStyleBackColor = true;
            // 
            // fcgCBRunBatAfter
            // 
            this->fcgCBRunBatAfter->AutoSize = true;
            this->fcgCBRunBatAfter->Location = System::Drawing::Point(18, 98);
            this->fcgCBRunBatAfter->Name = L"fcgCBRunBatAfter";
            this->fcgCBRunBatAfter->Size = System::Drawing::Size(179, 18);
            this->fcgCBRunBatAfter->TabIndex = 20;
            this->fcgCBRunBatAfter->Tag = L"chValue";
            this->fcgCBRunBatAfter->Text = L"エンコード終了後、バッチ処理を行う";
            this->fcgCBRunBatAfter->UseVisualStyleBackColor = true;
            // 
            // fcgtabPageMuxInternal
            // 
            this->fcgtabPageMuxInternal->Controls->Add(this->fcgCBCopyChapter);
            this->fcgtabPageMuxInternal->Controls->Add(this->fcgCBCopySubtitle);
            this->fcgtabPageMuxInternal->Location = System::Drawing::Point(4, 23);
            this->fcgtabPageMuxInternal->Name = L"fcgtabPageMuxInternal";
            this->fcgtabPageMuxInternal->Size = System::Drawing::Size(376, 174);
            this->fcgtabPageMuxInternal->TabIndex = 5;
            this->fcgtabPageMuxInternal->Text = L"内部muxer";
            this->fcgtabPageMuxInternal->UseVisualStyleBackColor = true;
            // 
            // fcgCBCopyChapter
            // 
            this->fcgCBCopyChapter->AutoSize = true;
            this->fcgCBCopyChapter->Location = System::Drawing::Point(15, 46);
            this->fcgCBCopyChapter->Name = L"fcgCBCopyChapter";
            this->fcgCBCopyChapter->Size = System::Drawing::Size(100, 18);
            this->fcgCBCopyChapter->TabIndex = 2;
            this->fcgCBCopyChapter->Tag = L"chValue";
            this->fcgCBCopyChapter->Text = L"チャプターをコピー";
            this->fcgCBCopyChapter->UseVisualStyleBackColor = true;
            // 
            // fcgCBCopySubtitle
            // 
            this->fcgCBCopySubtitle->AutoSize = true;
            this->fcgCBCopySubtitle->Location = System::Drawing::Point(15, 19);
            this->fcgCBCopySubtitle->Name = L"fcgCBCopySubtitle";
            this->fcgCBCopySubtitle->Size = System::Drawing::Size(81, 18);
            this->fcgCBCopySubtitle->TabIndex = 1;
            this->fcgCBCopySubtitle->Tag = L"chValue";
            this->fcgCBCopySubtitle->Text = L"字幕をコピー";
            this->fcgCBCopySubtitle->UseVisualStyleBackColor = true;
            // 
            // fcgTXCmd
            // 
            this->fcgTXCmd->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                | System::Windows::Forms::AnchorStyles::Left)
                | System::Windows::Forms::AnchorStyles::Right));
            this->fcgTXCmd->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgTXCmd->Location = System::Drawing::Point(9, 512);
            this->fcgTXCmd->Name = L"fcgTXCmd";
            this->fcgTXCmd->ReadOnly = true;
            this->fcgTXCmd->Size = System::Drawing::Size(992, 21);
            this->fcgTXCmd->TabIndex = 4;
            this->fcgTXCmd->Visible = false;
            this->fcgTXCmd->DoubleClick += gcnew System::EventHandler(this, &frmConfig::fcgTXCmd_DoubleClick);
            // 
            // fcgBTCancel
            // 
            this->fcgBTCancel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
            this->fcgBTCancel->Location = System::Drawing::Point(771, 546);
            this->fcgBTCancel->Name = L"fcgBTCancel";
            this->fcgBTCancel->Size = System::Drawing::Size(84, 28);
            this->fcgBTCancel->TabIndex = 5;
            this->fcgBTCancel->Text = L"キャンセル";
            this->fcgBTCancel->UseVisualStyleBackColor = true;
            this->fcgBTCancel->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTCancel_Click);
            // 
            // fcgBTOK
            // 
            this->fcgBTOK->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
            this->fcgBTOK->Location = System::Drawing::Point(893, 546);
            this->fcgBTOK->Name = L"fcgBTOK";
            this->fcgBTOK->Size = System::Drawing::Size(84, 28);
            this->fcgBTOK->TabIndex = 6;
            this->fcgBTOK->Text = L"OK";
            this->fcgBTOK->UseVisualStyleBackColor = true;
            this->fcgBTOK->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTOK_Click);
            // 
            // fcgBTDefault
            // 
            this->fcgBTDefault->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
            this->fcgBTDefault->Location = System::Drawing::Point(9, 548);
            this->fcgBTDefault->Name = L"fcgBTDefault";
            this->fcgBTDefault->Size = System::Drawing::Size(112, 28);
            this->fcgBTDefault->TabIndex = 7;
            this->fcgBTDefault->Text = L"デフォルト";
            this->fcgBTDefault->UseVisualStyleBackColor = true;
            this->fcgBTDefault->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTDefault_Click);
            // 
            // fcgLBVersionDate
            // 
            this->fcgLBVersionDate->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
            this->fcgLBVersionDate->AutoSize = true;
            this->fcgLBVersionDate->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgLBVersionDate->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
            this->fcgLBVersionDate->Location = System::Drawing::Point(490, 555);
            this->fcgLBVersionDate->Name = L"fcgLBVersionDate";
            this->fcgLBVersionDate->Size = System::Drawing::Size(47, 14);
            this->fcgLBVersionDate->TabIndex = 8;
            this->fcgLBVersionDate->Text = L"Version";
            // 
            // fcgLBVersion
            // 
            this->fcgLBVersion->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
            this->fcgLBVersion->AutoSize = true;
            this->fcgLBVersion->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgLBVersion->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
            this->fcgLBVersion->Location = System::Drawing::Point(131, 555);
            this->fcgLBVersion->Name = L"fcgLBVersion";
            this->fcgLBVersion->Size = System::Drawing::Size(47, 14);
            this->fcgLBVersion->TabIndex = 9;
            this->fcgLBVersion->Text = L"Version";
            // 
            // fcgOpenFileDialog
            // 
            this->fcgOpenFileDialog->FileName = L"openFileDialog1";
            // 
            // fcgTTEx
            // 
            this->fcgTTEx->AutomaticDelay = 200;
            this->fcgTTEx->AutoPopDelay = 9999;
            this->fcgTTEx->InitialDelay = 200;
            this->fcgTTEx->IsBalloon = true;
            this->fcgTTEx->ReshowDelay = 50;
            this->fcgTTEx->ShowAlways = true;
            this->fcgTTEx->UseAnimation = false;
            this->fcgTTEx->UseFading = false;
            // 
            // fcgtabControlQSV
            // 
            this->fcgtabControlQSV->Controls->Add(this->tabPageVideoEnc);
            this->fcgtabControlQSV->Controls->Add(this->tabPageVpp);
            this->fcgtabControlQSV->Controls->Add(this->tabPageExOpt);
            this->fcgtabControlQSV->Controls->Add(this->tabPageFeatures);
            this->fcgtabControlQSV->Location = System::Drawing::Point(4, 31);
            this->fcgtabControlQSV->Name = L"fcgtabControlQSV";
            this->fcgtabControlQSV->SelectedIndex = 0;
            this->fcgtabControlQSV->Size = System::Drawing::Size(616, 509);
            this->fcgtabControlQSV->TabIndex = 49;
            // 
            // tabPageVideoEnc
            // 
            this->tabPageVideoEnc->Controls->Add(this->fcgLBMFXLibDetectionHwValue);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBMFXLibDetectionHwStatus);
            this->tabPageVideoEnc->Controls->Add(this->fcgCBFadeDetect);
            this->tabPageVideoEnc->Controls->Add(this->fcgCBWeightB);
            this->tabPageVideoEnc->Controls->Add(this->fcgCBWeightP);
            this->tabPageVideoEnc->Controls->Add(this->fcgCBFixedFunc);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBWinBRCSizeAuto);
            this->tabPageVideoEnc->Controls->Add(this->fcgNUWinBRCSize);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBWinBRCSize);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBQPMinMaxAuto);
            this->tabPageVideoEnc->Controls->Add(this->fcgNUQPMax);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBQPMax);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBQPMinMAX);
            this->tabPageVideoEnc->Controls->Add(this->fcgNUQPMin);
            this->tabPageVideoEnc->Controls->Add(this->fcgPNQVBR);
            this->tabPageVideoEnc->Controls->Add(this->fcgPNICQ);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXLookaheadDS);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBLookaheadDS);
            this->tabPageVideoEnc->Controls->Add(this->fcgCBBPyramid);
            this->tabPageVideoEnc->Controls->Add(this->fcgCBAdaptiveB);
            this->tabPageVideoEnc->Controls->Add(this->fcgCBAdaptiveI);
            this->tabPageVideoEnc->Controls->Add(this->fcgPNLookahead);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBFullrange);
            this->tabPageVideoEnc->Controls->Add(this->fcgCBFullrange);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBBlurayCompat);
            this->tabPageVideoEnc->Controls->Add(this->fcgCBBlurayCompat);
            this->tabPageVideoEnc->Controls->Add(this->fcgPNAVBR);
            this->tabPageVideoEnc->Controls->Add(this->fcgPNBitrate);
            this->tabPageVideoEnc->Controls->Add(this->fcgPNQP);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBMFXLibDetectionSwValue);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBMFXLibDetectionSwStatus);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBMFXLibDetection);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBSlices2);
            this->tabPageVideoEnc->Controls->Add(this->fcgNUSlices);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBSlices);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBRefAuto);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBGOPLengthAuto);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBBframesAuto);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXVideoFormat);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBVideoFormat);
            this->tabPageVideoEnc->Controls->Add(this->fcggroupBoxColor);
            this->tabPageVideoEnc->Controls->Add(this->fcgCBOpenGOP);
            this->tabPageVideoEnc->Controls->Add(this->fcgCBSceneChange);
            this->tabPageVideoEnc->Controls->Add(this->fcgGroupBoxAspectRatio);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXInterlaced);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBInterlaced);
            this->tabPageVideoEnc->Controls->Add(this->fcgNUGopLength);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBGOPLength);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXCodecLevel);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXCodecProfile);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBCodecLevel);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBCodecProfile);
            this->tabPageVideoEnc->Controls->Add(this->fcgNUBframes);
            this->tabPageVideoEnc->Controls->Add(this->fcgNURef);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBBframes);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBRef);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBEncMode);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXEncMode);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBQuality);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBOutputType);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXQuality);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXOutputType);
            this->tabPageVideoEnc->Controls->Add(this->fcgCBHWEncode);
            this->tabPageVideoEnc->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->tabPageVideoEnc->Location = System::Drawing::Point(4, 24);
            this->tabPageVideoEnc->Name = L"tabPageVideoEnc";
            this->tabPageVideoEnc->Padding = System::Windows::Forms::Padding(3);
            this->tabPageVideoEnc->Size = System::Drawing::Size(608, 481);
            this->tabPageVideoEnc->TabIndex = 0;
            this->tabPageVideoEnc->Text = L"動画エンコード";
            this->tabPageVideoEnc->UseVisualStyleBackColor = true;
            // 
            // fcgCBFadeDetect
            // 
            this->fcgCBFadeDetect->AutoSize = true;
            this->fcgCBFadeDetect->Location = System::Drawing::Point(39, 365);
            this->fcgCBFadeDetect->Name = L"fcgCBFadeDetect";
            this->fcgCBFadeDetect->Size = System::Drawing::Size(79, 18);
            this->fcgCBFadeDetect->TabIndex = 157;
            this->fcgCBFadeDetect->Tag = L"chValue";
            this->fcgCBFadeDetect->Text = L"フェード検出";
            this->fcgCBFadeDetect->UseVisualStyleBackColor = true;
            // 
            // fcgCBWeightB
            // 
            this->fcgCBWeightB->AutoSize = true;
            this->fcgCBWeightB->Location = System::Drawing::Point(195, 343);
            this->fcgCBWeightB->Name = L"fcgCBWeightB";
            this->fcgCBWeightB->Size = System::Drawing::Size(106, 18);
            this->fcgCBWeightB->TabIndex = 156;
            this->fcgCBWeightB->Tag = L"chValue";
            this->fcgCBWeightB->Text = L"重み付きBフレーム";
            this->fcgCBWeightB->UseVisualStyleBackColor = true;
            // 
            // fcgCBWeightP
            // 
            this->fcgCBWeightP->AutoSize = true;
            this->fcgCBWeightP->Location = System::Drawing::Point(39, 343);
            this->fcgCBWeightP->Name = L"fcgCBWeightP";
            this->fcgCBWeightP->Size = System::Drawing::Size(106, 18);
            this->fcgCBWeightP->TabIndex = 155;
            this->fcgCBWeightP->Tag = L"chValue";
            this->fcgCBWeightP->Text = L"重み付きPフレーム";
            this->fcgCBWeightP->UseVisualStyleBackColor = true;
            // 
            // fcgCBFixedFunc
            // 
            this->fcgCBFixedFunc->AutoSize = true;
            this->fcgCBFixedFunc->Location = System::Drawing::Point(242, 11);
            this->fcgCBFixedFunc->Name = L"fcgCBFixedFunc";
            this->fcgCBFixedFunc->Size = System::Drawing::Size(80, 18);
            this->fcgCBFixedFunc->TabIndex = 154;
            this->fcgCBFixedFunc->Tag = L"chValue";
            this->fcgCBFixedFunc->Text = L"FixedFunc";
            this->fcgCBFixedFunc->UseVisualStyleBackColor = true;
            // 
            // fcgLBWinBRCSizeAuto
            // 
            this->fcgLBWinBRCSizeAuto->AutoSize = true;
            this->fcgLBWinBRCSizeAuto->Location = System::Drawing::Point(212, 422);
            this->fcgLBWinBRCSizeAuto->Name = L"fcgLBWinBRCSizeAuto";
            this->fcgLBWinBRCSizeAuto->Size = System::Drawing::Size(107, 14);
            this->fcgLBWinBRCSizeAuto->TabIndex = 153;
            this->fcgLBWinBRCSizeAuto->Text = L"フレーム  ※\"0\"で自動";
            // 
            // fcgNUWinBRCSize
            // 
            this->fcgNUWinBRCSize->Location = System::Drawing::Point(132, 420);
            this->fcgNUWinBRCSize->Name = L"fcgNUWinBRCSize";
            this->fcgNUWinBRCSize->Size = System::Drawing::Size(77, 21);
            this->fcgNUWinBRCSize->TabIndex = 152;
            this->fcgNUWinBRCSize->Tag = L"chValue";
            this->fcgNUWinBRCSize->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBWinBRCSize
            // 
            this->fcgLBWinBRCSize->AutoSize = true;
            this->fcgLBWinBRCSize->Location = System::Drawing::Point(16, 422);
            this->fcgLBWinBRCSize->Name = L"fcgLBWinBRCSize";
            this->fcgLBWinBRCSize->Size = System::Drawing::Size(103, 14);
            this->fcgLBWinBRCSize->TabIndex = 151;
            this->fcgLBWinBRCSize->Text = L"レート制御ウィンドウ幅";
            // 
            // fcgLBQPMinMaxAuto
            // 
            this->fcgLBQPMinMaxAuto->AutoSize = true;
            this->fcgLBQPMinMaxAuto->Location = System::Drawing::Point(247, 452);
            this->fcgLBQPMinMaxAuto->Name = L"fcgLBQPMinMaxAuto";
            this->fcgLBQPMinMaxAuto->Size = System::Drawing::Size(66, 14);
            this->fcgLBQPMinMaxAuto->TabIndex = 150;
            this->fcgLBQPMinMaxAuto->Text = L"※\"0\"で自動";
            // 
            // fcgNUQPMax
            // 
            this->fcgNUQPMax->Location = System::Drawing::Point(179, 450);
            this->fcgNUQPMax->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPMax->Name = L"fcgNUQPMax";
            this->fcgNUQPMax->Size = System::Drawing::Size(58, 21);
            this->fcgNUQPMax->TabIndex = 149;
            this->fcgNUQPMax->Tag = L"chValue";
            this->fcgNUQPMax->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBQPMax
            // 
            this->fcgLBQPMax->AutoSize = true;
            this->fcgLBQPMax->Location = System::Drawing::Point(144, 452);
            this->fcgLBQPMax->Name = L"fcgLBQPMax";
            this->fcgLBQPMax->Size = System::Drawing::Size(29, 14);
            this->fcgLBQPMax->TabIndex = 148;
            this->fcgLBQPMax->Text = L"最大";
            // 
            // fcgLBQPMinMAX
            // 
            this->fcgLBQPMinMAX->AutoSize = true;
            this->fcgLBQPMinMAX->Location = System::Drawing::Point(15, 452);
            this->fcgLBQPMinMAX->Name = L"fcgLBQPMinMAX";
            this->fcgLBQPMinMAX->Size = System::Drawing::Size(48, 14);
            this->fcgLBQPMinMAX->TabIndex = 147;
            this->fcgLBQPMinMAX->Text = L"QP 最小";
            // 
            // fcgNUQPMin
            // 
            this->fcgNUQPMin->Location = System::Drawing::Point(70, 450);
            this->fcgNUQPMin->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPMin->Name = L"fcgNUQPMin";
            this->fcgNUQPMin->Size = System::Drawing::Size(58, 21);
            this->fcgNUQPMin->TabIndex = 146;
            this->fcgNUQPMin->Tag = L"chValue";
            this->fcgNUQPMin->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgPNQVBR
            // 
            this->fcgPNQVBR->Controls->Add(this->fcgNUQVBR);
            this->fcgPNQVBR->Controls->Add(this->fcgLBQVBR);
            this->fcgPNQVBR->Location = System::Drawing::Point(8, 181);
            this->fcgPNQVBR->Name = L"fcgPNQVBR";
            this->fcgPNQVBR->Size = System::Drawing::Size(289, 26);
            this->fcgPNQVBR->TabIndex = 145;
            // 
            // fcgNUQVBR
            // 
            this->fcgNUQVBR->Location = System::Drawing::Point(125, 3);
            this->fcgNUQVBR->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQVBR->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->fcgNUQVBR->Name = L"fcgNUQVBR";
            this->fcgNUQVBR->Size = System::Drawing::Size(77, 21);
            this->fcgNUQVBR->TabIndex = 95;
            this->fcgNUQVBR->Tag = L"chValue";
            this->fcgNUQVBR->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUQVBR->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            // 
            // fcgLBQVBR
            // 
            this->fcgLBQVBR->AutoSize = true;
            this->fcgLBQVBR->Location = System::Drawing::Point(6, 6);
            this->fcgLBQVBR->Name = L"fcgLBQVBR";
            this->fcgLBQVBR->Size = System::Drawing::Size(71, 14);
            this->fcgLBQVBR->TabIndex = 96;
            this->fcgLBQVBR->Text = L"固定品質の値";
            // 
            // fcgPNICQ
            // 
            this->fcgPNICQ->Controls->Add(this->fcgNUICQQuality);
            this->fcgPNICQ->Controls->Add(this->fcgLBICQQuality);
            this->fcgPNICQ->Location = System::Drawing::Point(8, 128);
            this->fcgPNICQ->Name = L"fcgPNICQ";
            this->fcgPNICQ->Size = System::Drawing::Size(289, 26);
            this->fcgPNICQ->TabIndex = 144;
            // 
            // fcgNUICQQuality
            // 
            this->fcgNUICQQuality->Location = System::Drawing::Point(125, 3);
            this->fcgNUICQQuality->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUICQQuality->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->fcgNUICQQuality->Name = L"fcgNUICQQuality";
            this->fcgNUICQQuality->Size = System::Drawing::Size(77, 21);
            this->fcgNUICQQuality->TabIndex = 95;
            this->fcgNUICQQuality->Tag = L"chValue";
            this->fcgNUICQQuality->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUICQQuality->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            // 
            // fcgLBICQQuality
            // 
            this->fcgLBICQQuality->AutoSize = true;
            this->fcgLBICQQuality->Location = System::Drawing::Point(6, 6);
            this->fcgLBICQQuality->Name = L"fcgLBICQQuality";
            this->fcgLBICQQuality->Size = System::Drawing::Size(71, 14);
            this->fcgLBICQQuality->TabIndex = 96;
            this->fcgLBICQQuality->Text = L"固定品質の値";
            // 
            // fcgCXLookaheadDS
            // 
            this->fcgCXLookaheadDS->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXLookaheadDS->FormattingEnabled = true;
            this->fcgCXLookaheadDS->Location = System::Drawing::Point(132, 392);
            this->fcgCXLookaheadDS->Name = L"fcgCXLookaheadDS";
            this->fcgCXLookaheadDS->Size = System::Drawing::Size(121, 22);
            this->fcgCXLookaheadDS->TabIndex = 142;
            this->fcgCXLookaheadDS->Tag = L"chValue";
            // 
            // fcgLBLookaheadDS
            // 
            this->fcgLBLookaheadDS->AutoSize = true;
            this->fcgLBLookaheadDS->Location = System::Drawing::Point(14, 395);
            this->fcgLBLookaheadDS->Name = L"fcgLBLookaheadDS";
            this->fcgLBLookaheadDS->Size = System::Drawing::Size(73, 14);
            this->fcgLBLookaheadDS->TabIndex = 143;
            this->fcgLBLookaheadDS->Text = L"先行探索品質";
            // 
            // fcgCBBPyramid
            // 
            this->fcgCBBPyramid->AutoSize = true;
            this->fcgCBBPyramid->Location = System::Drawing::Point(195, 296);
            this->fcgCBBPyramid->Name = L"fcgCBBPyramid";
            this->fcgCBBPyramid->Size = System::Drawing::Size(85, 18);
            this->fcgCBBPyramid->TabIndex = 139;
            this->fcgCBBPyramid->Tag = L"chValue";
            this->fcgCBBPyramid->Text = L"ピラミッド参照";
            this->fcgCBBPyramid->UseVisualStyleBackColor = true;
            // 
            // fcgCBAdaptiveB
            // 
            this->fcgCBAdaptiveB->AutoSize = true;
            this->fcgCBAdaptiveB->Location = System::Drawing::Point(195, 319);
            this->fcgCBAdaptiveB->Name = L"fcgCBAdaptiveB";
            this->fcgCBAdaptiveB->Size = System::Drawing::Size(121, 18);
            this->fcgCBAdaptiveB->TabIndex = 138;
            this->fcgCBAdaptiveB->Tag = L"chValue";
            this->fcgCBAdaptiveB->Text = L"適応的Bフレーム挿入";
            this->fcgCBAdaptiveB->UseVisualStyleBackColor = true;
            // 
            // fcgCBAdaptiveI
            // 
            this->fcgCBAdaptiveI->AutoSize = true;
            this->fcgCBAdaptiveI->Location = System::Drawing::Point(39, 319);
            this->fcgCBAdaptiveI->Name = L"fcgCBAdaptiveI";
            this->fcgCBAdaptiveI->Size = System::Drawing::Size(118, 18);
            this->fcgCBAdaptiveI->TabIndex = 137;
            this->fcgCBAdaptiveI->Tag = L"chValue";
            this->fcgCBAdaptiveI->Text = L"適応的Iフレーム挿入";
            this->fcgCBAdaptiveI->UseVisualStyleBackColor = true;
            // 
            // fcgPNLookahead
            // 
            this->fcgPNLookahead->Controls->Add(this->label1);
            this->fcgPNLookahead->Controls->Add(this->fcgNULookaheadDepth);
            this->fcgPNLookahead->Controls->Add(this->fcgLBLookaheadDepth);
            this->fcgPNLookahead->Location = System::Drawing::Point(8, 181);
            this->fcgPNLookahead->Name = L"fcgPNLookahead";
            this->fcgPNLookahead->Size = System::Drawing::Size(289, 26);
            this->fcgPNLookahead->TabIndex = 134;
            // 
            // label1
            // 
            this->label1->AutoSize = true;
            this->label1->Location = System::Drawing::Point(206, 6);
            this->label1->Name = L"label1";
            this->label1->Size = System::Drawing::Size(66, 14);
            this->label1->TabIndex = 102;
            this->label1->Text = L"※\"0\"で自動";
            // 
            // fcgNULookaheadDepth
            // 
            this->fcgNULookaheadDepth->Location = System::Drawing::Point(125, 3);
            this->fcgNULookaheadDepth->Name = L"fcgNULookaheadDepth";
            this->fcgNULookaheadDepth->Size = System::Drawing::Size(77, 21);
            this->fcgNULookaheadDepth->TabIndex = 95;
            this->fcgNULookaheadDepth->Tag = L"chValue";
            this->fcgNULookaheadDepth->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBLookaheadDepth
            // 
            this->fcgLBLookaheadDepth->AutoSize = true;
            this->fcgLBLookaheadDepth->Location = System::Drawing::Point(6, 6);
            this->fcgLBLookaheadDepth->Name = L"fcgLBLookaheadDepth";
            this->fcgLBLookaheadDepth->Size = System::Drawing::Size(95, 14);
            this->fcgLBLookaheadDepth->TabIndex = 96;
            this->fcgLBLookaheadDepth->Text = L"先行探索フレーム数";
            // 
            // fcgLBFullrange
            // 
            this->fcgLBFullrange->AutoSize = true;
            this->fcgLBFullrange->Location = System::Drawing::Point(355, 346);
            this->fcgLBFullrange->Name = L"fcgLBFullrange";
            this->fcgLBFullrange->Size = System::Drawing::Size(55, 14);
            this->fcgLBFullrange->TabIndex = 131;
            this->fcgLBFullrange->Text = L"fullrange";
            // 
            // fcgCBFullrange
            // 
            this->fcgCBFullrange->AutoSize = true;
            this->fcgCBFullrange->Location = System::Drawing::Point(454, 349);
            this->fcgCBFullrange->Name = L"fcgCBFullrange";
            this->fcgCBFullrange->Size = System::Drawing::Size(15, 14);
            this->fcgCBFullrange->TabIndex = 37;
            this->fcgCBFullrange->Tag = L"chValue";
            this->fcgCBFullrange->UseVisualStyleBackColor = true;
            // 
            // fcgLBBlurayCompat
            // 
            this->fcgLBBlurayCompat->AutoSize = true;
            this->fcgLBBlurayCompat->Location = System::Drawing::Point(355, 159);
            this->fcgLBBlurayCompat->Name = L"fcgLBBlurayCompat";
            this->fcgLBBlurayCompat->Size = System::Drawing::Size(85, 14);
            this->fcgLBBlurayCompat->TabIndex = 130;
            this->fcgLBBlurayCompat->Text = L"Bluray互換出力";
            // 
            // fcgCBBlurayCompat
            // 
            this->fcgCBBlurayCompat->AutoSize = true;
            this->fcgCBBlurayCompat->Location = System::Drawing::Point(454, 160);
            this->fcgCBBlurayCompat->Name = L"fcgCBBlurayCompat";
            this->fcgCBBlurayCompat->Size = System::Drawing::Size(15, 14);
            this->fcgCBBlurayCompat->TabIndex = 31;
            this->fcgCBBlurayCompat->Tag = L"chValue";
            this->fcgCBBlurayCompat->UseVisualStyleBackColor = true;
            // 
            // fcgPNAVBR
            // 
            this->fcgPNAVBR->Controls->Add(this->fcgLBAVBRConvergence);
            this->fcgPNAVBR->Controls->Add(this->fcgNUAVBRAccuarcy);
            this->fcgPNAVBR->Controls->Add(this->fcgLBAVBRAccuarcy);
            this->fcgPNAVBR->Controls->Add(this->fcgNUAVBRConvergence);
            this->fcgPNAVBR->Controls->Add(this->fcgLBAVBRAccuarcy2);
            this->fcgPNAVBR->Controls->Add(this->fcgLBAVBRConvergence2);
            this->fcgPNAVBR->Location = System::Drawing::Point(8, 156);
            this->fcgPNAVBR->Name = L"fcgPNAVBR";
            this->fcgPNAVBR->Size = System::Drawing::Size(289, 50);
            this->fcgPNAVBR->TabIndex = 115;
            // 
            // fcgLBAVBRConvergence
            // 
            this->fcgLBAVBRConvergence->AutoSize = true;
            this->fcgLBAVBRConvergence->Location = System::Drawing::Point(5, 3);
            this->fcgLBAVBRConvergence->Name = L"fcgLBAVBRConvergence";
            this->fcgLBAVBRConvergence->Size = System::Drawing::Size(74, 14);
            this->fcgLBAVBRConvergence->TabIndex = 91;
            this->fcgLBAVBRConvergence->Text = L"ABVR 計算幅";
            // 
            // fcgNUAVBRAccuarcy
            // 
            this->fcgNUAVBRAccuarcy->DecimalPlaces = 1;
            this->fcgNUAVBRAccuarcy->Location = System::Drawing::Point(124, 26);
            this->fcgNUAVBRAccuarcy->Name = L"fcgNUAVBRAccuarcy";
            this->fcgNUAVBRAccuarcy->Size = System::Drawing::Size(77, 21);
            this->fcgNUAVBRAccuarcy->TabIndex = 11;
            this->fcgNUAVBRAccuarcy->Tag = L"chValue";
            this->fcgNUAVBRAccuarcy->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUAVBRAccuarcy->Visible = false;
            // 
            // fcgLBAVBRAccuarcy
            // 
            this->fcgLBAVBRAccuarcy->AutoSize = true;
            this->fcgLBAVBRAccuarcy->Location = System::Drawing::Point(5, 28);
            this->fcgLBAVBRAccuarcy->Name = L"fcgLBAVBRAccuarcy";
            this->fcgLBAVBRAccuarcy->Size = System::Drawing::Size(74, 14);
            this->fcgLBAVBRAccuarcy->TabIndex = 89;
            this->fcgLBAVBRAccuarcy->Text = L"AVBR 変動幅";
            this->fcgLBAVBRAccuarcy->Visible = false;
            // 
            // fcgNUAVBRConvergence
            // 
            this->fcgNUAVBRConvergence->Location = System::Drawing::Point(124, 1);
            this->fcgNUAVBRConvergence->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 120, 0, 0, 0 });
            this->fcgNUAVBRConvergence->Name = L"fcgNUAVBRConvergence";
            this->fcgNUAVBRConvergence->Size = System::Drawing::Size(77, 21);
            this->fcgNUAVBRConvergence->TabIndex = 10;
            this->fcgNUAVBRConvergence->Tag = L"chValue";
            this->fcgNUAVBRConvergence->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBAVBRAccuarcy2
            // 
            this->fcgLBAVBRAccuarcy2->AutoSize = true;
            this->fcgLBAVBRAccuarcy2->Location = System::Drawing::Point(206, 28);
            this->fcgLBAVBRAccuarcy2->Name = L"fcgLBAVBRAccuarcy2";
            this->fcgLBAVBRAccuarcy2->Size = System::Drawing::Size(18, 14);
            this->fcgLBAVBRAccuarcy2->TabIndex = 93;
            this->fcgLBAVBRAccuarcy2->Text = L"%";
            this->fcgLBAVBRAccuarcy2->Visible = false;
            // 
            // fcgLBAVBRConvergence2
            // 
            this->fcgLBAVBRConvergence2->AutoSize = true;
            this->fcgLBAVBRConvergence2->Location = System::Drawing::Point(206, 3);
            this->fcgLBAVBRConvergence2->Name = L"fcgLBAVBRConvergence2";
            this->fcgLBAVBRConvergence2->Size = System::Drawing::Size(74, 14);
            this->fcgLBAVBRConvergence2->TabIndex = 94;
            this->fcgLBAVBRConvergence2->Text = L"×100frames";
            // 
            // fcgPNBitrate
            // 
            this->fcgPNBitrate->Controls->Add(this->fcgLBBitrate);
            this->fcgPNBitrate->Controls->Add(this->fcgNUBitrate);
            this->fcgPNBitrate->Controls->Add(this->fcgLBBitrate2);
            this->fcgPNBitrate->Controls->Add(this->fcgNUMaxkbps);
            this->fcgPNBitrate->Controls->Add(this->fcgLBMaxkbps);
            this->fcgPNBitrate->Controls->Add(this->fcgLBMaxBitrate2);
            this->fcgPNBitrate->Location = System::Drawing::Point(8, 128);
            this->fcgPNBitrate->Name = L"fcgPNBitrate";
            this->fcgPNBitrate->Size = System::Drawing::Size(289, 54);
            this->fcgPNBitrate->TabIndex = 114;
            // 
            // fcgLBBitrate
            // 
            this->fcgLBBitrate->AutoSize = true;
            this->fcgLBBitrate->Location = System::Drawing::Point(5, 4);
            this->fcgLBBitrate->Name = L"fcgLBBitrate";
            this->fcgLBBitrate->Size = System::Drawing::Size(54, 14);
            this->fcgLBBitrate->TabIndex = 66;
            this->fcgLBBitrate->Text = L"ビットレート";
            // 
            // fcgNUBitrate
            // 
            this->fcgNUBitrate->Location = System::Drawing::Point(124, 2);
            this->fcgNUBitrate->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1000000, 0, 0, 0 });
            this->fcgNUBitrate->Name = L"fcgNUBitrate";
            this->fcgNUBitrate->Size = System::Drawing::Size(77, 21);
            this->fcgNUBitrate->TabIndex = 5;
            this->fcgNUBitrate->Tag = L"chValue";
            this->fcgNUBitrate->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBBitrate2
            // 
            this->fcgLBBitrate2->AutoSize = true;
            this->fcgLBBitrate2->Location = System::Drawing::Point(207, 4);
            this->fcgLBBitrate2->Name = L"fcgLBBitrate2";
            this->fcgLBBitrate2->Size = System::Drawing::Size(32, 14);
            this->fcgLBBitrate2->TabIndex = 69;
            this->fcgLBBitrate2->Text = L"kbps";
            // 
            // fcgNUMaxkbps
            // 
            this->fcgNUMaxkbps->Location = System::Drawing::Point(124, 26);
            this->fcgNUMaxkbps->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1000000, 0, 0, 0 });
            this->fcgNUMaxkbps->Name = L"fcgNUMaxkbps";
            this->fcgNUMaxkbps->Size = System::Drawing::Size(77, 21);
            this->fcgNUMaxkbps->TabIndex = 6;
            this->fcgNUMaxkbps->Tag = L"chValue";
            this->fcgNUMaxkbps->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBMaxkbps
            // 
            this->fcgLBMaxkbps->AutoSize = true;
            this->fcgLBMaxkbps->Location = System::Drawing::Point(5, 33);
            this->fcgLBMaxkbps->Name = L"fcgLBMaxkbps";
            this->fcgLBMaxkbps->Size = System::Drawing::Size(76, 14);
            this->fcgLBMaxkbps->TabIndex = 78;
            this->fcgLBMaxkbps->Text = L"最大ビットレート";
            // 
            // fcgLBMaxBitrate2
            // 
            this->fcgLBMaxBitrate2->AutoSize = true;
            this->fcgLBMaxBitrate2->Location = System::Drawing::Point(207, 31);
            this->fcgLBMaxBitrate2->Name = L"fcgLBMaxBitrate2";
            this->fcgLBMaxBitrate2->Size = System::Drawing::Size(32, 14);
            this->fcgLBMaxBitrate2->TabIndex = 80;
            this->fcgLBMaxBitrate2->Text = L"kbps";
            // 
            // fcgPNQP
            // 
            this->fcgPNQP->Controls->Add(this->fcgLBQPI);
            this->fcgPNQP->Controls->Add(this->fcgNUQPI);
            this->fcgPNQP->Controls->Add(this->fcgNUQPP);
            this->fcgPNQP->Controls->Add(this->fcgNUQPB);
            this->fcgPNQP->Controls->Add(this->fcgLBQPP);
            this->fcgPNQP->Controls->Add(this->fcgLBQPB);
            this->fcgPNQP->Location = System::Drawing::Point(8, 128);
            this->fcgPNQP->Name = L"fcgPNQP";
            this->fcgPNQP->Size = System::Drawing::Size(289, 79);
            this->fcgPNQP->TabIndex = 113;
            // 
            // fcgLBQPI
            // 
            this->fcgLBQPI->AutoSize = true;
            this->fcgLBQPI->Location = System::Drawing::Point(10, 4);
            this->fcgLBQPI->Name = L"fcgLBQPI";
            this->fcgLBQPI->Size = System::Drawing::Size(66, 14);
            this->fcgLBQPI->TabIndex = 75;
            this->fcgLBQPI->Text = L"QP I frame";
            // 
            // fcgNUQPI
            // 
            this->fcgNUQPI->Location = System::Drawing::Point(124, 2);
            this->fcgNUQPI->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPI->Name = L"fcgNUQPI";
            this->fcgNUQPI->Size = System::Drawing::Size(77, 21);
            this->fcgNUQPI->TabIndex = 7;
            this->fcgNUQPI->Tag = L"chValue";
            this->fcgNUQPI->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUQPP
            // 
            this->fcgNUQPP->Location = System::Drawing::Point(124, 29);
            this->fcgNUQPP->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPP->Name = L"fcgNUQPP";
            this->fcgNUQPP->Size = System::Drawing::Size(77, 21);
            this->fcgNUQPP->TabIndex = 8;
            this->fcgNUQPP->Tag = L"chValue";
            this->fcgNUQPP->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUQPB
            // 
            this->fcgNUQPB->Location = System::Drawing::Point(124, 55);
            this->fcgNUQPB->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPB->Name = L"fcgNUQPB";
            this->fcgNUQPB->Size = System::Drawing::Size(77, 21);
            this->fcgNUQPB->TabIndex = 9;
            this->fcgNUQPB->Tag = L"chValue";
            this->fcgNUQPB->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBQPP
            // 
            this->fcgLBQPP->AutoSize = true;
            this->fcgLBQPP->Location = System::Drawing::Point(10, 31);
            this->fcgLBQPP->Name = L"fcgLBQPP";
            this->fcgLBQPP->Size = System::Drawing::Size(69, 14);
            this->fcgLBQPP->TabIndex = 76;
            this->fcgLBQPP->Text = L"QP P frame";
            // 
            // fcgLBQPB
            // 
            this->fcgLBQPB->AutoSize = true;
            this->fcgLBQPB->Location = System::Drawing::Point(6, 57);
            this->fcgLBQPB->Name = L"fcgLBQPB";
            this->fcgLBQPB->Size = System::Drawing::Size(69, 14);
            this->fcgLBQPB->TabIndex = 77;
            this->fcgLBQPB->Text = L"QP B frame";
            // 
            // fcgLBMFXLibDetectionSwValue
            // 
            this->fcgLBMFXLibDetectionSwValue->AutoSize = true;
            this->fcgLBMFXLibDetectionSwValue->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9.75F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgLBMFXLibDetectionSwValue->ForeColor = System::Drawing::Color::DarkViolet;
            this->fcgLBMFXLibDetectionSwValue->Location = System::Drawing::Point(515, 28);
            this->fcgLBMFXLibDetectionSwValue->Name = L"fcgLBMFXLibDetectionSwValue";
            this->fcgLBMFXLibDetectionSwValue->Size = System::Drawing::Size(33, 17);
            this->fcgLBMFXLibDetectionSwValue->TabIndex = 110;
            this->fcgLBMFXLibDetectionSwValue->Text = L"hw:";
            this->fcgLBMFXLibDetectionSwValue->Visible = false;
            // 
            // fcgLBMFXLibDetectionHwValue
            // 
            this->fcgLBMFXLibDetectionHwValue->AutoSize = true;
            this->fcgLBMFXLibDetectionHwValue->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9.75F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgLBMFXLibDetectionHwValue->ForeColor = System::Drawing::Color::DarkViolet;
            this->fcgLBMFXLibDetectionHwValue->Location = System::Drawing::Point(515, 20);
            this->fcgLBMFXLibDetectionHwValue->Name = L"fcgLBMFXLibDetectionHwValue";
            this->fcgLBMFXLibDetectionHwValue->Size = System::Drawing::Size(33, 17);
            this->fcgLBMFXLibDetectionHwValue->TabIndex = 109;
            this->fcgLBMFXLibDetectionHwValue->Text = L"hw:";
            // 
            // fcgLBMFXLibDetectionSwStatus
            // 
            this->fcgLBMFXLibDetectionSwStatus->AutoSize = true;
            this->fcgLBMFXLibDetectionSwStatus->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9.75F, System::Drawing::FontStyle::Italic,
                System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(128)));
            this->fcgLBMFXLibDetectionSwStatus->ForeColor = System::Drawing::Color::Blue;
            this->fcgLBMFXLibDetectionSwStatus->Location = System::Drawing::Point(476, 28);
            this->fcgLBMFXLibDetectionSwStatus->Name = L"fcgLBMFXLibDetectionSwStatus";
            this->fcgLBMFXLibDetectionSwStatus->Size = System::Drawing::Size(31, 17);
            this->fcgLBMFXLibDetectionSwStatus->TabIndex = 108;
            this->fcgLBMFXLibDetectionSwStatus->Text = L"sw:";
            this->fcgLBMFXLibDetectionSwStatus->Visible = false;
            // 
            // fcgLBMFXLibDetectionHwStatus
            // 
            this->fcgLBMFXLibDetectionHwStatus->AutoSize = true;
            this->fcgLBMFXLibDetectionHwStatus->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9.75F, System::Drawing::FontStyle::Italic,
                System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(128)));
            this->fcgLBMFXLibDetectionHwStatus->ForeColor = System::Drawing::Color::Blue;
            this->fcgLBMFXLibDetectionHwStatus->Location = System::Drawing::Point(476, 20);
            this->fcgLBMFXLibDetectionHwStatus->Name = L"fcgLBMFXLibDetectionHwStatus";
            this->fcgLBMFXLibDetectionHwStatus->Size = System::Drawing::Size(33, 17);
            this->fcgLBMFXLibDetectionHwStatus->TabIndex = 107;
            this->fcgLBMFXLibDetectionHwStatus->Text = L"hw:";
            // 
            // fcgLBMFXLibDetection
            // 
            this->fcgLBMFXLibDetection->AutoSize = true;
            this->fcgLBMFXLibDetection->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 11.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgLBMFXLibDetection->ForeColor = System::Drawing::Color::Blue;
            this->fcgLBMFXLibDetection->Location = System::Drawing::Point(334, 17);
            this->fcgLBMFXLibDetection->Name = L"fcgLBMFXLibDetection";
            this->fcgLBMFXLibDetection->Size = System::Drawing::Size(127, 19);
            this->fcgLBMFXLibDetection->TabIndex = 106;
            this->fcgLBMFXLibDetection->Text = L"Intel Media SDK";
            // 
            // fcgLBSlices2
            // 
            this->fcgLBSlices2->AutoSize = true;
            this->fcgLBSlices2->Location = System::Drawing::Point(536, 287);
            this->fcgLBSlices2->Name = L"fcgLBSlices2";
            this->fcgLBSlices2->Size = System::Drawing::Size(66, 14);
            this->fcgLBSlices2->TabIndex = 105;
            this->fcgLBSlices2->Text = L"※\"0\"で自動";
            // 
            // fcgNUSlices
            // 
            this->fcgNUSlices->Location = System::Drawing::Point(455, 285);
            this->fcgNUSlices->Name = L"fcgNUSlices";
            this->fcgNUSlices->Size = System::Drawing::Size(70, 21);
            this->fcgNUSlices->TabIndex = 35;
            this->fcgNUSlices->Tag = L"chValue";
            this->fcgNUSlices->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBSlices
            // 
            this->fcgLBSlices->AutoSize = true;
            this->fcgLBSlices->Location = System::Drawing::Point(355, 287);
            this->fcgLBSlices->Name = L"fcgLBSlices";
            this->fcgLBSlices->Size = System::Drawing::Size(50, 14);
            this->fcgLBSlices->TabIndex = 103;
            this->fcgLBSlices->Text = L"スライス数";
            // 
            // fcgLBRefAuto
            // 
            this->fcgLBRefAuto->AutoSize = true;
            this->fcgLBRefAuto->Location = System::Drawing::Point(216, 246);
            this->fcgLBRefAuto->Name = L"fcgLBRefAuto";
            this->fcgLBRefAuto->Size = System::Drawing::Size(66, 14);
            this->fcgLBRefAuto->TabIndex = 102;
            this->fcgLBRefAuto->Text = L"※\"0\"で自動";
            // 
            // fcgLBGOPLengthAuto
            // 
            this->fcgLBGOPLengthAuto->AutoSize = true;
            this->fcgLBGOPLengthAuto->Location = System::Drawing::Point(214, 217);
            this->fcgLBGOPLengthAuto->Name = L"fcgLBGOPLengthAuto";
            this->fcgLBGOPLengthAuto->Size = System::Drawing::Size(66, 14);
            this->fcgLBGOPLengthAuto->TabIndex = 101;
            this->fcgLBGOPLengthAuto->Text = L"※\"0\"で自動";
            // 
            // fcgLBBframesAuto
            // 
            this->fcgLBBframesAuto->AutoSize = true;
            this->fcgLBBframesAuto->Location = System::Drawing::Point(215, 273);
            this->fcgLBBframesAuto->Name = L"fcgLBBframesAuto";
            this->fcgLBBframesAuto->Size = System::Drawing::Size(70, 14);
            this->fcgLBBframesAuto->TabIndex = 100;
            this->fcgLBBframesAuto->Text = L"※\"-1\"で自動";
            // 
            // fcgCXVideoFormat
            // 
            this->fcgCXVideoFormat->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVideoFormat->FormattingEnabled = true;
            this->fcgCXVideoFormat->Location = System::Drawing::Point(454, 317);
            this->fcgCXVideoFormat->Name = L"fcgCXVideoFormat";
            this->fcgCXVideoFormat->Size = System::Drawing::Size(121, 22);
            this->fcgCXVideoFormat->TabIndex = 36;
            this->fcgCXVideoFormat->Tag = L"chValue";
            // 
            // fcgLBVideoFormat
            // 
            this->fcgLBVideoFormat->AutoSize = true;
            this->fcgLBVideoFormat->Location = System::Drawing::Point(355, 318);
            this->fcgLBVideoFormat->Name = L"fcgLBVideoFormat";
            this->fcgLBVideoFormat->Size = System::Drawing::Size(73, 14);
            this->fcgLBVideoFormat->TabIndex = 98;
            this->fcgLBVideoFormat->Text = L"videoformat";
            // 
            // fcggroupBoxColor
            // 
            this->fcggroupBoxColor->Controls->Add(this->fcgCXTransfer);
            this->fcggroupBoxColor->Controls->Add(this->fcgCXColorPrim);
            this->fcggroupBoxColor->Controls->Add(this->fcgCXColorMatrix);
            this->fcggroupBoxColor->Controls->Add(this->fcgLBTransfer);
            this->fcggroupBoxColor->Controls->Add(this->fcgLBColorPrim);
            this->fcggroupBoxColor->Controls->Add(this->fcgLBColorMatrix);
            this->fcggroupBoxColor->Location = System::Drawing::Point(349, 370);
            this->fcggroupBoxColor->Name = L"fcggroupBoxColor";
            this->fcggroupBoxColor->Size = System::Drawing::Size(241, 103);
            this->fcggroupBoxColor->TabIndex = 40;
            this->fcggroupBoxColor->TabStop = false;
            this->fcggroupBoxColor->Text = L"色設定";
            // 
            // fcgCXTransfer
            // 
            this->fcgCXTransfer->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXTransfer->FormattingEnabled = true;
            this->fcgCXTransfer->Location = System::Drawing::Point(105, 72);
            this->fcgCXTransfer->Name = L"fcgCXTransfer";
            this->fcgCXTransfer->Size = System::Drawing::Size(121, 22);
            this->fcgCXTransfer->TabIndex = 2;
            this->fcgCXTransfer->Tag = L"chValue";
            // 
            // fcgCXColorPrim
            // 
            this->fcgCXColorPrim->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXColorPrim->FormattingEnabled = true;
            this->fcgCXColorPrim->Location = System::Drawing::Point(105, 44);
            this->fcgCXColorPrim->Name = L"fcgCXColorPrim";
            this->fcgCXColorPrim->Size = System::Drawing::Size(121, 22);
            this->fcgCXColorPrim->TabIndex = 1;
            this->fcgCXColorPrim->Tag = L"chValue";
            // 
            // fcgCXColorMatrix
            // 
            this->fcgCXColorMatrix->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXColorMatrix->FormattingEnabled = true;
            this->fcgCXColorMatrix->Location = System::Drawing::Point(105, 16);
            this->fcgCXColorMatrix->Name = L"fcgCXColorMatrix";
            this->fcgCXColorMatrix->Size = System::Drawing::Size(121, 22);
            this->fcgCXColorMatrix->TabIndex = 0;
            this->fcgCXColorMatrix->Tag = L"chValue";
            // 
            // fcgLBTransfer
            // 
            this->fcgLBTransfer->AutoSize = true;
            this->fcgLBTransfer->Location = System::Drawing::Point(18, 75);
            this->fcgLBTransfer->Name = L"fcgLBTransfer";
            this->fcgLBTransfer->Size = System::Drawing::Size(49, 14);
            this->fcgLBTransfer->TabIndex = 2;
            this->fcgLBTransfer->Text = L"transfer";
            // 
            // fcgLBColorPrim
            // 
            this->fcgLBColorPrim->AutoSize = true;
            this->fcgLBColorPrim->Location = System::Drawing::Point(18, 47);
            this->fcgLBColorPrim->Name = L"fcgLBColorPrim";
            this->fcgLBColorPrim->Size = System::Drawing::Size(61, 14);
            this->fcgLBColorPrim->TabIndex = 1;
            this->fcgLBColorPrim->Text = L"colorprim";
            // 
            // fcgLBColorMatrix
            // 
            this->fcgLBColorMatrix->AutoSize = true;
            this->fcgLBColorMatrix->Location = System::Drawing::Point(18, 19);
            this->fcgLBColorMatrix->Name = L"fcgLBColorMatrix";
            this->fcgLBColorMatrix->Size = System::Drawing::Size(70, 14);
            this->fcgLBColorMatrix->TabIndex = 0;
            this->fcgLBColorMatrix->Text = L"colormatrix";
            // 
            // fcgCBOpenGOP
            // 
            this->fcgCBOpenGOP->AutoSize = true;
            this->fcgCBOpenGOP->Location = System::Drawing::Point(195, 366);
            this->fcgCBOpenGOP->Name = L"fcgCBOpenGOP";
            this->fcgCBOpenGOP->Size = System::Drawing::Size(81, 18);
            this->fcgCBOpenGOP->TabIndex = 14;
            this->fcgCBOpenGOP->Tag = L"chValue";
            this->fcgCBOpenGOP->Text = L"open-GOP";
            this->fcgCBOpenGOP->UseVisualStyleBackColor = true;
            // 
            // fcgCBSceneChange
            // 
            this->fcgCBSceneChange->AutoSize = true;
            this->fcgCBSceneChange->Location = System::Drawing::Point(39, 296);
            this->fcgCBSceneChange->Name = L"fcgCBSceneChange";
            this->fcgCBSceneChange->Size = System::Drawing::Size(107, 18);
            this->fcgCBSceneChange->TabIndex = 13;
            this->fcgCBSceneChange->Tag = L"chValue";
            this->fcgCBSceneChange->Text = L"シーンチェンジ検出";
            this->fcgCBSceneChange->UseVisualStyleBackColor = true;
            // 
            // fcgGroupBoxAspectRatio
            // 
            this->fcgGroupBoxAspectRatio->Controls->Add(this->fcgLBAspectRatio);
            this->fcgGroupBoxAspectRatio->Controls->Add(this->fcgNUAspectRatioY);
            this->fcgGroupBoxAspectRatio->Controls->Add(this->fcgNUAspectRatioX);
            this->fcgGroupBoxAspectRatio->Controls->Add(this->fcgCXAspectRatio);
            this->fcgGroupBoxAspectRatio->Location = System::Drawing::Point(349, 54);
            this->fcgGroupBoxAspectRatio->Name = L"fcgGroupBoxAspectRatio";
            this->fcgGroupBoxAspectRatio->Size = System::Drawing::Size(219, 92);
            this->fcgGroupBoxAspectRatio->TabIndex = 30;
            this->fcgGroupBoxAspectRatio->TabStop = false;
            this->fcgGroupBoxAspectRatio->Text = L"アスペクト比";
            // 
            // fcgLBAspectRatio
            // 
            this->fcgLBAspectRatio->AutoSize = true;
            this->fcgLBAspectRatio->Location = System::Drawing::Point(102, 58);
            this->fcgLBAspectRatio->Name = L"fcgLBAspectRatio";
            this->fcgLBAspectRatio->Size = System::Drawing::Size(12, 14);
            this->fcgLBAspectRatio->TabIndex = 3;
            this->fcgLBAspectRatio->Text = L":";
            // 
            // fcgNUAspectRatioY
            // 
            this->fcgNUAspectRatioY->Location = System::Drawing::Point(120, 56);
            this->fcgNUAspectRatioY->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65535, 0, 0, 0 });
            this->fcgNUAspectRatioY->Name = L"fcgNUAspectRatioY";
            this->fcgNUAspectRatioY->Size = System::Drawing::Size(70, 21);
            this->fcgNUAspectRatioY->TabIndex = 2;
            this->fcgNUAspectRatioY->Tag = L"chValue";
            this->fcgNUAspectRatioY->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUAspectRatioX
            // 
            this->fcgNUAspectRatioX->Location = System::Drawing::Point(26, 56);
            this->fcgNUAspectRatioX->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65535, 0, 0, 0 });
            this->fcgNUAspectRatioX->Name = L"fcgNUAspectRatioX";
            this->fcgNUAspectRatioX->Size = System::Drawing::Size(70, 21);
            this->fcgNUAspectRatioX->TabIndex = 1;
            this->fcgNUAspectRatioX->Tag = L"chValue";
            this->fcgNUAspectRatioX->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgCXAspectRatio
            // 
            this->fcgCXAspectRatio->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAspectRatio->FormattingEnabled = true;
            this->fcgCXAspectRatio->Location = System::Drawing::Point(26, 24);
            this->fcgCXAspectRatio->Name = L"fcgCXAspectRatio";
            this->fcgCXAspectRatio->Size = System::Drawing::Size(174, 22);
            this->fcgCXAspectRatio->TabIndex = 0;
            this->fcgCXAspectRatio->Tag = L"chValue";
            // 
            // fcgCXInterlaced
            // 
            this->fcgCXInterlaced->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXInterlaced->FormattingEnabled = true;
            this->fcgCXInterlaced->Location = System::Drawing::Point(454, 187);
            this->fcgCXInterlaced->Name = L"fcgCXInterlaced";
            this->fcgCXInterlaced->Size = System::Drawing::Size(121, 22);
            this->fcgCXInterlaced->TabIndex = 32;
            this->fcgCXInterlaced->Tag = L"chValue";
            // 
            // fcgLBInterlaced
            // 
            this->fcgLBInterlaced->AutoSize = true;
            this->fcgLBInterlaced->Location = System::Drawing::Point(355, 190);
            this->fcgLBInterlaced->Name = L"fcgLBInterlaced";
            this->fcgLBInterlaced->Size = System::Drawing::Size(64, 14);
            this->fcgLBInterlaced->TabIndex = 86;
            this->fcgLBInterlaced->Text = L"フレームタイプ";
            // 
            // fcgNUGopLength
            // 
            this->fcgNUGopLength->Location = System::Drawing::Point(132, 214);
            this->fcgNUGopLength->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65535, 0, 0, 0 });
            this->fcgNUGopLength->Name = L"fcgNUGopLength";
            this->fcgNUGopLength->Size = System::Drawing::Size(77, 21);
            this->fcgNUGopLength->TabIndex = 12;
            this->fcgNUGopLength->Tag = L"chValue";
            this->fcgNUGopLength->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBGOPLength
            // 
            this->fcgLBGOPLength->AutoSize = true;
            this->fcgLBGOPLength->Location = System::Drawing::Point(14, 217);
            this->fcgLBGOPLength->Name = L"fcgLBGOPLength";
            this->fcgLBGOPLength->Size = System::Drawing::Size(41, 14);
            this->fcgLBGOPLength->TabIndex = 85;
            this->fcgLBGOPLength->Text = L"GOP長";
            // 
            // fcgCXCodecLevel
            // 
            this->fcgCXCodecLevel->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXCodecLevel->FormattingEnabled = true;
            this->fcgCXCodecLevel->Location = System::Drawing::Point(454, 251);
            this->fcgCXCodecLevel->Name = L"fcgCXCodecLevel";
            this->fcgCXCodecLevel->Size = System::Drawing::Size(121, 22);
            this->fcgCXCodecLevel->TabIndex = 34;
            this->fcgCXCodecLevel->Tag = L"chValue";
            // 
            // fcgCXCodecProfile
            // 
            this->fcgCXCodecProfile->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXCodecProfile->FormattingEnabled = true;
            this->fcgCXCodecProfile->Location = System::Drawing::Point(454, 219);
            this->fcgCXCodecProfile->Name = L"fcgCXCodecProfile";
            this->fcgCXCodecProfile->Size = System::Drawing::Size(121, 22);
            this->fcgCXCodecProfile->TabIndex = 33;
            this->fcgCXCodecProfile->Tag = L"chValue";
            // 
            // fcgLBCodecLevel
            // 
            this->fcgLBCodecLevel->AutoSize = true;
            this->fcgLBCodecLevel->Location = System::Drawing::Point(355, 254);
            this->fcgLBCodecLevel->Name = L"fcgLBCodecLevel";
            this->fcgLBCodecLevel->Size = System::Drawing::Size(33, 14);
            this->fcgLBCodecLevel->TabIndex = 84;
            this->fcgLBCodecLevel->Text = L"レベル";
            // 
            // fcgLBCodecProfile
            // 
            this->fcgLBCodecProfile->AutoSize = true;
            this->fcgLBCodecProfile->Location = System::Drawing::Point(355, 222);
            this->fcgLBCodecProfile->Name = L"fcgLBCodecProfile";
            this->fcgLBCodecProfile->Size = System::Drawing::Size(53, 14);
            this->fcgLBCodecProfile->TabIndex = 83;
            this->fcgLBCodecProfile->Text = L"プロファイル";
            // 
            // fcgNUBframes
            // 
            this->fcgNUBframes->Location = System::Drawing::Point(133, 268);
            this->fcgNUBframes->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 16, 0, 0, 0 });
            this->fcgNUBframes->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { -1, 0, 0, System::Int32::MinValue });
            this->fcgNUBframes->Name = L"fcgNUBframes";
            this->fcgNUBframes->Size = System::Drawing::Size(77, 21);
            this->fcgNUBframes->TabIndex = 16;
            this->fcgNUBframes->Tag = L"chValue";
            this->fcgNUBframes->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNURef
            // 
            this->fcgNURef->Location = System::Drawing::Point(133, 241);
            this->fcgNURef->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 16, 0, 0, 0 });
            this->fcgNURef->Name = L"fcgNURef";
            this->fcgNURef->Size = System::Drawing::Size(77, 21);
            this->fcgNURef->TabIndex = 15;
            this->fcgNURef->Tag = L"chValue";
            this->fcgNURef->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBBframes
            // 
            this->fcgLBBframes->AutoSize = true;
            this->fcgLBBframes->Location = System::Drawing::Point(13, 270);
            this->fcgLBBframes->Name = L"fcgLBBframes";
            this->fcgLBBframes->Size = System::Drawing::Size(80, 14);
            this->fcgLBBframes->TabIndex = 82;
            this->fcgLBBframes->Text = L"連続Bフレーム数";
            // 
            // fcgLBRef
            // 
            this->fcgLBRef->AutoSize = true;
            this->fcgLBRef->Location = System::Drawing::Point(13, 243);
            this->fcgLBRef->Name = L"fcgLBRef";
            this->fcgLBRef->Size = System::Drawing::Size(51, 14);
            this->fcgLBRef->TabIndex = 81;
            this->fcgLBRef->Text = L"参照距離";
            // 
            // fcgLBEncMode
            // 
            this->fcgLBEncMode->AutoSize = true;
            this->fcgLBEncMode->Location = System::Drawing::Point(13, 101);
            this->fcgLBEncMode->Name = L"fcgLBEncMode";
            this->fcgLBEncMode->Size = System::Drawing::Size(32, 14);
            this->fcgLBEncMode->TabIndex = 79;
            this->fcgLBEncMode->Text = L"モード";
            // 
            // fcgCXEncMode
            // 
            this->fcgCXEncMode->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXEncMode->FormattingEnabled = true;
            this->fcgCXEncMode->Items->AddRange(gcnew cli::array< System::Object^  >(3) { L"高品質", L"標準", L"高速" });
            this->fcgCXEncMode->Location = System::Drawing::Point(81, 98);
            this->fcgCXEncMode->Name = L"fcgCXEncMode";
            this->fcgCXEncMode->Size = System::Drawing::Size(199, 22);
            this->fcgCXEncMode->TabIndex = 4;
            this->fcgCXEncMode->Tag = L"chValue";
            this->fcgCXEncMode->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
            // 
            // fcgLBQuality
            // 
            this->fcgLBQuality->AutoSize = true;
            this->fcgLBQuality->Location = System::Drawing::Point(13, 71);
            this->fcgLBQuality->Name = L"fcgLBQuality";
            this->fcgLBQuality->Size = System::Drawing::Size(29, 14);
            this->fcgLBQuality->TabIndex = 74;
            this->fcgLBQuality->Text = L"速度";
            // 
            // fcgLBOutputType
            // 
            this->fcgLBOutputType->AutoSize = true;
            this->fcgLBOutputType->Location = System::Drawing::Point(13, 41);
            this->fcgLBOutputType->Name = L"fcgLBOutputType";
            this->fcgLBOutputType->Size = System::Drawing::Size(29, 14);
            this->fcgLBOutputType->TabIndex = 73;
            this->fcgLBOutputType->Text = L"出力";
            // 
            // fcgCXQuality
            // 
            this->fcgCXQuality->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXQuality->FormattingEnabled = true;
            this->fcgCXQuality->Items->AddRange(gcnew cli::array< System::Object^  >(3) { L"高品質", L"標準", L"高速" });
            this->fcgCXQuality->Location = System::Drawing::Point(81, 68);
            this->fcgCXQuality->Name = L"fcgCXQuality";
            this->fcgCXQuality->Size = System::Drawing::Size(160, 22);
            this->fcgCXQuality->TabIndex = 3;
            this->fcgCXQuality->Tag = L"chValue";
            // 
            // fcgCXOutputType
            // 
            this->fcgCXOutputType->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXOutputType->FormattingEnabled = true;
            this->fcgCXOutputType->Items->AddRange(gcnew cli::array< System::Object^  >(2) { L"H.264 / AVC", L"MPEG2" });
            this->fcgCXOutputType->Location = System::Drawing::Point(81, 38);
            this->fcgCXOutputType->Name = L"fcgCXOutputType";
            this->fcgCXOutputType->Size = System::Drawing::Size(143, 22);
            this->fcgCXOutputType->TabIndex = 2;
            this->fcgCXOutputType->Tag = L"chValue";
            this->fcgCXOutputType->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgCXOutputType_SelectedIndexChanged);
            // 
            // fcgCBHWEncode
            // 
            this->fcgCBHWEncode->AutoSize = true;
            this->fcgCBHWEncode->Location = System::Drawing::Point(9, 11);
            this->fcgCBHWEncode->Name = L"fcgCBHWEncode";
            this->fcgCBHWEncode->Size = System::Drawing::Size(217, 18);
            this->fcgCBHWEncode->TabIndex = 0;
            this->fcgCBHWEncode->Tag = L"chValue";
            this->fcgCBHWEncode->Text = L"ハードウェアエンコード (QuickSyncVideo)";
            this->fcgCBHWEncode->UseVisualStyleBackColor = true;
            this->fcgCBHWEncode->CheckedChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
            // 
            // tabPageVpp
            // 
            this->tabPageVpp->Controls->Add(this->fcggroupBoxDetail);
            this->tabPageVpp->Controls->Add(this->fcgPNExtSettings);
            this->tabPageVpp->Controls->Add(this->fcgCBUseVpp);
            this->tabPageVpp->Controls->Add(this->fcggroupBoxVpp);
            this->tabPageVpp->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->tabPageVpp->Location = System::Drawing::Point(4, 24);
            this->tabPageVpp->Name = L"tabPageVpp";
            this->tabPageVpp->Size = System::Drawing::Size(608, 481);
            this->tabPageVpp->TabIndex = 2;
            this->tabPageVpp->Text = L"Vpp / 詳細設定";
            this->tabPageVpp->UseVisualStyleBackColor = true;
            // 
            // fcggroupBoxDetail
            // 
            this->fcggroupBoxDetail->Controls->Add(this->fcgCBDirectBiasAdjust);
            this->fcggroupBoxDetail->Controls->Add(this->fcgCXMVCostScaling);
            this->fcggroupBoxDetail->Controls->Add(this->fcgLBMVCostScaling);
            this->fcggroupBoxDetail->Controls->Add(this->fcgCBExtBRC);
            this->fcggroupBoxDetail->Controls->Add(this->fcgCBMBBRC);
            this->fcggroupBoxDetail->Controls->Add(this->fcgCXTrellis);
            this->fcggroupBoxDetail->Controls->Add(this->fcgLBTrellis);
            this->fcggroupBoxDetail->Controls->Add(this->fcgCBIntraRefresh);
            this->fcggroupBoxDetail->Controls->Add(this->fcgCBDeblock);
            this->fcggroupBoxDetail->Location = System::Drawing::Point(12, 267);
            this->fcggroupBoxDetail->Name = L"fcggroupBoxDetail";
            this->fcggroupBoxDetail->Size = System::Drawing::Size(276, 200);
            this->fcggroupBoxDetail->TabIndex = 130;
            this->fcggroupBoxDetail->TabStop = false;
            this->fcggroupBoxDetail->Text = L"詳細設定";
            // 
            // fcgCBDirectBiasAdjust
            // 
            this->fcgCBDirectBiasAdjust->AutoSize = true;
            this->fcgCBDirectBiasAdjust->Location = System::Drawing::Point(12, 44);
            this->fcgCBDirectBiasAdjust->Name = L"fcgCBDirectBiasAdjust";
            this->fcgCBDirectBiasAdjust->Size = System::Drawing::Size(124, 18);
            this->fcgCBDirectBiasAdjust->TabIndex = 148;
            this->fcgCBDirectBiasAdjust->Tag = L"chValue";
            this->fcgCBDirectBiasAdjust->Text = L"ダイレクトモード最適化";
            this->fcgCBDirectBiasAdjust->UseVisualStyleBackColor = true;
            // 
            // fcgCXMVCostScaling
            // 
            this->fcgCXMVCostScaling->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXMVCostScaling->FormattingEnabled = true;
            this->fcgCXMVCostScaling->Location = System::Drawing::Point(129, 73);
            this->fcgCXMVCostScaling->Name = L"fcgCXMVCostScaling";
            this->fcgCXMVCostScaling->Size = System::Drawing::Size(121, 22);
            this->fcgCXMVCostScaling->TabIndex = 146;
            this->fcgCXMVCostScaling->Tag = L"chValue";
            // 
            // fcgLBMVCostScaling
            // 
            this->fcgLBMVCostScaling->AutoSize = true;
            this->fcgLBMVCostScaling->Location = System::Drawing::Point(11, 76);
            this->fcgLBMVCostScaling->Name = L"fcgLBMVCostScaling";
            this->fcgLBMVCostScaling->Size = System::Drawing::Size(70, 14);
            this->fcgLBMVCostScaling->TabIndex = 147;
            this->fcgLBMVCostScaling->Text = L"MVコスト調整";
            // 
            // fcgCBExtBRC
            // 
            this->fcgCBExtBRC->AutoSize = true;
            this->fcgCBExtBRC->Location = System::Drawing::Point(188, 131);
            this->fcgCBExtBRC->Name = L"fcgCBExtBRC";
            this->fcgCBExtBRC->Size = System::Drawing::Size(65, 18);
            this->fcgCBExtBRC->TabIndex = 145;
            this->fcgCBExtBRC->Tag = L"chValue";
            this->fcgCBExtBRC->Text = L"ExtBRC";
            this->fcgCBExtBRC->UseVisualStyleBackColor = true;
            // 
            // fcgCBMBBRC
            // 
            this->fcgCBMBBRC->AutoSize = true;
            this->fcgCBMBBRC->Location = System::Drawing::Point(31, 131);
            this->fcgCBMBBRC->Name = L"fcgCBMBBRC";
            this->fcgCBMBBRC->Size = System::Drawing::Size(149, 18);
            this->fcgCBMBBRC->TabIndex = 144;
            this->fcgCBMBBRC->Tag = L"chValue";
            this->fcgCBMBBRC->Text = L"マクロブロック単位レート制御";
            this->fcgCBMBBRC->UseVisualStyleBackColor = true;
            // 
            // fcgCXTrellis
            // 
            this->fcgCXTrellis->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXTrellis->FormattingEnabled = true;
            this->fcgCXTrellis->Location = System::Drawing::Point(129, 105);
            this->fcgCXTrellis->Name = L"fcgCXTrellis";
            this->fcgCXTrellis->Size = System::Drawing::Size(121, 22);
            this->fcgCXTrellis->TabIndex = 142;
            this->fcgCXTrellis->Tag = L"chValue";
            // 
            // fcgLBTrellis
            // 
            this->fcgLBTrellis->AutoSize = true;
            this->fcgLBTrellis->Location = System::Drawing::Point(10, 106);
            this->fcgLBTrellis->Name = L"fcgLBTrellis";
            this->fcgLBTrellis->Size = System::Drawing::Size(86, 14);
            this->fcgLBTrellis->TabIndex = 143;
            this->fcgLBTrellis->Text = L"歪みレート最適化";
            // 
            // fcgCBIntraRefresh
            // 
            this->fcgCBIntraRefresh->AutoSize = true;
            this->fcgCBIntraRefresh->Location = System::Drawing::Point(129, 20);
            this->fcgCBIntraRefresh->Name = L"fcgCBIntraRefresh";
            this->fcgCBIntraRefresh->Size = System::Drawing::Size(113, 18);
            this->fcgCBIntraRefresh->TabIndex = 26;
            this->fcgCBIntraRefresh->Tag = L"chValue";
            this->fcgCBIntraRefresh->Text = L"周期的イントラ更新";
            this->fcgCBIntraRefresh->UseVisualStyleBackColor = true;
            // 
            // fcgCBDeblock
            // 
            this->fcgCBDeblock->AutoSize = true;
            this->fcgCBDeblock->Location = System::Drawing::Point(13, 20);
            this->fcgCBDeblock->Name = L"fcgCBDeblock";
            this->fcgCBDeblock->Size = System::Drawing::Size(65, 18);
            this->fcgCBDeblock->TabIndex = 25;
            this->fcgCBDeblock->Tag = L"chValue";
            this->fcgCBDeblock->Text = L"デブロック";
            this->fcgCBDeblock->UseVisualStyleBackColor = true;
            // 
            // fcgPNExtSettings
            // 
            this->fcgPNExtSettings->Controls->Add(this->fcgLBInterPred);
            this->fcgPNExtSettings->Controls->Add(this->fcgLBIntraPred);
            this->fcgPNExtSettings->Controls->Add(this->fcgCXIntraPred);
            this->fcgPNExtSettings->Controls->Add(this->fcgCXInterPred);
            this->fcgPNExtSettings->Controls->Add(this->fcgCXMVPred);
            this->fcgPNExtSettings->Controls->Add(this->fcgLBMVPred);
            this->fcgPNExtSettings->Controls->Add(this->fcgLBMVWindowSize);
            this->fcgPNExtSettings->Controls->Add(this->fcgNUMVSearchWindow);
            this->fcgPNExtSettings->Controls->Add(this->fcgLBMVSearch);
            this->fcgPNExtSettings->Controls->Add(this->fcgCBRDO);
            this->fcgPNExtSettings->Controls->Add(this->fcgCBCABAC);
            this->fcgPNExtSettings->Location = System::Drawing::Point(303, 347);
            this->fcgPNExtSettings->Name = L"fcgPNExtSettings";
            this->fcgPNExtSettings->Size = System::Drawing::Size(302, 126);
            this->fcgPNExtSettings->TabIndex = 129;
            // 
            // fcgLBInterPred
            // 
            this->fcgLBInterPred->AutoSize = true;
            this->fcgLBInterPred->Location = System::Drawing::Point(5, 40);
            this->fcgLBInterPred->Name = L"fcgLBInterPred";
            this->fcgLBInterPred->Size = System::Drawing::Size(155, 14);
            this->fcgLBInterPred->TabIndex = 127;
            this->fcgLBInterPred->Text = L"フレーム間予測 最小ブロックサイズ";
            // 
            // fcgLBIntraPred
            // 
            this->fcgLBIntraPred->AutoSize = true;
            this->fcgLBIntraPred->Location = System::Drawing::Point(5, 9);
            this->fcgLBIntraPred->Name = L"fcgLBIntraPred";
            this->fcgLBIntraPred->Size = System::Drawing::Size(155, 14);
            this->fcgLBIntraPred->TabIndex = 126;
            this->fcgLBIntraPred->Text = L"フレーム内予測 最小ブロックサイズ";
            // 
            // fcgCXIntraPred
            // 
            this->fcgCXIntraPred->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXIntraPred->FormattingEnabled = true;
            this->fcgCXIntraPred->Location = System::Drawing::Point(171, 6);
            this->fcgCXIntraPred->Name = L"fcgCXIntraPred";
            this->fcgCXIntraPred->Size = System::Drawing::Size(122, 22);
            this->fcgCXIntraPred->TabIndex = 20;
            this->fcgCXIntraPred->Tag = L"chValue";
            // 
            // fcgCXInterPred
            // 
            this->fcgCXInterPred->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXInterPred->FormattingEnabled = true;
            this->fcgCXInterPred->Location = System::Drawing::Point(171, 37);
            this->fcgCXInterPred->Name = L"fcgCXInterPred";
            this->fcgCXInterPred->Size = System::Drawing::Size(122, 22);
            this->fcgCXInterPred->TabIndex = 21;
            this->fcgCXInterPred->Tag = L"chValue";
            // 
            // fcgCXMVPred
            // 
            this->fcgCXMVPred->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXMVPred->FormattingEnabled = true;
            this->fcgCXMVPred->Location = System::Drawing::Point(213, 71);
            this->fcgCXMVPred->Name = L"fcgCXMVPred";
            this->fcgCXMVPred->Size = System::Drawing::Size(80, 22);
            this->fcgCXMVPred->TabIndex = 23;
            this->fcgCXMVPred->Tag = L"chValue";
            // 
            // fcgLBMVPred
            // 
            this->fcgLBMVPred->AutoSize = true;
            this->fcgLBMVPred->Location = System::Drawing::Point(178, 74);
            this->fcgLBMVPred->Name = L"fcgLBMVPred";
            this->fcgLBMVPred->Size = System::Drawing::Size(29, 14);
            this->fcgLBMVPred->TabIndex = 121;
            this->fcgLBMVPred->Text = L"精度";
            // 
            // fcgLBMVWindowSize
            // 
            this->fcgLBMVWindowSize->AutoSize = true;
            this->fcgLBMVWindowSize->Location = System::Drawing::Point(67, 74);
            this->fcgLBMVWindowSize->Name = L"fcgLBMVWindowSize";
            this->fcgLBMVWindowSize->Size = System::Drawing::Size(29, 14);
            this->fcgLBMVWindowSize->TabIndex = 120;
            this->fcgLBMVWindowSize->Text = L"範囲";
            // 
            // fcgNUMVSearchWindow
            // 
            this->fcgNUMVSearchWindow->Location = System::Drawing::Point(100, 72);
            this->fcgNUMVSearchWindow->Name = L"fcgNUMVSearchWindow";
            this->fcgNUMVSearchWindow->Size = System::Drawing::Size(60, 21);
            this->fcgNUMVSearchWindow->TabIndex = 22;
            this->fcgNUMVSearchWindow->Tag = L"chValue";
            this->fcgNUMVSearchWindow->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBMVSearch
            // 
            this->fcgLBMVSearch->AutoSize = true;
            this->fcgLBMVSearch->Location = System::Drawing::Point(5, 74);
            this->fcgLBMVSearch->Name = L"fcgLBMVSearch";
            this->fcgLBMVSearch->Size = System::Drawing::Size(46, 14);
            this->fcgLBMVSearch->TabIndex = 118;
            this->fcgLBMVSearch->Text = L"MV探索";
            // 
            // fcgCBRDO
            // 
            this->fcgCBRDO->AutoSize = true;
            this->fcgCBRDO->Location = System::Drawing::Point(177, 105);
            this->fcgCBRDO->Name = L"fcgCBRDO";
            this->fcgCBRDO->Size = System::Drawing::Size(105, 18);
            this->fcgCBRDO->TabIndex = 25;
            this->fcgCBRDO->Tag = L"chValue";
            this->fcgCBRDO->Text = L"歪みレート最適化";
            this->fcgCBRDO->UseVisualStyleBackColor = true;
            // 
            // fcgCBCABAC
            // 
            this->fcgCBCABAC->AutoSize = true;
            this->fcgCBCABAC->Location = System::Drawing::Point(55, 105);
            this->fcgCBCABAC->Name = L"fcgCBCABAC";
            this->fcgCBCABAC->Size = System::Drawing::Size(61, 18);
            this->fcgCBCABAC->TabIndex = 24;
            this->fcgCBCABAC->Tag = L"chValue";
            this->fcgCBCABAC->Text = L"CABAC";
            this->fcgCBCABAC->UseVisualStyleBackColor = true;
            // 
            // fcgCBUseVpp
            // 
            this->fcgCBUseVpp->AutoSize = true;
            this->fcgCBUseVpp->Location = System::Drawing::Point(25, 8);
            this->fcgCBUseVpp->Name = L"fcgCBUseVpp";
            this->fcgCBUseVpp->Size = System::Drawing::Size(227, 18);
            this->fcgCBUseVpp->TabIndex = 0;
            this->fcgCBUseVpp->Tag = L"chValue";
            this->fcgCBUseVpp->Text = L"Intel Media SDK の 映像フィルタを使用する";
            this->fcgCBUseVpp->UseVisualStyleBackColor = true;
            this->fcgCBUseVpp->CheckedChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
            // 
            // fcggroupBoxVpp
            // 
            this->fcggroupBoxVpp->Controls->Add(this->fcgCXRotate);
            this->fcggroupBoxVpp->Controls->Add(this->fcgLBRotate);
            this->fcggroupBoxVpp->Controls->Add(this->fcgCXTelecinePatterns);
            this->fcggroupBoxVpp->Controls->Add(this->fcgCXImageStabilizer);
            this->fcggroupBoxVpp->Controls->Add(this->fcgLBImageStabilizer);
            this->fcggroupBoxVpp->Controls->Add(this->fcgLBFPSConversion);
            this->fcggroupBoxVpp->Controls->Add(this->fcgCXFPSConversion);
            this->fcggroupBoxVpp->Controls->Add(this->fcgLBDeinterlaceDesc);
            this->fcggroupBoxVpp->Controls->Add(this->fcgCXDeinterlace);
            this->fcggroupBoxVpp->Controls->Add(this->fcgLBDeinterlace);
            this->fcggroupBoxVpp->Controls->Add(this->fcgCBVppDetail);
            this->fcggroupBoxVpp->Controls->Add(this->fcggroupBoxVppDetail);
            this->fcggroupBoxVpp->Controls->Add(this->fcgCBVppDenoise);
            this->fcggroupBoxVpp->Controls->Add(this->fcggroupBoxVppDenoise);
            this->fcggroupBoxVpp->Controls->Add(this->fcgCBVppResize);
            this->fcggroupBoxVpp->Controls->Add(this->fcggroupBoxVppResize);
            this->fcggroupBoxVpp->Location = System::Drawing::Point(12, 8);
            this->fcggroupBoxVpp->Name = L"fcggroupBoxVpp";
            this->fcggroupBoxVpp->Size = System::Drawing::Size(585, 253);
            this->fcggroupBoxVpp->TabIndex = 1;
            this->fcggroupBoxVpp->TabStop = false;
            this->fcggroupBoxVpp->Tag = L"";
            // 
            // fcgCXRotate
            // 
            this->fcgCXRotate->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXRotate->FormattingEnabled = true;
            this->fcgCXRotate->Location = System::Drawing::Point(410, 175);
            this->fcgCXRotate->Name = L"fcgCXRotate";
            this->fcgCXRotate->Size = System::Drawing::Size(117, 22);
            this->fcgCXRotate->TabIndex = 27;
            this->fcgCXRotate->Tag = L"chValue";
            // 
            // fcgLBRotate
            // 
            this->fcgLBRotate->AutoSize = true;
            this->fcgLBRotate->Location = System::Drawing::Point(359, 178);
            this->fcgLBRotate->Name = L"fcgLBRotate";
            this->fcgLBRotate->Size = System::Drawing::Size(29, 14);
            this->fcgLBRotate->TabIndex = 26;
            this->fcgLBRotate->Text = L"回転";
            // 
            // fcgCXTelecinePatterns
            // 
            this->fcgCXTelecinePatterns->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXTelecinePatterns->FormattingEnabled = true;
            this->fcgCXTelecinePatterns->Location = System::Drawing::Point(275, 103);
            this->fcgCXTelecinePatterns->Name = L"fcgCXTelecinePatterns";
            this->fcgCXTelecinePatterns->Size = System::Drawing::Size(113, 22);
            this->fcgCXTelecinePatterns->TabIndex = 25;
            this->fcgCXTelecinePatterns->Tag = L"chValue";
            // 
            // fcgCXImageStabilizer
            // 
            this->fcgCXImageStabilizer->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXImageStabilizer->FormattingEnabled = true;
            this->fcgCXImageStabilizer->Location = System::Drawing::Point(153, 213);
            this->fcgCXImageStabilizer->Name = L"fcgCXImageStabilizer";
            this->fcgCXImageStabilizer->Size = System::Drawing::Size(117, 22);
            this->fcgCXImageStabilizer->TabIndex = 24;
            this->fcgCXImageStabilizer->Tag = L"chValue";
            // 
            // fcgLBImageStabilizer
            // 
            this->fcgLBImageStabilizer->AutoSize = true;
            this->fcgLBImageStabilizer->Location = System::Drawing::Point(17, 216);
            this->fcgLBImageStabilizer->Name = L"fcgLBImageStabilizer";
            this->fcgLBImageStabilizer->Size = System::Drawing::Size(91, 14);
            this->fcgLBImageStabilizer->TabIndex = 23;
            this->fcgLBImageStabilizer->Text = L"ImageStabilizer";
            // 
            // fcgLBFPSConversion
            // 
            this->fcgLBFPSConversion->AutoSize = true;
            this->fcgLBFPSConversion->Location = System::Drawing::Point(16, 178);
            this->fcgLBFPSConversion->Name = L"fcgLBFPSConversion";
            this->fcgLBFPSConversion->Size = System::Drawing::Size(128, 14);
            this->fcgLBFPSConversion->TabIndex = 5;
            this->fcgLBFPSConversion->Text = L"補完付きフレームレート変換";
            // 
            // fcgCXFPSConversion
            // 
            this->fcgCXFPSConversion->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXFPSConversion->FormattingEnabled = true;
            this->fcgCXFPSConversion->Location = System::Drawing::Point(153, 175);
            this->fcgCXFPSConversion->Name = L"fcgCXFPSConversion";
            this->fcgCXFPSConversion->Size = System::Drawing::Size(117, 22);
            this->fcgCXFPSConversion->TabIndex = 22;
            this->fcgCXFPSConversion->Tag = L"chValue";
            // 
            // fcgLBDeinterlaceDesc
            // 
            this->fcgLBDeinterlaceDesc->AutoSize = true;
            this->fcgLBDeinterlaceDesc->Location = System::Drawing::Point(86, 131);
            this->fcgLBDeinterlaceDesc->Name = L"fcgLBDeinterlaceDesc";
            this->fcgLBDeinterlaceDesc->Size = System::Drawing::Size(184, 14);
            this->fcgLBDeinterlaceDesc->TabIndex = 13;
            this->fcgLBDeinterlaceDesc->Text = L"※フレームタイプ interlaced時のみ有効";
            // 
            // fcgCXDeinterlace
            // 
            this->fcgCXDeinterlace->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXDeinterlace->FormattingEnabled = true;
            this->fcgCXDeinterlace->Location = System::Drawing::Point(84, 103);
            this->fcgCXDeinterlace->Name = L"fcgCXDeinterlace";
            this->fcgCXDeinterlace->Size = System::Drawing::Size(186, 22);
            this->fcgCXDeinterlace->TabIndex = 1;
            this->fcgCXDeinterlace->Tag = L"chValue";
            this->fcgCXDeinterlace->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
            // 
            // fcgLBDeinterlace
            // 
            this->fcgLBDeinterlace->AutoSize = true;
            this->fcgLBDeinterlace->Location = System::Drawing::Point(17, 106);
            this->fcgLBDeinterlace->Name = L"fcgLBDeinterlace";
            this->fcgLBDeinterlace->Size = System::Drawing::Size(61, 14);
            this->fcgLBDeinterlace->TabIndex = 12;
            this->fcgLBDeinterlace->Text = L"インタレ解除";
            // 
            // fcgCBVppDetail
            // 
            this->fcgCBVppDetail->AutoSize = true;
            this->fcgCBVppDetail->Location = System::Drawing::Point(436, 25);
            this->fcgCBVppDetail->Name = L"fcgCBVppDetail";
            this->fcgCBVppDetail->Size = System::Drawing::Size(116, 18);
            this->fcgCBVppDetail->TabIndex = 11;
            this->fcgCBVppDetail->Tag = L"chValue";
            this->fcgCBVppDetail->Text = L"ディテール/輪郭強調";
            this->fcgCBVppDetail->UseVisualStyleBackColor = true;
            this->fcgCBVppDetail->CheckedChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
            // 
            // fcggroupBoxVppDetail
            // 
            this->fcggroupBoxVppDetail->Controls->Add(this->fcgNUVppDetail);
            this->fcggroupBoxVppDetail->Controls->Add(this->fcgLBDetail);
            this->fcggroupBoxVppDetail->Location = System::Drawing::Point(423, 26);
            this->fcggroupBoxVppDetail->Name = L"fcggroupBoxVppDetail";
            this->fcggroupBoxVppDetail->Size = System::Drawing::Size(150, 60);
            this->fcggroupBoxVppDetail->TabIndex = 12;
            this->fcggroupBoxVppDetail->TabStop = false;
            // 
            // fcgNUVppDetail
            // 
            this->fcgNUVppDetail->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
            this->fcgNUVppDetail->Location = System::Drawing::Point(50, 27);
            this->fcgNUVppDetail->Name = L"fcgNUVppDetail";
            this->fcgNUVppDetail->Size = System::Drawing::Size(79, 21);
            this->fcgNUVppDetail->TabIndex = 0;
            this->fcgNUVppDetail->Tag = L"chValue";
            this->fcgNUVppDetail->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBDetail
            // 
            this->fcgLBDetail->AutoSize = true;
            this->fcgLBDetail->Location = System::Drawing::Point(15, 29);
            this->fcgLBDetail->Name = L"fcgLBDetail";
            this->fcgLBDetail->Size = System::Drawing::Size(26, 14);
            this->fcgLBDetail->TabIndex = 5;
            this->fcgLBDetail->Text = L"強さ";
            // 
            // fcgCBVppDenoise
            // 
            this->fcgCBVppDenoise->AutoSize = true;
            this->fcgCBVppDenoise->Location = System::Drawing::Point(268, 24);
            this->fcgCBVppDenoise->Name = L"fcgCBVppDenoise";
            this->fcgCBVppDenoise->Size = System::Drawing::Size(72, 18);
            this->fcgCBVppDenoise->TabIndex = 2;
            this->fcgCBVppDenoise->Tag = L"chValue";
            this->fcgCBVppDenoise->Text = L"ノイズ除去";
            this->fcgCBVppDenoise->UseVisualStyleBackColor = true;
            this->fcgCBVppDenoise->CheckedChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
            // 
            // fcggroupBoxVppDenoise
            // 
            this->fcggroupBoxVppDenoise->Controls->Add(this->fcgNUVppDenoise);
            this->fcggroupBoxVppDenoise->Controls->Add(this->fcgLBVppDenoise);
            this->fcggroupBoxVppDenoise->Location = System::Drawing::Point(260, 26);
            this->fcggroupBoxVppDenoise->Name = L"fcggroupBoxVppDenoise";
            this->fcggroupBoxVppDenoise->Size = System::Drawing::Size(145, 59);
            this->fcggroupBoxVppDenoise->TabIndex = 3;
            this->fcggroupBoxVppDenoise->TabStop = false;
            // 
            // fcgNUVppDenoise
            // 
            this->fcgNUVppDenoise->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
            this->fcgNUVppDenoise->Location = System::Drawing::Point(49, 27);
            this->fcgNUVppDenoise->Name = L"fcgNUVppDenoise";
            this->fcgNUVppDenoise->Size = System::Drawing::Size(79, 21);
            this->fcgNUVppDenoise->TabIndex = 4;
            this->fcgNUVppDenoise->Tag = L"chValue";
            this->fcgNUVppDenoise->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBVppDenoise
            // 
            this->fcgLBVppDenoise->AutoSize = true;
            this->fcgLBVppDenoise->Location = System::Drawing::Point(15, 29);
            this->fcgLBVppDenoise->Name = L"fcgLBVppDenoise";
            this->fcgLBVppDenoise->Size = System::Drawing::Size(26, 14);
            this->fcgLBVppDenoise->TabIndex = 0;
            this->fcgLBVppDenoise->Text = L"強さ";
            // 
            // fcgCBVppResize
            // 
            this->fcgCBVppResize->AutoSize = true;
            this->fcgCBVppResize->Location = System::Drawing::Point(26, 26);
            this->fcgCBVppResize->Name = L"fcgCBVppResize";
            this->fcgCBVppResize->Size = System::Drawing::Size(58, 18);
            this->fcgCBVppResize->TabIndex = 5;
            this->fcgCBVppResize->Tag = L"chValue";
            this->fcgCBVppResize->Text = L"リサイズ";
            this->fcgCBVppResize->UseVisualStyleBackColor = true;
            this->fcgCBVppResize->CheckedChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
            // 
            // fcggroupBoxVppResize
            // 
            this->fcggroupBoxVppResize->Controls->Add(this->fcgNUVppResizeW);
            this->fcggroupBoxVppResize->Controls->Add(this->fcgNUVppResizeH);
            this->fcggroupBoxVppResize->Controls->Add(this->fcgLBVppResize);
            this->fcggroupBoxVppResize->Location = System::Drawing::Point(13, 26);
            this->fcggroupBoxVppResize->Name = L"fcggroupBoxVppResize";
            this->fcggroupBoxVppResize->Size = System::Drawing::Size(228, 59);
            this->fcggroupBoxVppResize->TabIndex = 10;
            this->fcggroupBoxVppResize->TabStop = false;
            // 
            // fcgNUVppResizeW
            // 
            this->fcgNUVppResizeW->Location = System::Drawing::Point(18, 24);
            this->fcgNUVppResizeW->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1920, 0, 0, 0 });
            this->fcgNUVppResizeW->Name = L"fcgNUVppResizeW";
            this->fcgNUVppResizeW->Size = System::Drawing::Size(79, 21);
            this->fcgNUVppResizeW->TabIndex = 0;
            this->fcgNUVppResizeW->Tag = L"chValue";
            this->fcgNUVppResizeW->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUVppResizeH
            // 
            this->fcgNUVppResizeH->Location = System::Drawing::Point(122, 24);
            this->fcgNUVppResizeH->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1088, 0, 0, 0 });
            this->fcgNUVppResizeH->Name = L"fcgNUVppResizeH";
            this->fcgNUVppResizeH->Size = System::Drawing::Size(79, 21);
            this->fcgNUVppResizeH->TabIndex = 1;
            this->fcgNUVppResizeH->Tag = L"chValue";
            this->fcgNUVppResizeH->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBVppResize
            // 
            this->fcgLBVppResize->AutoSize = true;
            this->fcgLBVppResize->Location = System::Drawing::Point(103, 26);
            this->fcgLBVppResize->Name = L"fcgLBVppResize";
            this->fcgLBVppResize->Size = System::Drawing::Size(13, 14);
            this->fcgLBVppResize->TabIndex = 2;
            this->fcgLBVppResize->Tag = L"chValue";
            this->fcgLBVppResize->Text = L"x";
            // 
            // tabPageExOpt
            // 
            this->tabPageExOpt->Controls->Add(this->fcgCBD3DMemAlloc);
            this->tabPageExOpt->Controls->Add(this->fcgCBAuoTcfileout);
            this->tabPageExOpt->Controls->Add(this->fcgCBAFS);
            this->tabPageExOpt->Controls->Add(this->fcgNUInputBufSize);
            this->tabPageExOpt->Controls->Add(this->fcgLBInputBufSize);
            this->tabPageExOpt->Controls->Add(this->fcgLBTempDir);
            this->tabPageExOpt->Controls->Add(this->fcgBTCustomTempDir);
            this->tabPageExOpt->Controls->Add(this->fcgTXCustomTempDir);
            this->tabPageExOpt->Controls->Add(this->fcgCXTempDir);
            this->tabPageExOpt->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->tabPageExOpt->Location = System::Drawing::Point(4, 24);
            this->tabPageExOpt->Name = L"tabPageExOpt";
            this->tabPageExOpt->Size = System::Drawing::Size(608, 481);
            this->tabPageExOpt->TabIndex = 1;
            this->tabPageExOpt->Text = L"その他";
            this->tabPageExOpt->UseVisualStyleBackColor = true;
            // 
            // fcgCBD3DMemAlloc
            // 
            this->fcgCBD3DMemAlloc->AutoSize = true;
            this->fcgCBD3DMemAlloc->Location = System::Drawing::Point(18, 145);
            this->fcgCBD3DMemAlloc->Name = L"fcgCBD3DMemAlloc";
            this->fcgCBD3DMemAlloc->Size = System::Drawing::Size(190, 18);
            this->fcgCBD3DMemAlloc->TabIndex = 72;
            this->fcgCBD3DMemAlloc->Tag = L"chValue";
            this->fcgCBD3DMemAlloc->Text = L"ビデオメモリモード (QSV使用時推奨)";
            this->fcgCBD3DMemAlloc->UseVisualStyleBackColor = true;
            // 
            // fcgCBAuoTcfileout
            // 
            this->fcgCBAuoTcfileout->AutoSize = true;
            this->fcgCBAuoTcfileout->Location = System::Drawing::Point(325, 63);
            this->fcgCBAuoTcfileout->Name = L"fcgCBAuoTcfileout";
            this->fcgCBAuoTcfileout->Size = System::Drawing::Size(98, 18);
            this->fcgCBAuoTcfileout->TabIndex = 71;
            this->fcgCBAuoTcfileout->Tag = L"chValue,NoDirect";
            this->fcgCBAuoTcfileout->Text = L"タイムコード出力";
            this->fcgCBAuoTcfileout->UseVisualStyleBackColor = true;
            // 
            // fcgCBAFS
            // 
            this->fcgCBAFS->AutoSize = true;
            this->fcgCBAFS->Location = System::Drawing::Point(325, 30);
            this->fcgCBAFS->Name = L"fcgCBAFS";
            this->fcgCBAFS->Size = System::Drawing::Size(183, 18);
            this->fcgCBAFS->TabIndex = 70;
            this->fcgCBAFS->Tag = L"chValue,NoDirect";
            this->fcgCBAFS->Text = L"自動フィールドシフト(afs)を使用する";
            this->fcgCBAFS->UseVisualStyleBackColor = true;
            // 
            // fcgNUInputBufSize
            // 
            this->fcgNUInputBufSize->Location = System::Drawing::Point(126, 118);
            this->fcgNUInputBufSize->Name = L"fcgNUInputBufSize";
            this->fcgNUInputBufSize->Size = System::Drawing::Size(86, 21);
            this->fcgNUInputBufSize->TabIndex = 69;
            this->fcgNUInputBufSize->Tag = L"chValue";
            this->fcgNUInputBufSize->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBInputBufSize
            // 
            this->fcgLBInputBufSize->AutoSize = true;
            this->fcgLBInputBufSize->Location = System::Drawing::Point(15, 120);
            this->fcgLBInputBufSize->Name = L"fcgLBInputBufSize";
            this->fcgLBInputBufSize->Size = System::Drawing::Size(83, 14);
            this->fcgLBInputBufSize->TabIndex = 68;
            this->fcgLBInputBufSize->Text = L"入力バッファサイズ";
            // 
            // fcgLBTempDir
            // 
            this->fcgLBTempDir->AutoSize = true;
            this->fcgLBTempDir->Location = System::Drawing::Point(15, 9);
            this->fcgLBTempDir->Name = L"fcgLBTempDir";
            this->fcgLBTempDir->Size = System::Drawing::Size(60, 14);
            this->fcgLBTempDir->TabIndex = 67;
            this->fcgLBTempDir->Tag = L"NoDirect";
            this->fcgLBTempDir->Text = L"一時フォルダ";
            // 
            // fcgBTCustomTempDir
            // 
            this->fcgBTCustomTempDir->Location = System::Drawing::Point(215, 62);
            this->fcgBTCustomTempDir->Name = L"fcgBTCustomTempDir";
            this->fcgBTCustomTempDir->Size = System::Drawing::Size(29, 23);
            this->fcgBTCustomTempDir->TabIndex = 64;
            this->fcgBTCustomTempDir->Tag = L"NoDirect";
            this->fcgBTCustomTempDir->Text = L"...";
            this->fcgBTCustomTempDir->UseVisualStyleBackColor = true;
            this->fcgBTCustomTempDir->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTCustomTempDir_Click);
            // 
            // fcgTXCustomTempDir
            // 
            this->fcgTXCustomTempDir->Location = System::Drawing::Point(30, 63);
            this->fcgTXCustomTempDir->Name = L"fcgTXCustomTempDir";
            this->fcgTXCustomTempDir->Size = System::Drawing::Size(182, 21);
            this->fcgTXCustomTempDir->TabIndex = 63;
            this->fcgTXCustomTempDir->Tag = L"NoDirect";
            this->fcgTXCustomTempDir->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXCustomTempDir_TextChanged);
            // 
            // fcgCXTempDir
            // 
            this->fcgCXTempDir->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXTempDir->FormattingEnabled = true;
            this->fcgCXTempDir->Location = System::Drawing::Point(18, 35);
            this->fcgCXTempDir->Name = L"fcgCXTempDir";
            this->fcgCXTempDir->Size = System::Drawing::Size(209, 22);
            this->fcgCXTempDir->TabIndex = 62;
            this->fcgCXTempDir->Tag = L"chValue,NoDirect";
            // 
            // tabPageFeatures
            // 
            this->tabPageFeatures->Controls->Add(this->fcgLBGPUInfoOnFeatureTab);
            this->tabPageFeatures->Controls->Add(this->fcgLBGPUInfoLabelOnFeatureTab);
            this->tabPageFeatures->Controls->Add(this->fcgLBCPUInfoOnFeatureTab);
            this->tabPageFeatures->Controls->Add(this->fcgLBCPUInfoLabelOnFeatureTab);
            this->tabPageFeatures->Controls->Add(this->fcgBTSaveFeatureList);
            this->tabPageFeatures->Controls->Add(this->fcgLBFeaturesCurrentAPIVer);
            this->tabPageFeatures->Controls->Add(this->fcgLBFeaturesShowCurrentAPI);
            this->tabPageFeatures->Controls->Add(this->fcgDGVFeatures);
            this->tabPageFeatures->Location = System::Drawing::Point(4, 24);
            this->tabPageFeatures->Name = L"tabPageFeatures";
            this->tabPageFeatures->Size = System::Drawing::Size(608, 481);
            this->tabPageFeatures->TabIndex = 3;
            this->tabPageFeatures->Text = L"機能情報";
            this->tabPageFeatures->UseVisualStyleBackColor = true;
            // 
            // fcgLBGPUInfoOnFeatureTab
            // 
            this->fcgLBGPUInfoOnFeatureTab->AutoSize = true;
            this->fcgLBGPUInfoOnFeatureTab->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgLBGPUInfoOnFeatureTab->ForeColor = System::Drawing::Color::DarkViolet;
            this->fcgLBGPUInfoOnFeatureTab->Location = System::Drawing::Point(94, 28);
            this->fcgLBGPUInfoOnFeatureTab->Name = L"fcgLBGPUInfoOnFeatureTab";
            this->fcgLBGPUInfoOnFeatureTab->Size = System::Drawing::Size(30, 14);
            this->fcgLBGPUInfoOnFeatureTab->TabIndex = 116;
            this->fcgLBGPUInfoOnFeatureTab->Text = L"GPU";
            // 
            // fcgLBGPUInfoLabelOnFeatureTab
            // 
            this->fcgLBGPUInfoLabelOnFeatureTab->AutoSize = true;
            this->fcgLBGPUInfoLabelOnFeatureTab->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgLBGPUInfoLabelOnFeatureTab->ForeColor = System::Drawing::Color::Blue;
            this->fcgLBGPUInfoLabelOnFeatureTab->Location = System::Drawing::Point(11, 28);
            this->fcgLBGPUInfoLabelOnFeatureTab->Name = L"fcgLBGPUInfoLabelOnFeatureTab";
            this->fcgLBGPUInfoLabelOnFeatureTab->Size = System::Drawing::Size(32, 15);
            this->fcgLBGPUInfoLabelOnFeatureTab->TabIndex = 115;
            this->fcgLBGPUInfoLabelOnFeatureTab->Text = L"GPU";
            // 
            // fcgLBCPUInfoOnFeatureTab
            // 
            this->fcgLBCPUInfoOnFeatureTab->AutoSize = true;
            this->fcgLBCPUInfoOnFeatureTab->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgLBCPUInfoOnFeatureTab->ForeColor = System::Drawing::Color::DarkViolet;
            this->fcgLBCPUInfoOnFeatureTab->Location = System::Drawing::Point(95, 7);
            this->fcgLBCPUInfoOnFeatureTab->Name = L"fcgLBCPUInfoOnFeatureTab";
            this->fcgLBCPUInfoOnFeatureTab->Size = System::Drawing::Size(29, 14);
            this->fcgLBCPUInfoOnFeatureTab->TabIndex = 114;
            this->fcgLBCPUInfoOnFeatureTab->Text = L"CPU";
            // 
            // fcgLBCPUInfoLabelOnFeatureTab
            // 
            this->fcgLBCPUInfoLabelOnFeatureTab->AutoSize = true;
            this->fcgLBCPUInfoLabelOnFeatureTab->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgLBCPUInfoLabelOnFeatureTab->ForeColor = System::Drawing::Color::Blue;
            this->fcgLBCPUInfoLabelOnFeatureTab->Location = System::Drawing::Point(11, 7);
            this->fcgLBCPUInfoLabelOnFeatureTab->Name = L"fcgLBCPUInfoLabelOnFeatureTab";
            this->fcgLBCPUInfoLabelOnFeatureTab->Size = System::Drawing::Size(31, 15);
            this->fcgLBCPUInfoLabelOnFeatureTab->TabIndex = 113;
            this->fcgLBCPUInfoLabelOnFeatureTab->Text = L"CPU";
            // 
            // fcgBTSaveFeatureList
            // 
            this->fcgBTSaveFeatureList->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
            this->fcgBTSaveFeatureList->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgBTSaveFeatureList->Location = System::Drawing::Point(514, 37);
            this->fcgBTSaveFeatureList->Name = L"fcgBTSaveFeatureList";
            this->fcgBTSaveFeatureList->Size = System::Drawing::Size(88, 30);
            this->fcgBTSaveFeatureList->TabIndex = 112;
            this->fcgBTSaveFeatureList->Text = L"ファイルに保存..";
            this->fcgBTSaveFeatureList->UseVisualStyleBackColor = true;
            this->fcgBTSaveFeatureList->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTSaveFeatureList_Click);
            // 
            // fcgLBFeaturesCurrentAPIVer
            // 
            this->fcgLBFeaturesCurrentAPIVer->AutoSize = true;
            this->fcgLBFeaturesCurrentAPIVer->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgLBFeaturesCurrentAPIVer->ForeColor = System::Drawing::Color::DarkViolet;
            this->fcgLBFeaturesCurrentAPIVer->Location = System::Drawing::Point(96, 49);
            this->fcgLBFeaturesCurrentAPIVer->Name = L"fcgLBFeaturesCurrentAPIVer";
            this->fcgLBFeaturesCurrentAPIVer->Size = System::Drawing::Size(28, 14);
            this->fcgLBFeaturesCurrentAPIVer->TabIndex = 110;
            this->fcgLBFeaturesCurrentAPIVer->Text = L"hw:";
            // 
            // fcgLBFeaturesShowCurrentAPI
            // 
            this->fcgLBFeaturesShowCurrentAPI->AutoSize = true;
            this->fcgLBFeaturesShowCurrentAPI->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgLBFeaturesShowCurrentAPI->ForeColor = System::Drawing::Color::Blue;
            this->fcgLBFeaturesShowCurrentAPI->Location = System::Drawing::Point(11, 49);
            this->fcgLBFeaturesShowCurrentAPI->Name = L"fcgLBFeaturesShowCurrentAPI";
            this->fcgLBFeaturesShowCurrentAPI->Size = System::Drawing::Size(70, 15);
            this->fcgLBFeaturesShowCurrentAPI->TabIndex = 107;
            this->fcgLBFeaturesShowCurrentAPI->Text = L"Media SDK";
            // 
            // fcgDGVFeatures
            // 
            this->fcgDGVFeatures->BackgroundColor = System::Drawing::SystemColors::Control;
            this->fcgDGVFeatures->BorderStyle = System::Windows::Forms::BorderStyle::None;
            this->fcgDGVFeatures->ColumnHeadersHeightSizeMode = System::Windows::Forms::DataGridViewColumnHeadersHeightSizeMode::AutoSize;
            this->fcgDGVFeatures->Location = System::Drawing::Point(4, 74);
            this->fcgDGVFeatures->Name = L"fcgDGVFeatures";
            this->fcgDGVFeatures->ReadOnly = true;
            this->fcgDGVFeatures->RowTemplate->Height = 21;
            this->fcgDGVFeatures->Size = System::Drawing::Size(601, 405);
            this->fcgDGVFeatures->TabIndex = 0;
            this->fcgDGVFeatures->CellFormatting += gcnew System::Windows::Forms::DataGridViewCellFormattingEventHandler(this, &frmConfig::fcgDGVFeatures_CellFormatting);
            // 
            // fcgCSExeFiles
            // 
            this->fcgCSExeFiles->ImageScalingSize = System::Drawing::Size(18, 18);
            this->fcgCSExeFiles->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->fcgTSExeFileshelp });
            this->fcgCSExeFiles->Name = L"fcgCSx264";
            this->fcgCSExeFiles->Size = System::Drawing::Size(131, 26);
            // 
            // fcgTSExeFileshelp
            // 
            this->fcgTSExeFileshelp->Name = L"fcgTSExeFileshelp";
            this->fcgTSExeFileshelp->Size = System::Drawing::Size(130, 22);
            this->fcgTSExeFileshelp->Text = L"helpを表示";
            this->fcgTSExeFileshelp->Click += gcnew System::EventHandler(this, &frmConfig::fcgTSExeFileshelp_Click);
            // 
            // fcgLBguiExBlog
            // 
            this->fcgLBguiExBlog->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
            this->fcgLBguiExBlog->AutoSize = true;
            this->fcgLBguiExBlog->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgLBguiExBlog->LinkColor = System::Drawing::Color::Gray;
            this->fcgLBguiExBlog->Location = System::Drawing::Point(662, 555);
            this->fcgLBguiExBlog->Name = L"fcgLBguiExBlog";
            this->fcgLBguiExBlog->Size = System::Drawing::Size(86, 14);
            this->fcgLBguiExBlog->TabIndex = 50;
            this->fcgLBguiExBlog->TabStop = true;
            this->fcgLBguiExBlog->Text = L"QSVEncについて";
            this->fcgLBguiExBlog->VisitedLinkColor = System::Drawing::Color::Gray;
            this->fcgLBguiExBlog->LinkClicked += gcnew System::Windows::Forms::LinkLabelLinkClickedEventHandler(this, &frmConfig::fcgLBguiExBlog_LinkClicked);
            // 
            // fcgtabControlAudio
            // 
            this->fcgtabControlAudio->Controls->Add(this->fcgtabPageAudioMain);
            this->fcgtabControlAudio->Controls->Add(this->fcgtabPageAudioOther);
            this->fcgtabControlAudio->Controls->Add(this->fcgtabPageAvqsvAudio);
            this->fcgtabControlAudio->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F));
            this->fcgtabControlAudio->Location = System::Drawing::Point(622, 31);
            this->fcgtabControlAudio->Name = L"fcgtabControlAudio";
            this->fcgtabControlAudio->SelectedIndex = 0;
            this->fcgtabControlAudio->Size = System::Drawing::Size(384, 296);
            this->fcgtabControlAudio->TabIndex = 51;
            // 
            // fcgtabPageAudioMain
            // 
            this->fcgtabPageAudioMain->Controls->Add(this->fcgCXAudioDelayCut);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgLBAudioDelayCut);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgCBAudioEncTiming);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgCXAudioEncTiming);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgCXAudioTempDir);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgTXCustomAudioTempDir);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgBTCustomAudioTempDir);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgCBAudioUsePipe);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgLBAudioBitrate);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgNUAudioBitrate);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgCBAudio2pass);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgCXAudioEncMode);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgLBAudioEncMode);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgBTAudioEncoderPath);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgTXAudioEncoderPath);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgLBAudioEncoderPath);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgCBAudioOnly);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgCBFAWCheck);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgCXAudioEncoder);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgLBAudioEncoder);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgLBAudioTemp);
            this->fcgtabPageAudioMain->Location = System::Drawing::Point(4, 23);
            this->fcgtabPageAudioMain->Name = L"fcgtabPageAudioMain";
            this->fcgtabPageAudioMain->Padding = System::Windows::Forms::Padding(3);
            this->fcgtabPageAudioMain->Size = System::Drawing::Size(376, 269);
            this->fcgtabPageAudioMain->TabIndex = 0;
            this->fcgtabPageAudioMain->Tag = L"NoDirect";
            this->fcgtabPageAudioMain->Text = L"音声";
            this->fcgtabPageAudioMain->UseVisualStyleBackColor = true;
            // 
            // fcgCXAudioDelayCut
            // 
            this->fcgCXAudioDelayCut->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAudioDelayCut->FormattingEnabled = true;
            this->fcgCXAudioDelayCut->Location = System::Drawing::Point(291, 133);
            this->fcgCXAudioDelayCut->Name = L"fcgCXAudioDelayCut";
            this->fcgCXAudioDelayCut->Size = System::Drawing::Size(70, 22);
            this->fcgCXAudioDelayCut->TabIndex = 43;
            this->fcgCXAudioDelayCut->Tag = L"chValue";
            // 
            // fcgLBAudioDelayCut
            // 
            this->fcgLBAudioDelayCut->AutoSize = true;
            this->fcgLBAudioDelayCut->Location = System::Drawing::Point(224, 136);
            this->fcgLBAudioDelayCut->Name = L"fcgLBAudioDelayCut";
            this->fcgLBAudioDelayCut->Size = System::Drawing::Size(60, 14);
            this->fcgLBAudioDelayCut->TabIndex = 54;
            this->fcgLBAudioDelayCut->Tag = L"";
            this->fcgLBAudioDelayCut->Text = L"ディレイカット";
            // 
            // fcgCBAudioEncTiming
            // 
            this->fcgCBAudioEncTiming->AutoSize = true;
            this->fcgCBAudioEncTiming->Location = System::Drawing::Point(242, 54);
            this->fcgCBAudioEncTiming->Name = L"fcgCBAudioEncTiming";
            this->fcgCBAudioEncTiming->Size = System::Drawing::Size(40, 14);
            this->fcgCBAudioEncTiming->TabIndex = 53;
            this->fcgCBAudioEncTiming->Tag = L"NoDirect";
            this->fcgCBAudioEncTiming->Text = L"処理順";
            // 
            // fcgCXAudioEncTiming
            // 
            this->fcgCXAudioEncTiming->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAudioEncTiming->FormattingEnabled = true;
            this->fcgCXAudioEncTiming->Location = System::Drawing::Point(286, 51);
            this->fcgCXAudioEncTiming->Name = L"fcgCXAudioEncTiming";
            this->fcgCXAudioEncTiming->Size = System::Drawing::Size(68, 22);
            this->fcgCXAudioEncTiming->TabIndex = 52;
            this->fcgCXAudioEncTiming->Tag = L"chValue,NoDirect";
            // 
            // fcgCXAudioTempDir
            // 
            this->fcgCXAudioTempDir->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAudioTempDir->FormattingEnabled = true;
            this->fcgCXAudioTempDir->Location = System::Drawing::Point(135, 208);
            this->fcgCXAudioTempDir->Name = L"fcgCXAudioTempDir";
            this->fcgCXAudioTempDir->Size = System::Drawing::Size(150, 22);
            this->fcgCXAudioTempDir->TabIndex = 46;
            this->fcgCXAudioTempDir->Tag = L"chValue";
            // 
            // fcgTXCustomAudioTempDir
            // 
            this->fcgTXCustomAudioTempDir->Location = System::Drawing::Point(64, 236);
            this->fcgTXCustomAudioTempDir->Name = L"fcgTXCustomAudioTempDir";
            this->fcgTXCustomAudioTempDir->Size = System::Drawing::Size(245, 21);
            this->fcgTXCustomAudioTempDir->TabIndex = 47;
            this->fcgTXCustomAudioTempDir->Tag = L"";
            this->fcgTXCustomAudioTempDir->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXCustomAudioTempDir_TextChanged);
            // 
            // fcgBTCustomAudioTempDir
            // 
            this->fcgBTCustomAudioTempDir->Location = System::Drawing::Point(316, 234);
            this->fcgBTCustomAudioTempDir->Name = L"fcgBTCustomAudioTempDir";
            this->fcgBTCustomAudioTempDir->Size = System::Drawing::Size(29, 23);
            this->fcgBTCustomAudioTempDir->TabIndex = 49;
            this->fcgBTCustomAudioTempDir->Tag = L"";
            this->fcgBTCustomAudioTempDir->Text = L"...";
            this->fcgBTCustomAudioTempDir->UseVisualStyleBackColor = true;
            this->fcgBTCustomAudioTempDir->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTCustomAudioTempDir_Click);
            // 
            // fcgCBAudioUsePipe
            // 
            this->fcgCBAudioUsePipe->AutoSize = true;
            this->fcgCBAudioUsePipe->Location = System::Drawing::Point(130, 134);
            this->fcgCBAudioUsePipe->Name = L"fcgCBAudioUsePipe";
            this->fcgCBAudioUsePipe->Size = System::Drawing::Size(73, 18);
            this->fcgCBAudioUsePipe->TabIndex = 42;
            this->fcgCBAudioUsePipe->Tag = L"chValue";
            this->fcgCBAudioUsePipe->Text = L"パイプ処理";
            this->fcgCBAudioUsePipe->UseVisualStyleBackColor = true;
            // 
            // fcgLBAudioBitrate
            // 
            this->fcgLBAudioBitrate->AutoSize = true;
            this->fcgLBAudioBitrate->Location = System::Drawing::Point(284, 161);
            this->fcgLBAudioBitrate->Name = L"fcgLBAudioBitrate";
            this->fcgLBAudioBitrate->Size = System::Drawing::Size(32, 14);
            this->fcgLBAudioBitrate->TabIndex = 50;
            this->fcgLBAudioBitrate->Text = L"kbps";
            // 
            // fcgNUAudioBitrate
            // 
            this->fcgNUAudioBitrate->Location = System::Drawing::Point(212, 157);
            this->fcgNUAudioBitrate->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1536, 0, 0, 0 });
            this->fcgNUAudioBitrate->Name = L"fcgNUAudioBitrate";
            this->fcgNUAudioBitrate->Size = System::Drawing::Size(65, 21);
            this->fcgNUAudioBitrate->TabIndex = 40;
            this->fcgNUAudioBitrate->Tag = L"chValue";
            this->fcgNUAudioBitrate->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgCBAudio2pass
            // 
            this->fcgCBAudio2pass->AutoSize = true;
            this->fcgCBAudio2pass->Location = System::Drawing::Point(59, 134);
            this->fcgCBAudio2pass->Name = L"fcgCBAudio2pass";
            this->fcgCBAudio2pass->Size = System::Drawing::Size(56, 18);
            this->fcgCBAudio2pass->TabIndex = 41;
            this->fcgCBAudio2pass->Tag = L"chValue";
            this->fcgCBAudio2pass->Text = L"2pass";
            this->fcgCBAudio2pass->UseVisualStyleBackColor = true;
            this->fcgCBAudio2pass->CheckedChanged += gcnew System::EventHandler(this, &frmConfig::fcgCBAudio2pass_CheckedChanged);
            // 
            // fcgCXAudioEncMode
            // 
            this->fcgCXAudioEncMode->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAudioEncMode->FormattingEnabled = true;
            this->fcgCXAudioEncMode->Location = System::Drawing::Point(16, 156);
            this->fcgCXAudioEncMode->Name = L"fcgCXAudioEncMode";
            this->fcgCXAudioEncMode->Size = System::Drawing::Size(189, 22);
            this->fcgCXAudioEncMode->TabIndex = 39;
            this->fcgCXAudioEncMode->Tag = L"chValue";
            this->fcgCXAudioEncMode->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgCXAudioEncMode_SelectedIndexChanged);
            // 
            // fcgLBAudioEncMode
            // 
            this->fcgLBAudioEncMode->AutoSize = true;
            this->fcgLBAudioEncMode->Location = System::Drawing::Point(4, 136);
            this->fcgLBAudioEncMode->Name = L"fcgLBAudioEncMode";
            this->fcgLBAudioEncMode->Size = System::Drawing::Size(32, 14);
            this->fcgLBAudioEncMode->TabIndex = 48;
            this->fcgLBAudioEncMode->Text = L"モード";
            // 
            // fcgBTAudioEncoderPath
            // 
            this->fcgBTAudioEncoderPath->Location = System::Drawing::Point(324, 90);
            this->fcgBTAudioEncoderPath->Name = L"fcgBTAudioEncoderPath";
            this->fcgBTAudioEncoderPath->Size = System::Drawing::Size(30, 23);
            this->fcgBTAudioEncoderPath->TabIndex = 38;
            this->fcgBTAudioEncoderPath->Tag = L"";
            this->fcgBTAudioEncoderPath->Text = L"...";
            this->fcgBTAudioEncoderPath->UseVisualStyleBackColor = true;
            this->fcgBTAudioEncoderPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTAudioEncoderPath_Click);
            // 
            // fcgTXAudioEncoderPath
            // 
            this->fcgTXAudioEncoderPath->AllowDrop = true;
            this->fcgTXAudioEncoderPath->Location = System::Drawing::Point(16, 92);
            this->fcgTXAudioEncoderPath->Name = L"fcgTXAudioEncoderPath";
            this->fcgTXAudioEncoderPath->Size = System::Drawing::Size(303, 21);
            this->fcgTXAudioEncoderPath->TabIndex = 37;
            this->fcgTXAudioEncoderPath->Tag = L"";
            this->fcgTXAudioEncoderPath->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXAudioEncoderPath_TextChanged);
            this->fcgTXAudioEncoderPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
            this->fcgTXAudioEncoderPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
            // 
            // fcgLBAudioEncoderPath
            // 
            this->fcgLBAudioEncoderPath->AutoSize = true;
            this->fcgLBAudioEncoderPath->Location = System::Drawing::Point(12, 75);
            this->fcgLBAudioEncoderPath->Name = L"fcgLBAudioEncoderPath";
            this->fcgLBAudioEncoderPath->Size = System::Drawing::Size(49, 14);
            this->fcgLBAudioEncoderPath->TabIndex = 44;
            this->fcgLBAudioEncoderPath->Tag = L"";
            this->fcgLBAudioEncoderPath->Text = L"～の指定";
            // 
            // fcgCBAudioOnly
            // 
            this->fcgCBAudioOnly->AutoSize = true;
            this->fcgCBAudioOnly->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
            this->fcgCBAudioOnly->Location = System::Drawing::Point(252, 5);
            this->fcgCBAudioOnly->Name = L"fcgCBAudioOnly";
            this->fcgCBAudioOnly->Size = System::Drawing::Size(89, 18);
            this->fcgCBAudioOnly->TabIndex = 34;
            this->fcgCBAudioOnly->Tag = L"chValue";
            this->fcgCBAudioOnly->Text = L"音声のみ出力";
            this->fcgCBAudioOnly->UseVisualStyleBackColor = true;
            // 
            // fcgCBFAWCheck
            // 
            this->fcgCBFAWCheck->AutoSize = true;
            this->fcgCBFAWCheck->Location = System::Drawing::Point(252, 28);
            this->fcgCBFAWCheck->Name = L"fcgCBFAWCheck";
            this->fcgCBFAWCheck->Size = System::Drawing::Size(81, 18);
            this->fcgCBFAWCheck->TabIndex = 36;
            this->fcgCBFAWCheck->Tag = L"chValue";
            this->fcgCBFAWCheck->Text = L"FAWCheck";
            this->fcgCBFAWCheck->UseVisualStyleBackColor = true;
            // 
            // fcgCXAudioEncoder
            // 
            this->fcgCXAudioEncoder->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAudioEncoder->FormattingEnabled = true;
            this->fcgCXAudioEncoder->Location = System::Drawing::Point(17, 34);
            this->fcgCXAudioEncoder->Name = L"fcgCXAudioEncoder";
            this->fcgCXAudioEncoder->Size = System::Drawing::Size(172, 22);
            this->fcgCXAudioEncoder->TabIndex = 32;
            this->fcgCXAudioEncoder->Tag = L"chValue";
            this->fcgCXAudioEncoder->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgCXAudioEncoder_SelectedIndexChanged);
            // 
            // fcgLBAudioEncoder
            // 
            this->fcgLBAudioEncoder->AutoSize = true;
            this->fcgLBAudioEncoder->Location = System::Drawing::Point(5, 14);
            this->fcgLBAudioEncoder->Name = L"fcgLBAudioEncoder";
            this->fcgLBAudioEncoder->Size = System::Drawing::Size(48, 14);
            this->fcgLBAudioEncoder->TabIndex = 33;
            this->fcgLBAudioEncoder->Text = L"エンコーダ";
            // 
            // fcgLBAudioTemp
            // 
            this->fcgLBAudioTemp->AutoSize = true;
            this->fcgLBAudioTemp->Location = System::Drawing::Point(7, 211);
            this->fcgLBAudioTemp->Name = L"fcgLBAudioTemp";
            this->fcgLBAudioTemp->Size = System::Drawing::Size(114, 14);
            this->fcgLBAudioTemp->TabIndex = 51;
            this->fcgLBAudioTemp->Tag = L"";
            this->fcgLBAudioTemp->Text = L"音声一時ファイル出力先";
            // 
            // fcgtabPageAudioOther
            // 
            this->fcgtabPageAudioOther->Controls->Add(this->panel2);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgLBBatAfterAudioString);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgLBBatBeforeAudioString);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgBTBatAfterAudioPath);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgTXBatAfterAudioPath);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgLBBatAfterAudioPath);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgCBRunBatAfterAudio);
            this->fcgtabPageAudioOther->Controls->Add(this->panel1);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgBTBatBeforeAudioPath);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgTXBatBeforeAudioPath);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgLBBatBeforeAudioPath);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgCBRunBatBeforeAudio);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgCXAudioPriority);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgLBAudioPriority);
            this->fcgtabPageAudioOther->Location = System::Drawing::Point(4, 23);
            this->fcgtabPageAudioOther->Name = L"fcgtabPageAudioOther";
            this->fcgtabPageAudioOther->Padding = System::Windows::Forms::Padding(3);
            this->fcgtabPageAudioOther->Size = System::Drawing::Size(376, 269);
            this->fcgtabPageAudioOther->TabIndex = 1;
            this->fcgtabPageAudioOther->Tag = L"NoDirect";
            this->fcgtabPageAudioOther->Text = L"その他";
            this->fcgtabPageAudioOther->UseVisualStyleBackColor = true;
            // 
            // panel2
            // 
            this->panel2->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
            this->panel2->Location = System::Drawing::Point(18, 126);
            this->panel2->Name = L"panel2";
            this->panel2->Size = System::Drawing::Size(342, 1);
            this->panel2->TabIndex = 61;
            // 
            // fcgLBBatAfterAudioString
            // 
            this->fcgLBBatAfterAudioString->AutoSize = true;
            this->fcgLBBatAfterAudioString->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Italic | System::Drawing::FontStyle::Underline)),
                System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(128)));
            this->fcgLBBatAfterAudioString->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
            this->fcgLBBatAfterAudioString->Location = System::Drawing::Point(304, 208);
            this->fcgLBBatAfterAudioString->Name = L"fcgLBBatAfterAudioString";
            this->fcgLBBatAfterAudioString->Size = System::Drawing::Size(27, 15);
            this->fcgLBBatAfterAudioString->TabIndex = 60;
            this->fcgLBBatAfterAudioString->Text = L" 後& ";
            this->fcgLBBatAfterAudioString->TextAlign = System::Drawing::ContentAlignment::TopCenter;
            // 
            // fcgLBBatBeforeAudioString
            // 
            this->fcgLBBatBeforeAudioString->AutoSize = true;
            this->fcgLBBatBeforeAudioString->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Italic | System::Drawing::FontStyle::Underline)),
                System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(128)));
            this->fcgLBBatBeforeAudioString->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
            this->fcgLBBatBeforeAudioString->Location = System::Drawing::Point(304, 139);
            this->fcgLBBatBeforeAudioString->Name = L"fcgLBBatBeforeAudioString";
            this->fcgLBBatBeforeAudioString->Size = System::Drawing::Size(27, 15);
            this->fcgLBBatBeforeAudioString->TabIndex = 51;
            this->fcgLBBatBeforeAudioString->Text = L" 前& ";
            this->fcgLBBatBeforeAudioString->TextAlign = System::Drawing::ContentAlignment::TopCenter;
            // 
            // fcgBTBatAfterAudioPath
            // 
            this->fcgBTBatAfterAudioPath->Location = System::Drawing::Point(330, 231);
            this->fcgBTBatAfterAudioPath->Name = L"fcgBTBatAfterAudioPath";
            this->fcgBTBatAfterAudioPath->Size = System::Drawing::Size(30, 23);
            this->fcgBTBatAfterAudioPath->TabIndex = 59;
            this->fcgBTBatAfterAudioPath->Tag = L"chValue";
            this->fcgBTBatAfterAudioPath->Text = L"...";
            this->fcgBTBatAfterAudioPath->UseVisualStyleBackColor = true;
            this->fcgBTBatAfterAudioPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTBatAfterAudioPath_Click);
            // 
            // fcgTXBatAfterAudioPath
            // 
            this->fcgTXBatAfterAudioPath->AllowDrop = true;
            this->fcgTXBatAfterAudioPath->Location = System::Drawing::Point(126, 232);
            this->fcgTXBatAfterAudioPath->Name = L"fcgTXBatAfterAudioPath";
            this->fcgTXBatAfterAudioPath->Size = System::Drawing::Size(202, 21);
            this->fcgTXBatAfterAudioPath->TabIndex = 58;
            this->fcgTXBatAfterAudioPath->Tag = L"chValue";
            this->fcgTXBatAfterAudioPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
            this->fcgTXBatAfterAudioPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
            // 
            // fcgLBBatAfterAudioPath
            // 
            this->fcgLBBatAfterAudioPath->AutoSize = true;
            this->fcgLBBatAfterAudioPath->Location = System::Drawing::Point(40, 236);
            this->fcgLBBatAfterAudioPath->Name = L"fcgLBBatAfterAudioPath";
            this->fcgLBBatAfterAudioPath->Size = System::Drawing::Size(61, 14);
            this->fcgLBBatAfterAudioPath->TabIndex = 57;
            this->fcgLBBatAfterAudioPath->Text = L"バッチファイル";
            // 
            // fcgCBRunBatAfterAudio
            // 
            this->fcgCBRunBatAfterAudio->AutoSize = true;
            this->fcgCBRunBatAfterAudio->Location = System::Drawing::Point(18, 207);
            this->fcgCBRunBatAfterAudio->Name = L"fcgCBRunBatAfterAudio";
            this->fcgCBRunBatAfterAudio->Size = System::Drawing::Size(201, 18);
            this->fcgCBRunBatAfterAudio->TabIndex = 55;
            this->fcgCBRunBatAfterAudio->Tag = L"chValue";
            this->fcgCBRunBatAfterAudio->Text = L"音声エンコード終了後、バッチ処理を行う";
            this->fcgCBRunBatAfterAudio->UseVisualStyleBackColor = true;
            // 
            // panel1
            // 
            this->panel1->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
            this->panel1->Location = System::Drawing::Point(18, 196);
            this->panel1->Name = L"panel1";
            this->panel1->Size = System::Drawing::Size(342, 1);
            this->panel1->TabIndex = 54;
            // 
            // fcgBTBatBeforeAudioPath
            // 
            this->fcgBTBatBeforeAudioPath->Location = System::Drawing::Point(330, 164);
            this->fcgBTBatBeforeAudioPath->Name = L"fcgBTBatBeforeAudioPath";
            this->fcgBTBatBeforeAudioPath->Size = System::Drawing::Size(30, 23);
            this->fcgBTBatBeforeAudioPath->TabIndex = 53;
            this->fcgBTBatBeforeAudioPath->Tag = L"chValue";
            this->fcgBTBatBeforeAudioPath->Text = L"...";
            this->fcgBTBatBeforeAudioPath->UseVisualStyleBackColor = true;
            this->fcgBTBatBeforeAudioPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTBatBeforeAudioPath_Click);
            // 
            // fcgTXBatBeforeAudioPath
            // 
            this->fcgTXBatBeforeAudioPath->AllowDrop = true;
            this->fcgTXBatBeforeAudioPath->Location = System::Drawing::Point(126, 164);
            this->fcgTXBatBeforeAudioPath->Name = L"fcgTXBatBeforeAudioPath";
            this->fcgTXBatBeforeAudioPath->Size = System::Drawing::Size(202, 21);
            this->fcgTXBatBeforeAudioPath->TabIndex = 52;
            this->fcgTXBatBeforeAudioPath->Tag = L"chValue";
            this->fcgTXBatBeforeAudioPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
            this->fcgTXBatBeforeAudioPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
            // 
            // fcgLBBatBeforeAudioPath
            // 
            this->fcgLBBatBeforeAudioPath->AutoSize = true;
            this->fcgLBBatBeforeAudioPath->Location = System::Drawing::Point(40, 167);
            this->fcgLBBatBeforeAudioPath->Name = L"fcgLBBatBeforeAudioPath";
            this->fcgLBBatBeforeAudioPath->Size = System::Drawing::Size(61, 14);
            this->fcgLBBatBeforeAudioPath->TabIndex = 50;
            this->fcgLBBatBeforeAudioPath->Text = L"バッチファイル";
            // 
            // fcgCBRunBatBeforeAudio
            // 
            this->fcgCBRunBatBeforeAudio->AutoSize = true;
            this->fcgCBRunBatBeforeAudio->Location = System::Drawing::Point(18, 139);
            this->fcgCBRunBatBeforeAudio->Name = L"fcgCBRunBatBeforeAudio";
            this->fcgCBRunBatBeforeAudio->Size = System::Drawing::Size(201, 18);
            this->fcgCBRunBatBeforeAudio->TabIndex = 48;
            this->fcgCBRunBatBeforeAudio->Tag = L"chValue";
            this->fcgCBRunBatBeforeAudio->Text = L"音声エンコード開始前、バッチ処理を行う";
            this->fcgCBRunBatBeforeAudio->UseVisualStyleBackColor = true;
            // 
            // fcgCXAudioPriority
            // 
            this->fcgCXAudioPriority->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAudioPriority->FormattingEnabled = true;
            this->fcgCXAudioPriority->Location = System::Drawing::Point(156, 20);
            this->fcgCXAudioPriority->Name = L"fcgCXAudioPriority";
            this->fcgCXAudioPriority->Size = System::Drawing::Size(136, 22);
            this->fcgCXAudioPriority->TabIndex = 47;
            this->fcgCXAudioPriority->Tag = L"chValue";
            // 
            // fcgLBAudioPriority
            // 
            this->fcgLBAudioPriority->AutoSize = true;
            this->fcgLBAudioPriority->Location = System::Drawing::Point(29, 23);
            this->fcgLBAudioPriority->Name = L"fcgLBAudioPriority";
            this->fcgLBAudioPriority->Size = System::Drawing::Size(62, 14);
            this->fcgLBAudioPriority->TabIndex = 46;
            this->fcgLBAudioPriority->Text = L"音声優先度";
            // 
            // fcgtabPageAvqsvAudio
            // 
            this->fcgtabPageAvqsvAudio->Controls->Add(this->fcgLBAvqsvAudioBitrate2);
            this->fcgtabPageAvqsvAudio->Controls->Add(this->fcgNUAvqsvAudioBitrate);
            this->fcgtabPageAvqsvAudio->Controls->Add(this->fcgLBAvqsvAudioBitrate);
            this->fcgtabPageAvqsvAudio->Controls->Add(this->fcgCXAvqsvAudioEncoder);
            this->fcgtabPageAvqsvAudio->Controls->Add(this->fcgLBAvqsvAudioEncoder);
            this->fcgtabPageAvqsvAudio->Location = System::Drawing::Point(4, 23);
            this->fcgtabPageAvqsvAudio->Name = L"fcgtabPageAvqsvAudio";
            this->fcgtabPageAvqsvAudio->Size = System::Drawing::Size(376, 269);
            this->fcgtabPageAvqsvAudio->TabIndex = 2;
            this->fcgtabPageAvqsvAudio->Text = L"音声";
            this->fcgtabPageAvqsvAudio->UseVisualStyleBackColor = true;
            // 
            // fcgLBAvqsvAudioBitrate2
            // 
            this->fcgLBAvqsvAudioBitrate2->AutoSize = true;
            this->fcgLBAvqsvAudioBitrate2->Location = System::Drawing::Point(175, 99);
            this->fcgLBAvqsvAudioBitrate2->Name = L"fcgLBAvqsvAudioBitrate2";
            this->fcgLBAvqsvAudioBitrate2->Size = System::Drawing::Size(32, 14);
            this->fcgLBAvqsvAudioBitrate2->TabIndex = 55;
            this->fcgLBAvqsvAudioBitrate2->Text = L"kbps";
            // 
            // fcgNUAvqsvAudioBitrate
            // 
            this->fcgNUAvqsvAudioBitrate->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 32, 0, 0, 0 });
            this->fcgNUAvqsvAudioBitrate->Location = System::Drawing::Point(100, 97);
            this->fcgNUAvqsvAudioBitrate->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1536, 0, 0, 0 });
            this->fcgNUAvqsvAudioBitrate->Name = L"fcgNUAvqsvAudioBitrate";
            this->fcgNUAvqsvAudioBitrate->Size = System::Drawing::Size(65, 21);
            this->fcgNUAvqsvAudioBitrate->TabIndex = 53;
            this->fcgNUAvqsvAudioBitrate->Tag = L"chValue";
            this->fcgNUAvqsvAudioBitrate->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBAvqsvAudioBitrate
            // 
            this->fcgLBAvqsvAudioBitrate->AutoSize = true;
            this->fcgLBAvqsvAudioBitrate->Location = System::Drawing::Point(36, 99);
            this->fcgLBAvqsvAudioBitrate->Name = L"fcgLBAvqsvAudioBitrate";
            this->fcgLBAvqsvAudioBitrate->Size = System::Drawing::Size(54, 14);
            this->fcgLBAvqsvAudioBitrate->TabIndex = 54;
            this->fcgLBAvqsvAudioBitrate->Text = L"ビットレート";
            // 
            // fcgCXAvqsvAudioEncoder
            // 
            this->fcgCXAvqsvAudioEncoder->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAvqsvAudioEncoder->FormattingEnabled = true;
            this->fcgCXAvqsvAudioEncoder->Location = System::Drawing::Point(99, 49);
            this->fcgCXAvqsvAudioEncoder->Name = L"fcgCXAvqsvAudioEncoder";
            this->fcgCXAvqsvAudioEncoder->Size = System::Drawing::Size(172, 22);
            this->fcgCXAvqsvAudioEncoder->TabIndex = 51;
            this->fcgCXAvqsvAudioEncoder->Tag = L"chValue";
            // 
            // fcgLBAvqsvAudioEncoder
            // 
            this->fcgLBAvqsvAudioEncoder->AutoSize = true;
            this->fcgLBAvqsvAudioEncoder->Location = System::Drawing::Point(33, 52);
            this->fcgLBAvqsvAudioEncoder->Name = L"fcgLBAvqsvAudioEncoder";
            this->fcgLBAvqsvAudioEncoder->Size = System::Drawing::Size(48, 14);
            this->fcgLBAvqsvAudioEncoder->TabIndex = 52;
            this->fcgLBAvqsvAudioEncoder->Text = L"エンコーダ";
            // 
            // fcggroupBoxAvqsv
            // 
            this->fcggroupBoxAvqsv->Controls->Add(this->fcgLBTrimInfo);
            this->fcggroupBoxAvqsv->Controls->Add(this->fcgLBTrim);
            this->fcggroupBoxAvqsv->Controls->Add(this->fcgCBTrim);
            this->fcggroupBoxAvqsv->Controls->Add(this->fcgLBAvqsvEncWarn);
            this->fcggroupBoxAvqsv->Controls->Add(this->fcgBTAvqsvInputFile);
            this->fcggroupBoxAvqsv->Controls->Add(this->fcgTXAvqsvInputFile);
            this->fcggroupBoxAvqsv->Controls->Add(this->fcgLBAvqsvInputFile);
            this->fcggroupBoxAvqsv->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcggroupBoxAvqsv->Location = System::Drawing::Point(4, 27);
            this->fcggroupBoxAvqsv->Name = L"fcggroupBoxAvqsv";
            this->fcggroupBoxAvqsv->Size = System::Drawing::Size(997, 96);
            this->fcggroupBoxAvqsv->TabIndex = 52;
            this->fcggroupBoxAvqsv->TabStop = false;
            // 
            // fcgLBTrimInfo
            // 
            this->fcgLBTrimInfo->AutoSize = true;
            this->fcgLBTrimInfo->Location = System::Drawing::Point(761, 35);
            this->fcgLBTrimInfo->Name = L"fcgLBTrimInfo";
            this->fcgLBTrimInfo->Size = System::Drawing::Size(73, 14);
            this->fcgLBTrimInfo->TabIndex = 135;
            this->fcgLBTrimInfo->Text = L"カット編集情報";
            // 
            // fcgLBTrim
            // 
            this->fcgLBTrim->AutoSize = true;
            this->fcgLBTrim->Location = System::Drawing::Point(629, 35);
            this->fcgLBTrim->Name = L"fcgLBTrim";
            this->fcgLBTrim->Size = System::Drawing::Size(51, 14);
            this->fcgLBTrim->TabIndex = 134;
            this->fcgLBTrim->Text = L"カット編集";
            // 
            // fcgCBTrim
            // 
            this->fcgCBTrim->AutoSize = true;
            this->fcgCBTrim->Location = System::Drawing::Point(722, 37);
            this->fcgCBTrim->Name = L"fcgCBTrim";
            this->fcgCBTrim->Size = System::Drawing::Size(15, 14);
            this->fcgCBTrim->TabIndex = 133;
            this->fcgCBTrim->Tag = L"chValue";
            this->fcgCBTrim->UseVisualStyleBackColor = true;
            // 
            // fcgLBAvqsvEncWarn
            // 
            this->fcgLBAvqsvEncWarn->AutoSize = true;
            this->fcgLBAvqsvEncWarn->ForeColor = System::Drawing::Color::Red;
            this->fcgLBAvqsvEncWarn->Location = System::Drawing::Point(52, 64);
            this->fcgLBAvqsvEncWarn->Name = L"fcgLBAvqsvEncWarn";
            this->fcgLBAvqsvEncWarn->Size = System::Drawing::Size(346, 14);
            this->fcgLBAvqsvEncWarn->TabIndex = 132;
            this->fcgLBAvqsvEncWarn->Text = L"本モードを有効にした場合、Aviutlでのフィルタ類による編集は反映されません。";
            // 
            // fcgBTAvqsvInputFile
            // 
            this->fcgBTAvqsvInputFile->Location = System::Drawing::Point(437, 29);
            this->fcgBTAvqsvInputFile->Name = L"fcgBTAvqsvInputFile";
            this->fcgBTAvqsvInputFile->Size = System::Drawing::Size(30, 23);
            this->fcgBTAvqsvInputFile->TabIndex = 46;
            this->fcgBTAvqsvInputFile->Text = L"...";
            this->fcgBTAvqsvInputFile->UseVisualStyleBackColor = true;
            // 
            // fcgTXAvqsvInputFile
            // 
            this->fcgTXAvqsvInputFile->AllowDrop = true;
            this->fcgTXAvqsvInputFile->Location = System::Drawing::Point(127, 30);
            this->fcgTXAvqsvInputFile->Name = L"fcgTXAvqsvInputFile";
            this->fcgTXAvqsvInputFile->Size = System::Drawing::Size(303, 21);
            this->fcgTXAvqsvInputFile->TabIndex = 45;
            this->fcgTXAvqsvInputFile->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXDirectInputFile_TextChanged);
            // 
            // fcgLBAvqsvInputFile
            // 
            this->fcgLBAvqsvInputFile->AutoSize = true;
            this->fcgLBAvqsvInputFile->Location = System::Drawing::Point(22, 33);
            this->fcgLBAvqsvInputFile->Name = L"fcgLBAvqsvInputFile";
            this->fcgLBAvqsvInputFile->Size = System::Drawing::Size(90, 14);
            this->fcgLBAvqsvInputFile->TabIndex = 47;
            this->fcgLBAvqsvInputFile->Text = L"入力ファイルの指定";
            // 
            // fcgCBAvqsv
            // 
            this->fcgCBAvqsv->AutoSize = true;
            this->fcgCBAvqsv->Location = System::Drawing::Point(11, 30);
            this->fcgCBAvqsv->Name = L"fcgCBAvqsv";
            this->fcgCBAvqsv->Size = System::Drawing::Size(167, 19);
            this->fcgCBAvqsv->TabIndex = 89;
            this->fcgCBAvqsv->Tag = L"chValue";
            this->fcgCBAvqsv->Text = L"avqsvで直接エンコードを行う";
            this->fcgCBAvqsv->UseVisualStyleBackColor = true;
            this->fcgCBAvqsv->CheckedChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeVisibleDirectEnc);
            // 
            // frmConfig
            // 
            this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
            this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
            this->ClientSize = System::Drawing::Size(1008, 577);
            this->Controls->Add(this->fcgtabControlAudio);
            this->Controls->Add(this->fcgLBguiExBlog);
            this->Controls->Add(this->fcgtabControlMux);
            this->Controls->Add(this->fcgtabControlQSV);
            this->Controls->Add(this->fcgLBVersion);
            this->Controls->Add(this->fcgLBVersionDate);
            this->Controls->Add(this->fcgBTDefault);
            this->Controls->Add(this->fcgBTOK);
            this->Controls->Add(this->fcgBTCancel);
            this->Controls->Add(this->fcgTXCmd);
            this->Controls->Add(this->fcgtoolStripSettings);
            this->Controls->Add(this->fcgCBAvqsv);
            this->Controls->Add(this->fcggroupBoxAvqsv);
            this->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
            this->MaximizeBox = false;
            this->Name = L"frmConfig";
            this->ShowIcon = false;
            this->Text = L"Aviutl 出力 プラグイン";
            this->FormClosed += gcnew System::Windows::Forms::FormClosedEventHandler(this, &frmConfig::frmConfig_FormClosed);
            this->Load += gcnew System::EventHandler(this, &frmConfig::frmConfig_Load);
            this->fcgtoolStripSettings->ResumeLayout(false);
            this->fcgtoolStripSettings->PerformLayout();
            this->fcgtabControlMux->ResumeLayout(false);
            this->fcgtabPageMP4->ResumeLayout(false);
            this->fcgtabPageMP4->PerformLayout();
            this->fcgtabPageMKV->ResumeLayout(false);
            this->fcgtabPageMKV->PerformLayout();
            this->fcgtabPageMPG->ResumeLayout(false);
            this->fcgtabPageMPG->PerformLayout();
            this->fcgtabPageMux->ResumeLayout(false);
            this->fcgtabPageMux->PerformLayout();
            this->fcgtabPageBat->ResumeLayout(false);
            this->fcgtabPageBat->PerformLayout();
            this->fcgtabPageMuxInternal->ResumeLayout(false);
            this->fcgtabPageMuxInternal->PerformLayout();
            this->fcgtabControlQSV->ResumeLayout(false);
            this->tabPageVideoEnc->ResumeLayout(false);
            this->tabPageVideoEnc->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUWinBRCSize))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMax))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMin))->EndInit();
            this->fcgPNQVBR->ResumeLayout(false);
            this->fcgPNQVBR->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQVBR))->EndInit();
            this->fcgPNICQ->ResumeLayout(false);
            this->fcgPNICQ->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUICQQuality))->EndInit();
            this->fcgPNLookahead->ResumeLayout(false);
            this->fcgPNLookahead->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNULookaheadDepth))->EndInit();
            this->fcgPNAVBR->ResumeLayout(false);
            this->fcgPNAVBR->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAVBRAccuarcy))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAVBRConvergence))->EndInit();
            this->fcgPNBitrate->ResumeLayout(false);
            this->fcgPNBitrate->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUBitrate))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUMaxkbps))->EndInit();
            this->fcgPNQP->ResumeLayout(false);
            this->fcgPNQP->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPI))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPP))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPB))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUSlices))->EndInit();
            this->fcggroupBoxColor->ResumeLayout(false);
            this->fcggroupBoxColor->PerformLayout();
            this->fcgGroupBoxAspectRatio->ResumeLayout(false);
            this->fcgGroupBoxAspectRatio->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAspectRatioY))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAspectRatioX))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUGopLength))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUBframes))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNURef))->EndInit();
            this->tabPageVpp->ResumeLayout(false);
            this->tabPageVpp->PerformLayout();
            this->fcggroupBoxDetail->ResumeLayout(false);
            this->fcggroupBoxDetail->PerformLayout();
            this->fcgPNExtSettings->ResumeLayout(false);
            this->fcgPNExtSettings->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUMVSearchWindow))->EndInit();
            this->fcggroupBoxVpp->ResumeLayout(false);
            this->fcggroupBoxVpp->PerformLayout();
            this->fcggroupBoxVppDetail->ResumeLayout(false);
            this->fcggroupBoxVppDetail->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDetail))->EndInit();
            this->fcggroupBoxVppDenoise->ResumeLayout(false);
            this->fcggroupBoxVppDenoise->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoise))->EndInit();
            this->fcggroupBoxVppResize->ResumeLayout(false);
            this->fcggroupBoxVppResize->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppResizeW))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppResizeH))->EndInit();
            this->tabPageExOpt->ResumeLayout(false);
            this->tabPageExOpt->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUInputBufSize))->EndInit();
            this->tabPageFeatures->ResumeLayout(false);
            this->tabPageFeatures->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgDGVFeatures))->EndInit();
            this->fcgCSExeFiles->ResumeLayout(false);
            this->fcgtabControlAudio->ResumeLayout(false);
            this->fcgtabPageAudioMain->ResumeLayout(false);
            this->fcgtabPageAudioMain->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAudioBitrate))->EndInit();
            this->fcgtabPageAudioOther->ResumeLayout(false);
            this->fcgtabPageAudioOther->PerformLayout();
            this->fcgtabPageAvqsvAudio->ResumeLayout(false);
            this->fcgtabPageAvqsvAudio->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAvqsvAudioBitrate))->EndInit();
            this->fcggroupBoxAvqsv->ResumeLayout(false);
            this->fcggroupBoxAvqsv->PerformLayout();
            this->ResumeLayout(false);
            this->PerformLayout();

        }
#pragma endregion
    private:
        delegate void SetCPUInfoDelegate();
        String^ StrCPUInfo;
        SetCPUInfoDelegate ^getCPUInfoDelegate;
        delegate void SetGPUInfoDelegate();
        String^ StrGPUInfo;
        SetGPUInfoDelegate ^getGPUInfoDelegate;
        const SYSTEM_DATA *sys_dat;
        CONF_GUIEX *conf;
        AUO_LINK_PARAM *conf_link_prm;
        LocalSettings LocalStg;
        bool CurrentPipeEnabled;
        bool stgChanged;
        String^ CurrentStgDir;
        ToolStripMenuItem^ CheckedStgMenuItem;
        CONF_GUIEX *cnf_stgSelected;
        String^ lastQualityStr;
        SaveFileDialog^ saveFileQSVFeautures;
        QSVFeatures^ featuresHW;
        QSVFeatures^ featuresSW;
#ifdef HIDE_MPEG2
        TabPage^ tabPageMpgMux;
#endif
    private:
        System::Int32 GetCurrentAudioDefaultBitrate();
        delegate System::Void qualityTimerChangeDelegate();
        System::Void InitComboBox();
        System::Void setAudioDisplay();
        System::Void AudioEncodeModeChanged();
        System::Void InitStgFileList();
        System::Void RebuildStgFileDropDown(String^ stgDir);
        System::Void RebuildStgFileDropDown(ToolStripDropDownItem^ TS, String^ dir);
        System::Void SetLocalStg();
        System::Void LoadLocalStg();
        System::Void SaveLocalStg();
        System::Boolean CheckLocalStg();
        System::Void SetTXMaxLen(TextBox^ TX, int max_len);
        System::Void SetTXMaxLenAll();
        System::Void InitForm();
        System::Void ConfToFrm(CONF_GUIEX *cnf);
        System::Void FrmToConf(CONF_GUIEX *cnf);
        System::Void SetChangedEvent(Control^ control, System::EventHandler^ _event);
        System::Void fcgCXOutputType_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e);
        System::Void SetAllCheckChangedEvents(Control ^top);
        System::Void SaveToStgFile(String^ stgName);
        System::Void DeleteStgFile(ToolStripMenuItem^ mItem);
        System::Boolean EnableSettingsNoteChange(bool Enable);
        System::Void fcgTSLSettingsNotes_DoubleClick(System::Object^  sender, System::EventArgs^  e);
        System::Void fcgTSTSettingsNotes_Leave(System::Object^  sender, System::EventArgs^  e);
        System::Void fcgTSTSettingsNotes_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e);
        System::Void fcgTSTSettingsNotes_TextChanged(System::Object^  sender, System::EventArgs^  e);
        System::Void GetfcgTSLSettingsNotes(char *notes, int nSize);
        System::Void SetfcgTSLSettingsNotes(const char *notes);
        System::Void SetfcgTSLSettingsNotes(String^ notes);
        System::Void fcgTSBSave_Click(System::Object^  sender, System::EventArgs^  e);
        System::Void fcgTSBSaveNew_Click(System::Object^  sender, System::EventArgs^  e);
        System::Void fcgTSBDelete_Click(System::Object^  sender, System::EventArgs^  e);
        System::Void fcgTSSettings_DropDownItemClicked(System::Object^  sender, System::Windows::Forms::ToolStripItemClickedEventArgs^  e);
        System::Void UncheckAllDropDownItem(ToolStripItem^ mItem);
        ToolStripMenuItem^ fcgTSSettingsSearchItem(String^ stgPath, ToolStripItem^ mItem);
        ToolStripMenuItem^ fcgTSSettingsSearchItem(String^ stgPath);
        System::Void CheckTSSettingsDropDownItem(ToolStripMenuItem^ mItem);
        System::Void CheckTSItemsEnabled(CONF_GUIEX *current_conf);
        System::Void fcgChangeMuxerVisible(System::Object^  sender, System::EventArgs^  e);
        System::Void SetHelpToolTips();
        System::Void ShowExehelp(String^ ExePath, String^ args);
        System::Void fcgTSBOtherSettings_Click(System::Object^  sender, System::EventArgs^  e);
        System::Void fcgChangeEnabled(System::Object^  sender, System::EventArgs^  e);
        System::Void fcgTSBBitrateCalc_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
        System::Void CloseBitrateCalc();
        System::Void SetfbcBTABEnable(bool enable, int max);
        System::Void SetfbcBTVBEnable(bool enable);
        System::Void fcgCBAudio2pass_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
        System::Void fcgCXAudioEncoder_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e);
        System::Void fcgCXAudioEncMode_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e);
        System::Void AdjustLocation();
        System::Void ActivateToolTip(bool Enable);
        System::Void SetStgEscKey(bool Enable);
        System::Boolean CheckVppResolution(CONF_GUIEX *conf);
        System::Void SetToolStripEvents(ToolStrip^ TS, System::Windows::Forms::MouseEventHandler^ _event);
        System::Void SetInputBufRange();
        System::Boolean fcgCheckLibRateControl(mfxU32 mfxlib_current, mfxU64 available_features);
        System::Void fcgCheckLibVersion(mfxU32 mfxlib_current, mfxU64 available_features);
        System::Boolean fcgCheckRCModeLibVersion(int rc_mode_target, int rc_mode_replace, bool mode_supported);
        System::Void UpdateMfxLibDetection();
        System::Void UpdateFeatures();
        System::Void fcgCheckVppFeatures();
        System::Void fcgCBHWLibChanged(System::Object^  sender, System::EventArgs^  e);
        System::Void SaveQSVFeatureAsImg(String^ SavePath);
        System::Void SaveQSVFeatureAsTxt(String^ SavePath);
        System::Void SaveQSVFeature();
        System::Void SetCPUInfo();
        System::Void SetGPUInfo();
        System::Void CheckQSVLink(CONF_GUIEX *conf);
        System::Void ChangeVisiableDirectEncPerControl(Control ^top, bool visible);
        System::Void fcgChangeVisibleDirectEnc(System::Object^  sender, System::EventArgs^  e);
    public:
        System::Void InitData(CONF_GUIEX *set_config, const SYSTEM_DATA *system_data);
        System::Void SetVideoBitrate(int bitrate);
        System::Void SetAudioBitrate(int bitrate);
        System::Void InformfbcClosed();
    private:
        System::Void frmConfig_FormClosed(System::Object^  sender, System::Windows::Forms::FormClosedEventArgs^  e) {
            if (featuresHW != nullptr) {
                delete featuresHW;
                featuresHW = nullptr;
            }
            if (featuresSW != nullptr) {
                delete featuresSW;
                featuresSW = nullptr;
            }
        }
    private:
        System::Void fcgTSItem_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
            EnableSettingsNoteChange(false);
        }
    private:
        System::Void frmConfig_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
            if (e->KeyCode == Keys::Escape)
                this->Close();
            if ((e->KeyData & (Keys::Control | Keys::Shift | Keys::Enter)) == (Keys::Control | Keys::Shift | Keys::Enter))
                fcgBTOK_Click(sender, nullptr);
        }
    private:
        System::Void NUSelectAll(System::Object^  sender, System::EventArgs^  e) {
             NumericUpDown^ NU = dynamic_cast<NumericUpDown^>(sender);
             NU->Select(0, NU->Text->Length);
         }
    private:
        System::Void setComboBox(ComboBox^ CX, const X264_OPTION_STR * list) {
            CX->BeginUpdate();
            CX->Items->Clear();
            for (int i = 0; list[i].desc; i++)
                CX->Items->Add(String(list[i].desc).ToString());
            CX->EndUpdate();
        }
    private:
        System::Void setComboBox(ComboBox^ CX, const CX_DESC * list) {
            CX->BeginUpdate();
            CX->Items->Clear();
            for (int i = 0; list[i].desc; i++)
                CX->Items->Add(String(list[i].desc).ToString());
            CX->EndUpdate();
        }
    private:
        System::Void setComboBox(ComboBox^ CX, const char * const * list) {
            CX->BeginUpdate();
            CX->Items->Clear();
            for (int i = 0; list[i]; i++)
                CX->Items->Add(String(list[i]).ToString());
            CX->EndUpdate();
        }
    private:
        System::Void setComboBox(ComboBox^ CX, const WCHAR * const * list) {
            CX->BeginUpdate();
            CX->Items->Clear();
            for (int i = 0; list[i]; i++)
                CX->Items->Add(String(list[i]).ToString());
            CX->EndUpdate();
        }
    private:
        System::Void setPriorityList(ComboBox^ CX) {
            CX->BeginUpdate();
            CX->Items->Clear();
            for (int i = 0; priority_table[i].text; i++)
                CX->Items->Add(String(priority_table[i].text).ToString());
            CX->EndUpdate();
        }
    private:
        System::Void setMuxerCmdExNames(ComboBox^ CX, int muxer_index) {
            CX->BeginUpdate();
            CX->Items->Clear();
            MUXER_SETTINGS *mstg = &sys_dat->exstg->s_mux[muxer_index];
            for (int i = 0; i < mstg->ex_count; i++)
                CX->Items->Add(String(mstg->ex_cmd[i].name).ToString());
            CX->EndUpdate();
        }
    private:
        System::Void setAudioEncoderNames() {
            fcgCXAudioEncoder->BeginUpdate();
            fcgCXAudioEncoder->Items->Clear();
            //fcgCXAudioEncoder->Items->AddRange(reinterpret_cast<array<String^>^>(LocalStg.audEncName->ToArray(String::typeid)));
            fcgCXAudioEncoder->Items->AddRange(LocalStg.audEncName->ToArray());
            fcgCXAudioEncoder->EndUpdate();
        }
    private:
        System::Void TX_LimitbyBytes(System::Object^  sender, System::ComponentModel::CancelEventArgs^ e) {
            int maxLength = 0;
            int stringBytes = 0;
            TextBox^ TX = nullptr;
            if ((TX = dynamic_cast<TextBox^>(sender)) == nullptr)
                return;
            stringBytes = CountStringBytes(TX->Text);
            maxLength = TX->MaxLength;
            if (stringBytes > maxLength - 1) {
                e->Cancel = true;
                MessageBox::Show(this, L"入力された文字数が多すぎます。減らしてください。", L"エラー", MessageBoxButtons::OK, MessageBoxIcon::Error);
            }
        }
    private:
        System::Boolean openExeFile(TextBox^ TX, String^ ExeName) {
            //WinXPにおいて、OpenFileDialogはCurrentDirctoryを勝手に変更しやがるので、
            //一度保存し、あとから再適用する
            String^ CurrentDir = Directory::GetCurrentDirectory();
            OpenFileDialog ofd;
            ofd.Multiselect = false;
            ofd.FileName = ExeName;
            ofd.Filter = MakeExeFilter(ExeName);
            if (Directory::Exists(LocalStg.LastAppDir))
                ofd.InitialDirectory = Path::GetFullPath(LocalStg.LastAppDir);
            else if (File::Exists(TX->Text))
                ofd.InitialDirectory = Path::GetFullPath(Path::GetDirectoryName(TX->Text));
            else
                ofd.InitialDirectory = String(sys_dat->aviutl_dir).ToString();
            bool ret = (ofd.ShowDialog() == System::Windows::Forms::DialogResult::OK);
            if (ret) {
                if (sys_dat->exstg->s_local.get_relative_path)
                    ofd.FileName = GetRelativePath(ofd.FileName, CurrentDir);
                LocalStg.LastAppDir = Path::GetDirectoryName(Path::GetFullPath(ofd.FileName));
                TX->Text = ofd.FileName;
                TX->SelectionStart = TX->Text->Length;
            }
            Directory::SetCurrentDirectory(CurrentDir);
            return ret;
        }
    private: 
        System::Void fcgBTMP4MuxerPath_Click(System::Object^  sender, System::EventArgs^  e) {
            openExeFile(fcgTXMP4MuxerPath, LocalStg.MP4MuxerExeName);
        }
    private: 
        System::Void fcgBTTC2MP4Path_Click(System::Object^  sender, System::EventArgs^  e) {
            openExeFile(fcgTXTC2MP4Path, LocalStg.TC2MP4ExeName);
        }
    private:
        System::Void fcgBTMP4RawMuxerPath_Click(System::Object^  sender, System::EventArgs^  e) {
            openExeFile(fcgTXMP4RawPath, LocalStg.MP4RawExeName);
        }
    private: 
        System::Void fcgBTAudioEncoderPath_Click(System::Object^  sender, System::EventArgs^  e) {
            int index = fcgCXAudioEncoder->SelectedIndex;
            openExeFile(fcgTXAudioEncoderPath, LocalStg.audEncExeName[index]);
        }
    private: 
        System::Void fcgBTMKVMuxerPath_Click(System::Object^  sender, System::EventArgs^  e) {
            openExeFile(fcgTXMKVMuxerPath, LocalStg.MKVMuxerExeName);
        }
    private:
        System::Void fcgBTMPGMuxerPath_Click(System::Object^  sender, System::EventArgs^  e) {
            openExeFile(fcgTXMPGMuxerPath, LocalStg.MPGMuxerExeName);
        }
    private:
        System::Void openTempFolder(TextBox^ TX) {
            FolderBrowserDialog^ fbd = fcgfolderBrowserTemp;
            if (Directory::Exists(TX->Text))
                fbd->SelectedPath = TX->Text;
            if (fbd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
                if (sys_dat->exstg->s_local.get_relative_path)
                    fbd->SelectedPath = GetRelativePath(fbd->SelectedPath);
                TX->Text = fbd->SelectedPath;
                TX->SelectionStart = TX->Text->Length;
            }
        }
    private: 
        System::Void fcgBTCustomAudioTempDir_Click(System::Object^  sender, System::EventArgs^  e) {
            openTempFolder(fcgTXCustomAudioTempDir);
        }
    private: 
        System::Void fcgBTMP4BoxTempDir_Click(System::Object^  sender, System::EventArgs^  e) {
            openTempFolder(fcgTXMP4BoxTempDir);
        }
    private: 
        System::Void fcgBTCustomTempDir_Click(System::Object^  sender, System::EventArgs^  e) {
            openTempFolder(fcgTXCustomTempDir);
        }
    private:
        System::Boolean openAndSetFilePath(TextBox^ TX, String^ fileTypeName) {
            return openAndSetFilePath(TX, fileTypeName, nullptr, nullptr);
        }
    private:
        System::Boolean openAndSetFilePath(TextBox^ TX, String^ fileTypeName, String^ ext) {
            return openAndSetFilePath(TX, fileTypeName, ext, nullptr);
        }
    private:
        System::Boolean openAndSetFilePath(TextBox^ TX, String^ fileTypeName, String^ ext, String^ dir) {
            //WinXPにおいて、OpenFileDialogはCurrentDirctoryを勝手に変更しやがるので、
            //一度保存し、あとから再適用する
            String^ CurrentDir = Directory::GetCurrentDirectory();
            //設定
            if (ext == nullptr)
                ext = L".*";
            OpenFileDialog^ ofd = fcgOpenFileDialog;
            ofd->FileName = L"";
            if (dir != nullptr && Directory::Exists(dir))
                ofd->InitialDirectory = dir;
            if (TX->Text->Length) {
                String^ fileName = nullptr;
                try {
                    fileName = Path::GetFileName(TX->Text);
                } catch (...) {
                    //invalid charによる例外は破棄
                }
                if (fileName != nullptr)
                    ofd->FileName = fileName;
            }
            ofd->Multiselect = false;
            ofd->Filter = fileTypeName + L"(*" + ext + L")|*" + ext;
            bool ret = (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK);
            if (ret) {
                if (sys_dat->exstg->s_local.get_relative_path)
                    ofd->FileName = GetRelativePath(ofd->FileName, CurrentDir);
                TX->Text = ofd->FileName;
                TX->SelectionStart = TX->Text->Length;
            }
            Directory::SetCurrentDirectory(CurrentDir);
            return ret;
        }
    private:
        System::Void fcgBTBatAfterPath_Click(System::Object^  sender, System::EventArgs^  e) {
            if (openAndSetFilePath(fcgTXBatAfterPath, L"バッチファイル", ".bat", LocalStg.LastBatDir))
                LocalStg.LastBatDir = Path::GetDirectoryName(fcgTXBatAfterPath->Text);
        }
    private:
        System::Void fcgBTBatBeforePath_Click(System::Object^  sender, System::EventArgs^  e) {
            if (openAndSetFilePath(fcgTXBatBeforePath, L"バッチファイル", ".bat", LocalStg.LastBatDir))
                LocalStg.LastBatDir = Path::GetDirectoryName(fcgTXBatBeforePath->Text);
        }
    private:
        System::Void fcgBTBatBeforeAudioPath_Click(System::Object^  sender, System::EventArgs^  e) {
            if (openAndSetFilePath(fcgTXBatBeforeAudioPath, L"バッチファイル", ".bat", LocalStg.LastBatDir))
                LocalStg.LastBatDir = Path::GetDirectoryName(fcgTXBatBeforeAudioPath->Text);
        }
    private:
        System::Void fcgBTBatAfterAudioPath_Click(System::Object^  sender, System::EventArgs^  e) {
            if (openAndSetFilePath(fcgTXBatAfterAudioPath, L"バッチファイル", ".bat", LocalStg.LastBatDir))
                LocalStg.LastBatDir = Path::GetDirectoryName(fcgTXBatAfterAudioPath->Text);
        }
    private:
        System::Void SetCXIndex(ComboBox^ CX, int index) {
            CX->SelectedIndex = clamp(index, 0, CX->Items->Count - 1);
        }
    private:
        System::Void SetNUValue(NumericUpDown^ NU, Decimal d) {
            NU->Value = clamp(d, NU->Minimum, NU->Maximum);
        }
    private:
        System::Void SetNUValue(NumericUpDown^ NU, int i) {
            NU->Value = clamp(Convert::ToDecimal(i), NU->Minimum, NU->Maximum);
        }
    private:
        System::Void SetNUValue(NumericUpDown^ NU, unsigned int i) {
            NU->Value = clamp(Convert::ToDecimal(i), NU->Minimum, NU->Maximum);
        }
    private:
        System::Void SetNUValue(NumericUpDown^ NU, float f) {
            NU->Value = clamp(Convert::ToDecimal(f), NU->Minimum, NU->Maximum);
        }
    private: 
        System::Void frmConfig_Load(System::Object^  sender, System::EventArgs^  e) {
            InitForm();
        }
    private: 
        System::Void fcgBTOK_Click(System::Object^  sender, System::EventArgs^  e) {
            if (CheckLocalStg())
                return;
            init_CONF_GUIEX(conf, false);
            FrmToConf(conf);
            if (CheckVppResolution(conf))
                return;
            SaveLocalStg();
            ZeroMemory(conf->oth.notes, sizeof(conf->oth.notes));
            this->Close();
        }
    private: 
        System::Void fcgBTCancel_Click(System::Object^  sender, System::EventArgs^  e) {
            this->Close();
        }
    private: 
        System::Void fcgBTDefault_Click(System::Object^  sender, System::EventArgs^  e) {
            CONF_GUIEX confDefault;
            init_CONF_GUIEX(&confDefault, FALSE);
            ConfToFrm(&confDefault);
        }
    private:
        System::Void ChangePresetNameDisplay(bool changed) {
            if (CheckedStgMenuItem != nullptr) {
                fcgTSSettings->Text = (changed) ? L"[" + CheckedStgMenuItem->Text + L"]*" : CheckedStgMenuItem->Text;
                fcgTSBSave->Enabled = changed;
            }
        }
    private:
        System::Void CheckOtherChanges(System::Object^  sender, System::EventArgs^  e) {
            if (CheckedStgMenuItem == nullptr)
                return;
            CONF_GUIEX check_change;
            init_CONF_GUIEX(&check_change, FALSE);
            FrmToConf(&check_change);
            ChangePresetNameDisplay(memcmp(&check_change, cnf_stgSelected, sizeof(CONF_GUIEX)) != 0);
        }
    private: 
        System::Void fcgTXCmd_DoubleClick(System::Object^  sender, System::EventArgs^  e) {
            int offset = (fcgTXCmd->Multiline) ? -fcgTXCmdfulloffset : fcgTXCmdfulloffset;
            fcgTXCmd->Height += offset;
            this->Height += offset;
            fcgTXCmd->Multiline = !fcgTXCmd->Multiline;
        }
    private: 
        System::Void fcgTSSettings_Click(System::Object^  sender, System::EventArgs^  e) {
            if (EnableSettingsNoteChange(false))
                fcgTSSettings->ShowDropDown();
        }
    private: 
        System::Void fcgTXAudioEncoderPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            LocalStg.audEncPath[fcgCXAudioEncoder->SelectedIndex] = fcgTXAudioEncoderPath->Text;
            fcgBTAudioEncoderPath->ContextMenuStrip = (File::Exists(fcgTXAudioEncoderPath->Text)) ? fcgCSExeFiles : nullptr;
        }
    private: 
        System::Void fcgTXMP4MuxerPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            LocalStg.MP4MuxerPath = fcgTXMP4MuxerPath->Text;
            fcgBTMP4MuxerPath->ContextMenuStrip = (File::Exists(fcgTXMP4MuxerPath->Text)) ? fcgCSExeFiles : nullptr;
        }
    private: 
        System::Void fcgTXTC2MP4Path_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            LocalStg.TC2MP4Path = fcgTXTC2MP4Path->Text;
            fcgBTTC2MP4Path->ContextMenuStrip = (File::Exists(fcgTXTC2MP4Path->Text)) ? fcgCSExeFiles : nullptr;
        }
    private:
        System::Void fcgTXMP4RawMuxerPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            LocalStg.MP4RawPath = fcgTXMP4RawPath->Text;
            fcgBTMP4RawPath->ContextMenuStrip = (File::Exists(fcgTXMP4RawPath->Text)) ? fcgCSExeFiles : nullptr;
        }
    private: 
        System::Void fcgTXMKVMuxerPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            LocalStg.MKVMuxerPath = fcgTXMKVMuxerPath->Text;
            fcgBTMKVMuxerPath->ContextMenuStrip = (File::Exists(fcgTXMKVMuxerPath->Text)) ? fcgCSExeFiles : nullptr;
        }
    private:
        System::Void fcgTXMPGMuxerPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            LocalStg.MPGMuxerPath = fcgTXMPGMuxerPath->Text;
            fcgBTMPGMuxerPath->ContextMenuStrip = (File::Exists(fcgTXMPGMuxerPath->Text)) ? fcgCSExeFiles : nullptr;
        }
    private: 
        System::Void fcgTXMP4BoxTempDir_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            LocalStg.CustomMP4TmpDir = fcgTXMP4BoxTempDir->Text;
        }
    private: 
        System::Void fcgTXCustomAudioTempDir_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            LocalStg.CustomAudTmpDir = fcgTXCustomAudioTempDir->Text;
        }
    private: 
        System::Void fcgTXCustomTempDir_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            LocalStg.CustomTmpDir = fcgTXCustomTempDir->Text;
        }
    private:
        System::Void fcgTXDirectInputFile_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            LocalStg.AuoLinkSrcPath = fcgTXAvqsvInputFile->Text;
        }
    private:
        System::Void fcgSetDragDropFilename_Enter(System::Object^  sender, DragEventArgs^  e) {
            e->Effect = (e->Data->GetDataPresent(DataFormats::FileDrop)) ? DragDropEffects::Copy : DragDropEffects::None;
        }
    private:
        System::Void fcgSetDragDropFilename_DragDrop(System::Object^  sender, DragEventArgs^  e) {
            TextBox^ TX = dynamic_cast<TextBox^>(sender);
            array<System::String ^>^ filelist = dynamic_cast<array<System::String ^>^>(e->Data->GetData(DataFormats::FileDrop, false));
            if (filelist == nullptr || TX == nullptr)
                return;
            String^ filePath = filelist[0]; //複数だった場合は先頭のものを使用
            if (sys_dat->exstg->s_local.get_relative_path)
                filePath = GetRelativePath(filePath);
            TX->Text = filePath;
        }
    private:
        System::Void fcgTSExeFileshelp_Click(System::Object^  sender, System::EventArgs^  e) {
            System::Windows::Forms::ToolStripMenuItem^ TS = dynamic_cast<System::Windows::Forms::ToolStripMenuItem^>(sender);
            if (TS == nullptr) return;
            System::Windows::Forms::ContextMenuStrip^ CS = dynamic_cast<System::Windows::Forms::ContextMenuStrip^>(TS->Owner);
            if (CS == nullptr) return;

            //Name, args, Path の順番
            array<ExeControls>^ ControlList = {
                { fcgBTAudioEncoderPath->Name,   fcgTXAudioEncoderPath->Text,   sys_dat->exstg->s_aud[fcgCXAudioEncoder->SelectedIndex].cmd_help },
                { fcgBTMP4MuxerPath->Name,       fcgTXMP4MuxerPath->Text,       sys_dat->exstg->s_mux[MUXER_MP4].help_cmd },
                { fcgBTTC2MP4Path->Name,         fcgTXTC2MP4Path->Text,         sys_dat->exstg->s_mux[MUXER_TC2MP4].help_cmd },
                { fcgBTMP4RawPath->Name,         fcgTXMP4RawPath->Text,         sys_dat->exstg->s_mux[MUXER_MP4_RAW].help_cmd },
                { fcgBTMKVMuxerPath->Name,       fcgTXMKVMuxerPath->Text,       sys_dat->exstg->s_mux[MUXER_MKV].help_cmd },
                { fcgBTMPGMuxerPath->Name,       fcgTXMPGMuxerPath->Text,       sys_dat->exstg->s_mux[MUXER_MPG].help_cmd }
            };
            for (int i = 0; i < ControlList->Length; i++) {
                if (NULL == String::Compare(CS->SourceControl->Name, ControlList[i].Name)) {
                    ShowExehelp(ControlList[i].Path, String(ControlList[i].args).ToString());
                    return;
                }
            }
            MessageBox::Show(L"ヘルプ表示用のコマンドが設定されていません。", L"エラー", MessageBoxButtons::OK, MessageBoxIcon::Error);
        }
    private:
        System::Void fcgLBguiExBlog_LinkClicked(System::Object^  sender, System::Windows::Forms::LinkLabelLinkClickedEventArgs^  e) {
            fcgLBguiExBlog->LinkVisited = true;
            try {
                System::Diagnostics::Process::Start(String(sys_dat->exstg->blog_url).ToString());
            } catch (...) {
                //いちいちメッセージとか出さない
            };
        }
    private:
        System::Void fcgDGVFeatures_CellFormatting(System::Object^  sender, System::Windows::Forms::DataGridViewCellFormattingEventArgs^  e) {
            if (e->ColumnIndex)
                e->CellStyle->BackColor = (0 <= Convert::ToString(e->Value)->IndexOf(L"×")) ? Color::LightSalmon : Color::LightGreen;
        }
    private:
        System::Void fcgBTSaveFeatureList_Click(System::Object^  sender, System::EventArgs^  e) {
            SaveQSVFeature();
        }
};
}
