@echo off
set QSVENCC_PATH=exe_files\QSVEncC\x64\QSVEncC64.exe
if "%PROCESSOR_ARCHITECTURE%" == "x86" (
    set QSVENCC_PATH=exe_files\QSVEncC\x86\QSVEncC.exe
)
%QSVENCC_PATH% --check-features-html