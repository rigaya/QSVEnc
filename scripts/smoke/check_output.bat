@echo off
setlocal EnableExtensions
rem QSVEnc after-process bat entry
rem Set this bat full path in QSVEnc "after encode bat"
rem Macros are expanded by QSVEnc before run

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=python"

if defined SMOKE_PYTHON (
  set "PYTHON_EXE=%SMOKE_PYTHON%"
)

"%PYTHON_EXE%" "%SCRIPT_DIR%check_output.py" --savpath "%{savpath}" --logpath "%{logpath}" --config "%SCRIPT_DIR%smoke_config.json"
set "ERR=%ERRORLEVEL%"

if not "%ERR%"=="0" (
  echo [smoke] FAIL exit=%ERR%
) else (
  echo [smoke] PASS
)

exit /b %ERR%
