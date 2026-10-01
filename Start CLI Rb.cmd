@echo off
chcp 65001 >nul
set "CLI_RB_DEFAULT_REPO=%~dp0"
if exist "%~dp0.env.local" set "CLI_RB_ENV_FILE=%~dp0.env.local"
if not defined CLI_RB_ENV_FILE if exist "%~dp0.env.sub2api" set "CLI_RB_ENV_FILE=%~dp0.env.sub2api"
if not exist "%~dp0dist\CLI-Rb-Windows\runtime\python.exe" (
  echo 请先生成 Windows 便携包，或使用解压目录中的 Start CLI Rb.cmd。
  pause
  exit /b 1
)
"%~dp0dist\CLI-Rb-Windows\runtime\python.exe" -X utf8 -m cli_rb.interface.launcher %*
