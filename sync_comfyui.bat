@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "SOURCE_DIR=%cd%"
set "LOG_FILE=%~dp0sync_comfyui.log"
set "TARGET_DIR=E:\draw\ComfyUI_windows_portable\ComfyUI\custom_nodes\Comfyui_Consistency_xiling"

call :log "=============================="
call :log "Start local sync in %SOURCE_DIR%"
call :log "Target directory: %TARGET_DIR%"

if not exist "%TARGET_DIR%" (
  mkdir "%TARGET_DIR%" >> "%LOG_FILE%" 2>&1
  if errorlevel 1 (
    call :log "[ERROR] Failed to create target directory: %TARGET_DIR%"
    exit /b 1
  )
)

robocopy "%SOURCE_DIR%" "%TARGET_DIR%" /E /XD ".git" "__pycache__" /XF "*.pyc" "*.bat" "sync_comfyui.log" /R:2 /W:1 /NFL /NDL /NP >> "%LOG_FILE%" 2>&1
if errorlevel 8 (
  call :log "[ERROR] robocopy failed. Check log for details."
  exit /b 1
)

call :log "Copied files to %TARGET_DIR%"
call :log "Local sync done."
exit /b 0

:log
echo [%date% %time%] %~1
>>"%LOG_FILE%" echo [%date% %time%] %~1
exit /b 0
