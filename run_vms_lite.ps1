# GapLink-VMS (ban gon, 3 video). Dung: Ctrl+C.
# powershell -ExecutionPolicy Bypass -File D:\gaplink\run_vms_lite.ps1
$ErrorActionPreference = "Continue"
Set-Location D:\gaplink
Start-Job -ScriptBlock { Start-Sleep -Seconds 6; Start-Process "http://127.0.0.1:8010" } | Out-Null
& D:\gaplink\.venv\Scripts\python.exe -m demo.vms_lite --port 8010
