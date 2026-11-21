# Mo GapLink-VMS (v1.3 vs C-3) va tu mo trinh duyet. Dung: Ctrl+C trong cua so nay.
# PowerShell:  powershell -ExecutionPolicy Bypass -File D:\gaplink\run_vms.ps1            (GapLink tren CPU)
#              powershell -ExecutionPolicy Bypass -File D:\gaplink\run_vms.ps1 -GlDevice cuda   (mac dinh: cpu, nhanh hon tren RTX 3050)
param([string]$GlDevice = "")
$ErrorActionPreference = "Continue"
Set-Location D:\gaplink
if ($GlDevice -ne "") { $env:GAPLINK_GL_DEVICE = $GlDevice } else { Remove-Item Env:GAPLINK_GL_DEVICE -ErrorAction SilentlyContinue }
Start-Job -ScriptBlock { Start-Sleep -Seconds 6; Start-Process "http://127.0.0.1:8000" } | Out-Null
& D:\gaplink\.venv\Scripts\python.exe demo\security_vms_app.py --port 8000
