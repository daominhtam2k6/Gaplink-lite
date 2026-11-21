# Tim khoanh khac demo (GapLink v1.3 vs C-3) tren 40 video test. ~20-40 phut, chay lai duoc.
# PowerShell:  powershell -ExecutionPolicy Bypass -File D:\gaplink\run_mine_demo.ps1
$ErrorActionPreference = "Continue"
Set-Location D:\gaplink
& D:\gaplink\.venv\Scripts\python.exe -m scripts.mine_demo_moments --baseline C-3 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath D:\gaplink\runs\mine_demo.log
