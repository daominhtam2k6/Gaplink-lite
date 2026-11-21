# Render lai 3 video kich ban demo voi cau hinh live v1.3 (vs C-2 manh nhat, va vs C-3 kieu SORT/DeepSORT).
# PowerShell:  powershell -ExecutionPolicy Bypass -File D:\gaplink\run_demo_v13.ps1
$ErrorActionPreference = "Continue"
Set-Location D:\gaplink
$py = "D:\gaplink\.venv\Scripts\python.exe"
$log = "D:\gaplink\runs\demo_v13.log"
"==== demo v1.3 $(Get-Date -Format s) ====" | Tee-Object -FilePath $log
& D:\gaplink\.venv\Scripts\pytest.exe -c pytest.ini tests -q -p no:cacheprovider 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { "FAILED: tests" | Tee-Object -FilePath $log -Append; exit 1 }
foreach ($b in @("C-2", "C-3")) {
    "---- render vs $b ----" | Tee-Object -FilePath $log -Append
    & $py demo\render_security_demo.py --baseline $b 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { "FAILED: render $b" | Tee-Object -FilePath $log -Append; exit 1 }
}
"DONE $(Get-Date -Format s)  -> demo\out\v1.3_vs_C-2\, demo\out\v1.3_vs_C-3\" | Tee-Object -FilePath $log -Append
