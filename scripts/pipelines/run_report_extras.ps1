# Bo sung bao cao: phan tich loi (PREREG §4) + render demo v1.3 (cap nhat anh giao dien trong bao cao).
# PowerShell:  powershell -ExecutionPolicy Bypass -File D:\gaplink\scripts\pipelines\run_report_extras.ps1
$ErrorActionPreference = "Continue"
Set-Location D:\gaplink
$py = "D:\gaplink\.venv\Scripts\python.exe"
$log = "D:\gaplink\runs\report_extras.log"
function Step($name, [scriptblock]$cmd) {
    "`n==== $name  $(Get-Date -Format s) ====" | Tee-Object -FilePath $log -Append
    & $cmd 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { "FAILED: $name (exit $LASTEXITCODE)" | Tee-Object -FilePath $log -Append; exit 1 }
}
Step "0. unit tests" { & D:\gaplink\.venv\Scripts\pytest.exe -c pytest.ini tests -q -p no:cacheprovider }
Step "1. failure analysis (frozen predictions, PREREG §4 cases)" { & $py -m scripts.failure_analysis }
Step "2. demo v1.3 vs C-3 (updates report UI figures)" { & $py demo\render_security_demo.py --baseline C-3 --update-report-figs }
Step "3. demo v1.3 vs C-2" { & $py demo\render_security_demo.py --baseline C-2 }
"`nALL DONE $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
