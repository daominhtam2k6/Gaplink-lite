# GapLink v1.2 pipeline - chay 1 lan, co the chay lai (resume tung buoc).
# PowerShell:  powershell -ExecutionPolicy Bypass -File <project>\scripts\pipelines\run_v12_pipeline.ps1
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path   # project root, wherever the repo is unpacked
Set-Location $root
$py = "$root\.venv\Scripts\python.exe"
$log = "$root\runs\pipeline_v12.log"
New-Item -ItemType Directory -Force $root\runs | Out-Null
function Step($name, [scriptblock]$cmd) {
    "`n==== $name  $(Get-Date -Format s) ====" | Tee-Object -FilePath $log -Append
    # native tools (git, python) write warnings to stderr; judge success by exit code only
    $ErrorActionPreference = "Continue"
    & $cmd 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { "FAILED: $name (exit $LASTEXITCODE)" | Tee-Object -FilePath $log -Append; exit 1 }
}
Step "0. unit tests" { & "$root\.venv\Scripts\pytest.exe" -c pytest.ini tests -q }
Step "1. verify v1.1 freeze" { & $py scripts\freeze_results.py }
Step "2. D5 location bootstrap (event + tracker, frozen ckpt, no threshold change)" { & $py -m scripts.rebootstrap_location }
if (-not (Test-Path release\live_v1.2.yaml)) {
    Step "3. v1.2 live operating point on VAL (YOLO sweep, resumable)" { & $py -m scripts.eval_live_v12 --select }
    Step "3b. commit val selection BEFORE test" {
        git add src/live_eval.py src/location_bootstrap.py scripts/eval_live_v12.py scripts/rebootstrap_location.py tests/test_live_eval.py tests/test_location_bootstrap.py release/live_v1.2.yaml release/results_v1.1/location_bootstrap.json run_v12_pipeline.ps1
        git commit -m "v1.2-live: operating points selected on val (PREREG §3 rule), location bootstrap D5"
        git tag -a v1.2-live-selected -m "live operating points frozen before test"
    }
}
if (-not (Test-Path runs\live_v12_test\live_v12_results.json)) {
    Step "4. v1.2 SINGLE test run (+ parity with v1.1)" { & $py -m scripts.eval_live_v12 --test }
}
"`nALL DONE $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
