# GapLink v1.2b: them bien the GapLink co cong Gaussian, chon tren VAL, roi test 1 lan.
# Chay SAU KHI run_v12_pipeline.ps1 da dung o buoc 4 ("test run blocked").
# PowerShell:  powershell -ExecutionPolicy Bypass -File <project>\scripts\pipelines\run_v12b_pipeline.ps1
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path   # project root, wherever the repo is unpacked
Set-Location $root
$py = "$root\.venv\Scripts\python.exe"
$log = "$root\runs\pipeline_v12b.log"
function Step($name, [scriptblock]$cmd) {
    "`n==== $name  $(Get-Date -Format s) ====" | Tee-Object -FilePath $log -Append
    # native tools (git, python) write warnings to stderr; judge success by exit code only
    $ErrorActionPreference = "Continue"
    & $cmd 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { "FAILED: $name (exit $LASTEXITCODE)" | Tee-Object -FilePath $log -Append; exit 1 }
}
Step "0. unit tests" { & "$root\.venv\Scripts\pytest.exe" -c pytest.ini tests -q -p no:cacheprovider }
if (-not (git tag -l v1.2b-prereg)) {
    Step "1. commit PREREG_live_v1.2 + gated code BEFORE evaluating it" {
        git add docs/PREREG_live_v1.2.md docs/DEVIATIONS.md src/gated_assoc.py src/live_eval.py src/location_bootstrap.py tests/test_gated_assoc.py tests/test_live_eval.py tests/test_location_bootstrap.py scripts/eval_live_v12.py scripts/rebootstrap_location.py run_v12_pipeline.ps1 run_v12b_pipeline.ps1 release/FROZEN_RESULTS.md release/results_v1.1/location_bootstrap.json report/latex/chapter01.tex report/latex/chapter01_vi.tex report/latex/chapter03.tex report/latex/chapter03_vi.tex
        git commit -m "D5 location-bootstrap CIs (event+tracker) in report; PREREG live v1.2 with Gaussian-gated GapLink candidates, rule fixed before evaluation"
        git tag -a v1.2b-prereg -m "candidate set + rule locked before gated val results"
    }
}
if (-not (git tag -l v1.2b-live-selected)) {
    Step "2. VAL sweep incl. gated variants (resumes cached configs)" { & $py -m scripts.eval_live_v12 --select }
    Step "3. commit val selection BEFORE test" {
        git add release/live_v1.2.yaml
        git commit -m "v1.2b-live: operating points selected on val (gated candidates included)"
        git tag -a v1.2b-live-selected -m "live operating points frozen before the single v1.2 test run"
    }
}
if (Test-Path release\live_v1.2.HOLD) { Remove-Item release\live_v1.2.HOLD }
if (-not (Test-Path runs\live_v12_test\live_v12_results.json)) {
    Step "4. v1.2 SINGLE live test run (+ parity with v1.1)" { & $py -m scripts.eval_live_v12 --test }
}
"`nALL DONE $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
