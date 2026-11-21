# GapLink live v1.3: detector-aware fine-tuning -> chon tren VAL -> test 1 lan.
# PowerShell:  powershell -ExecutionPolicy Bypass -File D:\gaplink\run_v13_pipeline.ps1   (chay lai duoc, tu resume)
$ErrorActionPreference = "Stop"
Set-Location D:\gaplink
$py = "D:\gaplink\.venv\Scripts\python.exe"
$log = "D:\gaplink\runs\pipeline_v13.log"
$seed = 20261121
$run = "runs\live_v13_s$seed"
function Step($name, [scriptblock]$cmd) {
    "`n==== $name  $(Get-Date -Format s) ====" | Tee-Object -FilePath $log -Append
    $ErrorActionPreference = "Continue"
    & $cmd 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { "FAILED: $name (exit $LASTEXITCODE)" | Tee-Object -FilePath $log -Append; exit 1 }
}
Step "0. unit tests" { & D:\gaplink\.venv\Scripts\pytest.exe -c pytest.ini tests -q -p no:cacheprovider }
if (-not (git tag -l v1.3-prereg)) {
    Step "1. commit PREREG v1.3 + code BEFORE generating data / training" {
        git add docs/PREREG_live_v1.3.md docs/DEVIATIONS.md src/det_events.py tests/test_det_events.py scripts/gen_det_events.py scripts/finetune_live.py scripts/eval_live_v12.py run_v12_pipeline.ps1 run_v12b_pipeline.ps1 run_v13_pipeline.ps1
        git commit -m "PREREG live v1.3: detector-aware fine-tuning plan, candidate set and selection rule fixed before training"
        git tag -a v1.3-prereg -m "v1.3 plan locked before data generation and training"
    }
}
if (-not (Test-Path data\cache\det_events_val.pkl)) {
    Step "2. detector-aware events on TRAIN + VAL (YOLO, resumable)" { & $py -m scripts.gen_det_events }
}
if (-not (Test-Path release\live_v1.3.pt)) {
    Step "3. fine-tune from final.pt (select ckpt on YOLO-val events)" { & $py -m scripts.finetune_live --seed $seed }
    Copy-Item "$run\best.pt" release\live_v1.3.pt
    (Get-FileHash release\live_v1.3.pt -Algorithm SHA256).Hash.ToLower() | Set-Content -NoNewline release\live_v1.3.sha256
}
if (-not (git tag -l v1.3-live-selected)) {
    Step "4. VAL sweep {final, v1.3} x tau + baselines" { & $py -m scripts.eval_live_v12 --select --version v13 --ckpt release\live_v1.3.pt }
    Step "5. commit checkpoint + val selection BEFORE test" {
        Copy-Item "$run\train_log.json" release\live_v1.3_train_log.json
        git add -f release/live_v1.3.pt
        git add release/live_v1.3.sha256 release/live_v1.3.yaml release/live_v1.3_train_log.json
        git commit -m "v1.3-live: detector-aware checkpoint + operating points selected on val"
        git tag -a v1.3-live-selected -m "v1.3 checkpoint and live operating points frozen before the single test run"
    }
}
if (-not (Test-Path runs\live_v13_test\live_v13_results.json)) {
    Step "6. v1.3 SINGLE live test run (+ parity with v1.1)" { & $py -m scripts.eval_live_v12 --test --version v13 }
}
"`nALL DONE $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
