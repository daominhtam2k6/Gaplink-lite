# Thu nghiem ngoai mien: xe may tren VisDrone2019-MOT (docs/PREREG_ood_motor.md)
# Truoc khi chay: giai nen VisDrone2019-MOT-val.zip va VisDrone2019-MOT-test-dev.zip vao D:\gaplink\data\raw\visdrone\
# PowerShell:  powershell -ExecutionPolicy Bypass -File D:\gaplink\scripts\pipelines\run_ood_motor.ps1
# Chay lai duoc: buoc nao da co ket qua se bo qua. Buoc --eval chi chay MOT lan.
$ErrorActionPreference = "Continue"
Set-Location D:\gaplink
$py  = "D:\gaplink\.venv\Scripts\python.exe"
$log = "D:\gaplink\runs\ood_visdrone.log"
New-Item -ItemType Directory -Force -Path D:\gaplink\runs | Out-Null

function Step($name, [scriptblock]$cmd) {
    "=== $name  $(Get-Date -Format 'HH:mm:ss') ===" | Tee-Object -FilePath $log -Append
    $ErrorActionPreference = "Continue"
    & $cmd 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { "!!! $name FAILED (exit $LASTEXITCODE)" | Tee-Object -FilePath $log -Append; exit 1 }
}

foreach ($d in "VisDrone2019-MOT-val", "VisDrone2019-MOT-test-dev") {
    if (-not (Test-Path "D:\gaplink\data\raw\visdrone\$d\annotations")) {
        "Thieu D:\gaplink\data\raw\visdrone\$d\annotations - hay giai nen zip truoc." | Tee-Object -FilePath $log -Append; exit 1
    }
}

Step "pytest" { & $py -m pytest -q -p no:cacheprovider tests\test_ood_visdrone.py }
if (-not (Test-Path D:\gaplink\data\ood_visdrone\seqs.csv)) {
    Step "convert" { & $py -m scripts.convert_visdrone }
}
Step "hist motor" { & $py -m scripts.extract_hist_ood --group motor --workers 6 }
Step "hist car"   { & $py -m scripts.extract_hist_ood --group car --workers 6 }
if (-not (Test-Path D:\gaplink\data\ood_visdrone\motor\events.pkl)) {
    Step "make events" { & $py -m scripts.eval_ood_motor --make-events }
}
if (-not (Test-Path D:\gaplink\runs\ood_visdrone\results.json)) {
    Step "eval (single run)" { & $py -m scripts.eval_ood_motor --eval }
}
"=== DONE $(Get-Date -Format 'HH:mm:ss') - xem runs\ood_visdrone\summary.md ===" | Tee-Object -FilePath $log -Append
