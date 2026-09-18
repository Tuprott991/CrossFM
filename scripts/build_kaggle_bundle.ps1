param(
    [ValidateSet('smoke','phase1','smoke_v2','phase1_v2','phase2_smoke','phase2','phase25_smoke','phase25','phase275_smoke','phase275','phase3_smoke','phase3_smoke_vast','phase3')]
    [string]$Profile = 'smoke',
    [string]$KernelSlug = 'crossfm-phase-1-smoke-t4x2',
    [string]$KernelTitle = 'CrossFM Phase 1 Smoke T4x2'
)

$ErrorActionPreference = 'Stop'
function Write-Utf8NoBom([string]$Path, [string]$Content) {
    [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
}
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$isPhase2 = $Profile.StartsWith('phase2')
$isPhase25 = $Profile.StartsWith('phase25')
$isPhase275 = $Profile.StartsWith('phase275')
$isPhase3 = $Profile.StartsWith('phase3')
$stagingName = if ($Profile -eq 'phase3_smoke_vast') { 'dist\vast_phase3' } elseif ($isPhase3) { 'dist\kaggle_phase3' } elseif ($isPhase275) { 'dist\kaggle_phase275' } elseif ($isPhase25) { 'dist\kaggle_phase25' } elseif ($isPhase2) { 'dist\kaggle_phase2' } else { 'dist\kaggle' }
$distRoot = [System.IO.Path]::GetFullPath((Join-Path $repo $stagingName))
$expectedRoot = [System.IO.Path]::GetFullPath((Join-Path $repo 'dist'))
if (-not $distRoot.StartsWith($expectedRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe staging path: $distRoot"
}
if (Test-Path -LiteralPath $distRoot) {
    $resolvedExisting = (Resolve-Path -LiteralPath $distRoot).Path
    if ($resolvedExisting -ne $distRoot) { throw "Staging path resolution mismatch" }
    Remove-Item -LiteralPath $resolvedExisting -Recurse -Force
}
$datasetDir = New-Item -ItemType Directory -Path (Join-Path $distRoot 'dataset') -Force
$kernelDir = New-Item -ItemType Directory -Path (Join-Path $distRoot 'kernel') -Force
$wheelBuild = New-Item -ItemType Directory -Path (Join-Path $distRoot 'wheel') -Force

$commit = (git -C $repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'A Git commit is required before building' }
$dirty = [bool](git -C $repo status --porcelain)
if ($dirty) { throw 'Refusing to freeze a dirty source tree; commit the experiment first' }

python -m pip wheel $repo --no-deps --wheel-dir $wheelBuild.FullName
if ($LASTEXITCODE -ne 0) { throw 'Wheel build failed' }
$wheel = Get-ChildItem -LiteralPath $wheelBuild.FullName -Filter 'crossfm-*.whl'
if ($wheel.Count -ne 1) { throw "Expected one CrossFM wheel, found $($wheel.Count)" }
$wheelTarget = Join-Path $datasetDir.FullName $wheel.Name
Copy-Item -LiteralPath $wheel.FullName -Destination $wheelTarget

$configSource = Join-Path $repo "configs\$Profile.yaml"
$configText = Get-Content -LiteralPath $configSource -Raw
$configText = $configText.Replace('git_commit: BUILD_TIME', "git_commit: $commit").Replace('dirty: BUILD_TIME', 'dirty: false')
$configTarget = Join-Path $datasetDir.FullName 'frozen_experiment.yaml'
[System.IO.File]::WriteAllText($configTarget, $configText, [System.Text.UTF8Encoding]::new($false))

$wheelHash = (Get-FileHash -LiteralPath $wheelTarget -Algorithm SHA256).Hash.ToLowerInvariant()
$configHash = (Get-FileHash -LiteralPath $configTarget -Algorithm SHA256).Hash.ToLowerInvariant()
$manifest = [ordered]@{
    schema_version = '1.0.0'
    git_commit = $commit
    source_dirty = $false
    wheel = [ordered]@{ name = [System.IO.Path]::GetFileName($wheelTarget); sha256 = $wheelHash; bytes = (Get-Item $wheelTarget).Length }
    config = [ordered]@{ name = 'frozen_experiment.yaml'; sha256 = $configHash; bytes = (Get-Item $configTarget).Length }
    dependencies = @('numpy==1.26.4','pandas==2.2.3','PyYAML==6.0.2','scikit-learn==1.6.1','tabicl==2.2.0','transformers==4.57.6','huggingface-hub==0.36.0','safetensors==0.7.0')
}
Write-Utf8NoBom (Join-Path $datasetDir.FullName 'bundle_manifest.json') ($manifest | ConvertTo-Json -Depth 5)

$datasetMetadata = [ordered]@{
    title = if ($isPhase3) { 'CrossFM Phase 3 Immutable Bundle' } elseif ($isPhase275) { 'CrossFM Phase 2.75 Immutable Bundle' } elseif ($isPhase25) { 'CrossFM Phase 2.5 Immutable Bundle' } elseif ($isPhase2) { 'CrossFM Phase 2 Immutable Bundle' } else { 'CrossFM Phase 1 Immutable Bundle' }
    id = if ($isPhase3) { 'tuktuai/crossfm-phase3-bundle' } elseif ($isPhase275) { 'tuktuai/crossfm-phase275-bundle' } elseif ($isPhase25) { 'tuktuai/crossfm-phase25-bundle' } elseif ($isPhase2) { 'tuktuai/crossfm-phase2-bundle' } else { 'tuktuai/crossfm-phase1-bundle' }
    licenses = @([ordered]@{ name = 'other' })
    isPrivate = $true
}
Write-Utf8NoBom (Join-Path $datasetDir.FullName 'dataset-metadata.json') ($datasetMetadata | ConvertTo-Json -Depth 4)
$driverName = if ($isPhase3) { 'kaggle_phase3_driver.py' } elseif ($isPhase2) { 'kaggle_phase2_driver.py' } else { 'kaggle_driver.py' }
Copy-Item -LiteralPath (Join-Path $repo "scripts\$driverName") -Destination (Join-Path $kernelDir.FullName 'driver.py')
$kernelMetadata = [ordered]@{
    id = "tuktuai/$KernelSlug"
    title = $KernelTitle
    code_file = 'driver.py'
    language = 'python'
    kernel_type = 'script'
    is_private = $true
    enable_gpu = $true
    enable_internet = $true
    machine_shape = 'NvidiaTeslaT4'
    dataset_sources = @($(if ($isPhase3) { 'tuktuai/crossfm-phase3-bundle' } elseif ($isPhase275) { 'tuktuai/crossfm-phase275-bundle' } elseif ($isPhase25) { 'tuktuai/crossfm-phase25-bundle' } elseif ($isPhase2) { 'tuktuai/crossfm-phase2-bundle' } else { 'tuktuai/crossfm-phase1-bundle' }))
    competition_sources = @()
    kernel_sources = @()
    model_sources = @()
}
Write-Utf8NoBom (Join-Path $kernelDir.FullName 'kernel-metadata.json') ($kernelMetadata | ConvertTo-Json -Depth 4)
python (Join-Path $repo 'scripts\validate_kaggle_bundle.py') $distRoot
if ($LASTEXITCODE -ne 0) { throw 'Kaggle bundle validation failed' }
Write-Output "Built $Profile bundle at $distRoot"
Write-Output "wheel_sha256=$wheelHash"
Write-Output "config_sha256=$configHash"
