param(
    [Parameter(Mandatory=$true)][string]$Profile,
    [Parameter(Mandatory=$true)][string]$KaggleOwner,
    [Parameter(Mandatory=$true)][string]$KernelSlug,
    [string]$KernelTitle = 'CrossFM-Align Full-Paper Experiment',
    [string]$DataOwner = $KaggleOwner,
    [string]$DataSlug = 'crossfm-fullpaper-data'
)

$ErrorActionPreference = 'Stop'
function Write-Utf8NoBom([string]$Path, [string]$Content) {
    [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$distBase = [System.IO.Path]::GetFullPath((Join-Path $repo 'dist\fullpaper'))
$pathProfile = $Profile -replace '[^a-zA-Z0-9_.-]', '-'
$slugProfile = $Profile -replace '[^a-zA-Z0-9.-]', '-'
$distRoot = [System.IO.Path]::GetFullPath((Join-Path $distBase $pathProfile))
if (-not $distRoot.StartsWith($distBase + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe staging path: $distRoot"
}
$commit = (git -C $repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'A Git commit is required before building' }
if (git -C $repo status --porcelain) { throw 'Commit all experiment code before freezing a bundle' }

$rawConfig = Get-Content -LiteralPath (Join-Path $repo 'configs\fullpaper.yaml') -Raw
$env:CROSSFM_BUILD_PROFILE = $Profile
$env:CROSSFM_BUILD_CONFIG = (Join-Path $repo 'configs\fullpaper.yaml')
try {
    $profileJson = python -c "import json,os,yaml; c=yaml.safe_load(open(os.environ['CROSSFM_BUILD_CONFIG'], encoding='utf-8')); p=os.environ['CROSSFM_BUILD_PROFILE']; datasets=c['profiles'][p]['datasets']; local=any(c['datasets'][d]['loader']!='beyondarena' for d in datasets); beyond=any(c['datasets'][d]['loader']=='beyondarena' for d in datasets); print(json.dumps({'protocol':c['experiment']['protocol_id'],'accelerator':c['profiles'][p]['accelerator'],'requires_local_data':local,'has_beyondarena':beyond,'methods':c['profiles'][p]['methods']}))"
    if ($LASTEXITCODE -ne 0) { throw "Unknown full-paper profile: $Profile" }
} finally {
    Remove-Item Env:CROSSFM_BUILD_PROFILE -ErrorAction SilentlyContinue
    Remove-Item Env:CROSSFM_BUILD_CONFIG -ErrorAction SilentlyContinue
}
$profileInfo = $profileJson | ConvertFrom-Json
if ($profileInfo.accelerator -ne 'kaggle_t4x2') {
    throw "Profile $Profile is not a Kaggle T4x2 profile"
}

if (Test-Path -LiteralPath $distRoot) {
    $resolved = (Resolve-Path -LiteralPath $distRoot).Path
    if ($resolved -ne $distRoot) { throw 'Staging path resolution mismatch' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
$datasetDir = New-Item -ItemType Directory -Path (Join-Path $distRoot 'dataset') -Force
$kernelDir = New-Item -ItemType Directory -Path (Join-Path $distRoot 'kernel') -Force
$wheelDir = New-Item -ItemType Directory -Path (Join-Path $distRoot 'wheel') -Force

python -m pip wheel $repo --no-deps --wheel-dir $wheelDir.FullName
if ($LASTEXITCODE -ne 0) { throw 'Wheel build failed' }
$wheels = @(Get-ChildItem -LiteralPath $wheelDir.FullName -Filter 'crossfm-*.whl')
if ($wheels.Count -ne 1) { throw "Expected one wheel, found $($wheels.Count)" }
$wheelTarget = Join-Path $datasetDir.FullName $wheels[0].Name
Copy-Item -LiteralPath $wheels[0].FullName -Destination $wheelTarget

$frozen = $rawConfig.Replace('git_commit: BUILD_TIME', "git_commit: $commit").Replace('dirty: BUILD_TIME', 'dirty: false')
$configTarget = Join-Path $datasetDir.FullName 'frozen_experiment.yaml'
Write-Utf8NoBom $configTarget $frozen
$wheelHash = (Get-FileHash -LiteralPath $wheelTarget -Algorithm SHA256).Hash.ToLowerInvariant()
$configHash = (Get-FileHash -LiteralPath $configTarget -Algorithm SHA256).Hash.ToLowerInvariant()
$bundleSlug = "crossfm-$slugProfile-bundle".ToLowerInvariant()
$runId = "$($profileInfo.protocol)-$slugProfile-$($commit.Substring(0, 8))"
$dependencies = @(
    'numpy==1.26.4', 'pandas==2.2.3', 'python-dotenv==1.0.1',
    'PyYAML==6.0.2', 'scikit-learn==1.6.1',
    'pyarrow==24.0.0', 'openpyxl==3.1.5', 'tabicl==2.2.0',
    'transformers==4.57.6', 'huggingface-hub==0.36.0', 'safetensors==0.7.0'
)
if ($profileInfo.has_beyondarena) { $dependencies += 'data-foundry==0.0.5' }
if ($profileInfo.methods -contains 'catboost') { $dependencies += 'catboost==1.2.10' }
if ($profileInfo.methods -contains 'lightgbm') { $dependencies += 'lightgbm==4.7.0' }
if ($profileInfo.methods -contains 'xgboost') { $dependencies += 'xgboost==3.2.0' }
if (($profileInfo.methods -contains 'cache_tabpfn3') -or ($profileInfo.methods -contains 'tabpfn3_only')) { $dependencies += 'tabpfn==9.0.0' }
if ($profileInfo.methods -contains 'tabm') { $dependencies += 'tabm==0.0.3' }
if ($profileInfo.methods -contains 'autogluon') { $dependencies += 'autogluon.tabular==1.6.3' }
$manifest = [ordered]@{
    schema_version = 'crossfm-fullpaper-bundle-v1'
    run_id = $runId
    profile = $Profile
    git_commit = $commit
    source_dirty = $false
    wheel = [ordered]@{ name = [IO.Path]::GetFileName($wheelTarget); sha256 = $wheelHash; bytes = (Get-Item $wheelTarget).Length }
    config = [ordered]@{ name = 'frozen_experiment.yaml'; sha256 = $configHash; bytes = (Get-Item $configTarget).Length }
    dependencies = $dependencies
}
Write-Utf8NoBom (Join-Path $datasetDir.FullName 'bundle_manifest.json') ($manifest | ConvertTo-Json -Depth 6)
$datasetMetadata = [ordered]@{
    title = "CrossFM $Profile"
    id = "$KaggleOwner/$bundleSlug"
    licenses = @([ordered]@{ name = 'other' })
    isPrivate = $true
}
Write-Utf8NoBom (Join-Path $datasetDir.FullName 'dataset-metadata.json') ($datasetMetadata | ConvertTo-Json -Depth 4)

$driver = Get-Content -LiteralPath (Join-Path $repo 'scripts\kaggle_fullpaper_driver.py') -Raw
$driver = $driver.Replace('__PROFILE__', $Profile).Replace('__BUNDLE_OWNER__', $KaggleOwner).Replace('__BUNDLE_SLUG__', $bundleSlug).Replace('__DATA_OWNER__', $DataOwner).Replace('__DATA_SLUG__', $DataSlug)
Write-Utf8NoBom (Join-Path $kernelDir.FullName 'driver.py') $driver
$datasetSources = @("$KaggleOwner/$bundleSlug")
if ($profileInfo.requires_local_data) { $datasetSources += "$DataOwner/$DataSlug" }
$kernelMetadata = [ordered]@{
    id = "$KaggleOwner/$KernelSlug"
    title = $KernelTitle
    code_file = 'driver.py'
    language = 'python'
    kernel_type = 'script'
    is_private = $true
    enable_gpu = $true
    enable_internet = $true
    machine_shape = 'NvidiaTeslaT4'
    dataset_sources = $datasetSources
    competition_sources = @()
    kernel_sources = @()
    model_sources = @()
}
Write-Utf8NoBom (Join-Path $kernelDir.FullName 'kernel-metadata.json') ($kernelMetadata | ConvertTo-Json -Depth 5)

python (Join-Path $repo 'scripts\validate_fullpaper_bundle.py') $distRoot --profile $Profile
if ($LASTEXITCODE -ne 0) { throw 'Full-paper bundle validation failed' }
Write-Output "Built $Profile at $distRoot"
Write-Output "bundle_dataset=$KaggleOwner/$bundleSlug"
Write-Output "kernel=$KaggleOwner/$KernelSlug"
Write-Output "wheel_sha256=$wheelHash"
Write-Output "config_sha256=$configHash"
