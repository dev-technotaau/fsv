# One-shot Cloud Run GPU deployment for PowerShell.
#
# Same end result as deploy.sh but for Windows users without bash. Run from
# this directory:
#       .\deploy.ps1
#
# Prerequisites
# -------------
# 1. gcloud CLI installed + authed: `gcloud auth login`
# 2. GCP_PROJECT env var set:
#       $env:GCP_PROJECT = "your-project-id"
# 3. APIs enabled on that project (idempotent - safe to re-run):
#       gcloud services enable run.googleapis.com cloudbuild.googleapis.com `
#           artifactregistry.googleapis.com --project $env:GCP_PROJECT
# 4. Approved nvidia_l4_gpu_allocation_no_zonal_redundancy quota in us-central1
#    (see GCP console -> IAM and Admin -> Quotas).
# 5. Both model files present in THIS directory:
#       fence_dinov3_phase1.onnx          (~14 MB graph)
#       fence_dinov3_phase1.onnx.data     (~2.45 GB weights)
#
# NOTE on file encoding: keep this script ASCII-only. PowerShell on Windows
# defaults to Windows-1252 when reading .ps1 files without a BOM; non-ASCII
# punctuation (em-dash, smart quotes, etc.) becomes mojibake and the parser
# silently breaks several lines downstream of the first bad char.

$ErrorActionPreference = "Stop"

if (-not $env:GCP_PROJECT) {
    Write-Error "GCP_PROJECT env var must be set. Run: `$env:GCP_PROJECT = 'your-project-id'"
    exit 1
}

$PROJECT = $env:GCP_PROJECT
$SERVICE = if ($env:FSV_SERVICE) { $env:FSV_SERVICE } else { "fsv-dinov3" }
$REGION  = if ($env:FSV_REGION)  { $env:FSV_REGION }  else { "us-central1" }
$IMAGE   = "gcr.io/$PROJECT/$SERVICE"

# Sanity check: model files present in build context. Without them, the
# Dockerfile COPY fails with a cryptic Cloud Build error.
foreach ($f in @("fence_dinov3_phase1.onnx", "fence_dinov3_phase1.onnx.data")) {
    if (-not (Test-Path $f)) {
        Write-Error "ERROR: .\$f not found. Copy it from ..\models\ first:`n    Copy-Item ..\models\fence_dinov3_phase1.onnx* ."
        exit 1
    }
}

Write-Host "==> Building image (uploads ~2.5 GB build context -- first build is slow)"
gcloud builds submit `
    --tag $IMAGE `
    --project $PROJECT `
    --timeout 30m `
    .
if ($LASTEXITCODE -ne 0) { Write-Error "Cloud Build failed"; exit 1 }

Write-Host ""
Write-Host "==> Deploying to Cloud Run (scale-to-zero, no-zonal-redundancy L4)"
gcloud run deploy $SERVICE `
    --image $IMAGE `
    --gpu 1 --gpu-type nvidia-l4 `
    --no-gpu-zonal-redundancy `
    --memory 16Gi --cpu 4 `
    --concurrency 4 `
    --min-instances 0 --max-instances 1 `
    --region $REGION `
    --allow-unauthenticated `
    --port 8080 `
    --timeout 300s `
    --execution-environment gen2 `
    --no-cpu-throttling `
    --project $PROJECT
if ($LASTEXITCODE -ne 0) { Write-Error "Cloud Run deploy failed"; exit 1 }

$URL = gcloud run services describe $SERVICE `
    --region $REGION `
    --project $PROJECT `
    --format='value(status.url)'

Write-Host ""
Write-Host "==> Deployed. Endpoint URL:"
Write-Host "    $URL"
Write-Host ""
Write-Host "==> Update the JS client CONFIG.MODAL_ENDPOINT:"
Write-Host "    MODAL_ENDPOINT: '$URL/detect',"
Write-Host ""
Write-Host "==> Health check (also warms a fresh instance):"
Write-Host "    curl $URL/"
