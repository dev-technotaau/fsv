# One-shot Cloud Run GPU deployment for PowerShell.
#
# Same end result as deploy.sh but for Windows users without bash. Run from
# this directory:
#       .\deploy.ps1
#
# SAFE TEST-FIRST (recommended for the /render rollout): this defaults to a
# SEPARATE service (fsv-dinov3-v2) so the live /detect (fsv-dinov3) is never
# touched. Test both endpoints on it, then set FSV_SERVICE=fsv-dinov3 to cut over.
# Nothing is ever deleted -- Cloud Run keeps every revision (rollback in RENDER.md).
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
$SERVICE = if ($env:FSV_SERVICE) { $env:FSV_SERVICE } else { "fsv-dinov3-v2" }  # v2 = safe test service; set FSV_SERVICE=fsv-dinov3 to hit live
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

Write-Host "==> Building image (bakes ~40 GB of Qwen weights -- first build is SLOW, ~30-60 min)"
gcloud builds submit `
    --tag $IMAGE `
    --project $PROJECT `
    --machine-type e2-highcpu-8 `
    --disk-size 200 `
    --timeout 3600s `
    .
if ($LASTEXITCODE -ne 0) { Write-Error "Cloud Build failed"; exit 1 }

Write-Host ""
Write-Host "==> Deploying to Cloud Run (scale-to-zero, no-zonal-redundancy L4)"
gcloud run deploy $SERVICE `
    --image $IMAGE `
    --gpu 1 --gpu-type nvidia-l4 `
    --no-gpu-zonal-redundancy `
    --memory 32Gi --cpu 8 `
    --concurrency 1 `
    --set-env-vars="QWEN_QUANT=nunchaku,FSV_WORKING_RES=1280" `
    --clear-volumes --clear-volume-mounts `
    --min-instances 0 --max-instances 1 `
    --region $REGION `
    --allow-unauthenticated `
    --port 8080 `
    --timeout 3600s `
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
Write-Host "==> Endpoints:  $URL/detect (seg)   and   $URL/render (renovate)"
Write-Host ""
Write-Host "==> Test /render (first call cold-loads Qwen ~1-2 min, then ~30s):"
Write-Host "    curl -F image=@fence.jpg -F colorHex=#7d4f28 -F family=general $URL/render -o out.jpg -D -"
Write-Host ""
Write-Host "==> Health check (also warms a fresh instance):"
Write-Host "    curl $URL/"
