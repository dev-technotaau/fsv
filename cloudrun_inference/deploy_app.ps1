# FAST app-only redeploy. Rebuilds ON TOP of the last baked image, swapping only the .py files, so
# the build context is ~30KB (no 2.3GB ONNX re-upload -> no slow-upload timeout). The 25GB base is
# pulled server-side in Google's network. Use this for app.py / qwen_engine.py / color_finish.py
# tweaks. Use the full deploy.ps1 ONLY when requirements.txt, the Qwen bake, or the ONNX change.
#
#   $env:GCP_PROJECT = "your-project"; .\deploy_app.ps1

$ErrorActionPreference = "Stop"
if (-not $env:GCP_PROJECT) { Write-Error "Set `$env:GCP_PROJECT first"; exit 1 }
$PROJECT = $env:GCP_PROJECT
$SERVICE = if ($env:FSV_SERVICE) { $env:FSV_SERVICE } else { "fsv-dinov3-v2" }
$REGION  = if ($env:FSV_REGION)  { $env:FSV_REGION }  else { "us-central1" }
$IMAGE   = "gcr.io/$PROJECT/$SERVICE"

# Tiny build dir: only the .py files + a FROM-latest Dockerfile.
$tmp = Join-Path $env:TEMP ("fsv_app_" + [System.Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $tmp | Out-Null
Copy-Item color_finish.py, qwen_engine.py, app.py $tmp
Set-Content -Path (Join-Path $tmp "Dockerfile") -Encoding ascii -Value @"
FROM ${IMAGE}:latest
WORKDIR /app
COPY color_finish.py qwen_engine.py app.py ./
"@

Write-Host "==> Fast app-only build (context ~30KB; base pulled server-side)"
gcloud builds submit $tmp --tag $IMAGE --project $PROJECT --timeout 1200s
$ok = $?
Remove-Item -Recurse -Force $tmp
if (-not $ok) { Write-Error "Build failed"; exit 1 }

Write-Host "==> Deploying $SERVICE"
gcloud run deploy $SERVICE `
    --image $IMAGE `
    --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy `
    --memory 32Gi --cpu 8 --concurrency 1 `
    --set-env-vars="QWEN_QUANT=nunchaku,FSV_WORKING_RES=1280" `
    --clear-volumes --clear-volume-mounts `
    --min-instances 0 --max-instances 1 `
    --region $REGION --allow-unauthenticated --port 8080 `
    --timeout 3600s --execution-environment gen2 --no-cpu-throttling `
    --project $PROJECT
if ($LASTEXITCODE -ne 0) { Write-Error "Cloud Run deploy failed"; exit 1 }

$URL = gcloud run services describe $SERVICE --region $REGION --project $PROJECT --format='value(status.url)'
Write-Host ""
Write-Host "==> Deployed: $URL"
Write-Host "==> Test: curl.exe -F image=@fence.jpg -F colorHex=#7d4f28 -F family=general $URL/render -o out.jpg -D -"
