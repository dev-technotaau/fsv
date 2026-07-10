#!/usr/bin/env bash
# One-shot Cloud Run GPU deployment.
#
# Prerequisites
# -------------
# 1. gcloud CLI installed + authed: `gcloud auth login`
# 2. A GCP project with billing enabled. Set GCP_PROJECT env var:
#       export GCP_PROJECT=your-project-id
# 3. APIs enabled on that project (one-time, idempotent):
#       gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
#           artifactregistry.googleapis.com --project $GCP_PROJECT
# 4. Quota for nvidia-l4 GPU in us-central1. New accounts may need to request
#    it via the GCP console: IAM & Admin → Quotas → search "nvidia_l4_gpus".
# 5. The two model files present in THIS directory before running:
#       fence_dinov3_phase1.onnx          (graph, ~13 MB)
#       fence_dinov3_phase1.onnx.data     (weights, ~2.45 GB)
#    They live in ../models/ by default — symlink or copy them in:
#       cp ../models/fence_dinov3_phase1.onnx* .
#
# Usage
# -----
#       bash deploy.sh                       # deploy/update the live service (as a new revision)
#
# SAFE TEST-FIRST (recommended for the /render rollout) — deploy the combined
# image as a SEPARATE new service so the working /detect (fsv-dinov3) is never
# touched. Test BOTH endpoints on it, then point the browser at it and retire the old one:
#       FSV_SERVICE=fsv-dinov3-v2 bash deploy.sh
#
# Nothing is ever deleted: Cloud Run keeps every revision, and a deploy only shifts
# traffic once the new revision is healthy. Roll back by routing traffic to a previous
# revision (see RENDER.md). The old /detect keeps serving throughout the build.
#
# Iteration: re-run after code changes. Cloud Build caches the pip-install
# layer when requirements.txt is unchanged, so subsequent builds finish in
# 2-3 min instead of 8-10.

set -euo pipefail

PROJECT="${GCP_PROJECT:?GCP_PROJECT env var must be set}"
SERVICE="${FSV_SERVICE:-fsv-dinov3-v2}"   # set FSV_SERVICE=fsv-dinov3-v2 to test WITHOUT touching live /detect
REGION="${FSV_REGION:-us-central1}"   # us-central1 is the canonical Cloud Run GPU region
IMAGE="gcr.io/${PROJECT}/${SERVICE}"

# Sanity check: model files present in build context. Without them, the
# Dockerfile COPY fails with a cryptic Cloud Build error.
for f in fence_dinov3_phase1.onnx fence_dinov3_phase1.onnx.data; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: ./$f not found. Copy it from ../models/ first:"
        echo "    cp ../models/fence_dinov3_phase1.onnx* ."
        exit 1
    fi
done

echo "==> Building image (bakes ~40 GB of Qwen weights — first build is SLOW, ~30-60 min)"
gcloud builds submit \
    --tag "${IMAGE}" \
    --project "${PROJECT}" \
    --machine-type e2-highcpu-8 \
    --disk-size 200 \
    --timeout 3600s \
    .

echo ""
echo "==> Deploying to Cloud Run (scale-to-zero, no min-instances)"
gcloud run deploy "${SERVICE}" \
    --image "${IMAGE}" \
    --gpu 1 --gpu-type nvidia-l4 \
    --no-gpu-zonal-redundancy \
    --memory 32Gi --cpu 8 \
    --concurrency 1 \
    --set-env-vars="QWEN_QUANT=nunchaku,FSV_WORKING_RES=1280" \
    --clear-volumes --clear-volume-mounts \
    --min-instances 0 --max-instances 1 \
    --region "${REGION}" \
    --allow-unauthenticated \
    --port 8080 \
    --timeout 3600s \
    --execution-environment gen2 \
    --no-cpu-throttling \
    --project "${PROJECT}"

URL=$(gcloud run services describe "${SERVICE}" \
    --region "${REGION}" \
    --project "${PROJECT}" \
    --format='value(status.url)')

echo ""
echo "==> Deployed. Endpoint URL:"
echo "    ${URL}"
echo ""
echo "==> Update the JS client CONFIG.MODAL_ENDPOINT:"
echo "    MODAL_ENDPOINT: '${URL}/detect',"
echo ""
echo "==> Health check (also warms a fresh instance):"
echo "    curl ${URL}/"
