function initFenceSimulator(rootElement) {
  // Proxy shim so existing `document.getElementById / querySelector / querySelectorAll`
  // calls route into the shadow root, while everything else (createElement,
  // addEventListener, body, fonts, etc.) falls through to the real document.
  const document = new Proxy(window.document, {
    get(target, prop) {
      if (prop === "getElementById")
        return (id) => rootElement.getElementById(id);
      if (prop === "querySelector")
        return (sel) => rootElement.querySelector(sel);
      if (prop === "querySelectorAll")
        return (sel) => rootElement.querySelectorAll(sel);
      if (prop === "activeElement") return rootElement.activeElement;
      const value = target[prop];
      return typeof value === "function" ? value.bind(target) : value;
    },
  });

  const CONFIG = {
    MODAL_ENDPOINT:
      "https://fsv-dinov3-467125191853.us-central1.run.app/detect",
  UPLOAD_MAX_DIM: 1024,
  UPLOAD_JPEG_QUALITY: 0.85,

  // Cap the working resolution for all client-side mask post-processing.
  // The model itself runs at 512px so there's no detection signal beyond
  // ~1536px. Without this cap, on a 5000x4000 source image the post-process
  // pipeline (bilinear upsample + dilate-radius scans + per-pixel Lab
  // conversions + connected-components labelling) takes 25-35s in
  // single-threaded JS. At 1536 longest-side it's ~3s with no quality loss:
  // the final mask is bilinear-upsampled back to source dims before
  // recolor/clean composite so output still matches the original image.
  POSTPROCESS_MAX_DIM: 1536,

  // Cap the working resolution for the cleanFence pipeline. The 6 guided
  // filter calls + dual sRGB<->Lab per-pixel loop dominate cost; at 5000x4000
  // the full-res path takes ~30-45s in single-threaded JS. At 1536 longest-
  // side it's ~3-5s. The cleaned RGB is bilinear-upsampled back to source
  // dims and composited onto the original photo (mask-gated) so non-fence
  // pixels retain full resolution. Cleaning is a homogenizing operation by
  // design (smooths color variation), so the slight detail loss inside the
  // cleaned fence region is consistent with the algorithm's intent.
  CLEAN_MAX_DIM: 1536,

  INPUT_SIZE: 512,
  DEFAULT_THRESHOLD: 0.5,

  CC_MIN_BLOB_AREA_PCT: 0.05,
  CC_KEEP_TOP_K_BLOBS: 0,

  POST_ERODE_PX: 0,

  USE_SOFT_MASK: true,

  SOFT_MASK_LOW: 0.5,
  SOFT_MASK_HIGH: 0.85,

  FILTER_VEGETATION: true,

  VEGETATION_GREEN_DOMINANCE: 25,

  FILTER_BARK: true,
  BARK_SAT_GAP: 0.1,
  BARK_BRIGHT_DELTA: 40,

  FILTER_TRUNKS: true,
  TRUNK_COLOR_DIST_HARD: 90,
  TRUNK_COLOR_DIST_SOFT: 55,
  TRUNK_DESAT_DELTA: 0.1,

  FILTER_BUILDINGS: true,
  BUILDING_MIN_MEAN_CONF: 0.05,

  BUILDING_CONF_RATIO: 0.45,
  BUILDING_MIN_CC_PX: 500,

  BUILDING_BLOCKY_FILL_RATIO: 0.72,
  BUILDING_BLOCKY_CONF_BOOST: 0.12,

  FILTER_SKY: true,
  SKY_TOP_FRACTION: 0.4,
  SKY_MIN_LUMINANCE: 175,
  SKY_MAX_SATURATION: 0.1,

  FILL_HOLES: true,
  HOLE_FILL_MAX_PCT: 0.3,
  HOLE_FILL_VALUE_SCALE: 0.95,

  RECOLOR_BINARIZE_MASK: true,
  RECOLOR_FULL_ALPHA_THRESHOLD: 0.15,

  FILTER_SPATIAL_RECOVERY: true,
  RECOVERY_CORE_THRESHOLD: 0.85,
  RECOVERY_FILL_THRESHOLD: 0.4,
  RECOVERY_DILATE_PX: 20,

  BOTTOM_EXTEND_MAX_PX: 5,
  BOTTOM_EXTEND_CHROMA_MAX: 10,

  FILTER_CC_COLOR_OUTLIERS: true,
  CC_OUTLIER_K_STDDEV: 2.8,
  CC_OUTLIER_MIN_DIST: 80,
  CC_OUTLIER_MIN_PX: 300,

  FILTER_ORIENTATION: false,
  ORIENTATION_RATIO: 1.6,
  ORIENTATION_MIN_COUNT: 500,

  FILTER_CC_PRINCIPAL_AXIS: false,
  CC_AXIS_ANGLE_TOLERANCE_DEG: 30,
  CC_AXIS_MIN_PX: 500,
  CC_AXIS_MIN_ASPECT_RATIO: 2.0,

  FILTER_JUNK_BLOBS: true,
  JUNK_BLOB_MAX_AREA_PCT: 0.6,
  JUNK_BLOB_MIN_ASPECT: 1.8,

  FILTER_BUILDING_WALLS: true,
  BUILDING_WALL_SAT_GAP: 0.2,
  BUILDING_WALL_MIN_CC_PX: 400,
  BUILDING_WALL_MIN_LARGEST_SAT: 0.25,

  BUILDING_WALL_MAX_SAT_STDDEV: 0.12,
};

let modalReady = false;
let originalImage = null;
let maskData = null;

let cleanedImageData = null;

let resultState = "original";

const fileInput = document.getElementById("file-input");
const uploadSection = document.getElementById("upload-section");
const originalCanvas = document.getElementById("original-canvas");
const maskCanvas = document.getElementById("mask-canvas");
const resultCanvas = document.getElementById("result-canvas");
const colorPickerButton = document.getElementById("color-picker-button");
const colorPickerModal = document.getElementById("color-picker-modal");
const colorPickerClose = document.getElementById("color-picker-close");
const colorPickerGrid = document.getElementById("color-picker-grid");
const currentColorDisplay = document.getElementById("current-color-display");
const colorNameDisplay = document.getElementById("color-name-display");
const blendMode = document.getElementById("blend-mode");
const opacity = document.getElementById("opacity");
const threshold = document.getElementById("threshold");
const edgeSmoothing = document.getElementById("edge-smoothing");
const status = document.getElementById("status");
const loader = document.getElementById("loader");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingText = document.getElementById("loading-text");

const cleanBtn = document.getElementById("clean-btn");
const detectBtn = document.getElementById("detect-btn");
const recolorBtn = document.getElementById("recolor-btn");
const downloadBtn = document.getElementById("download-btn");
const downloadBtnMobile = document.getElementById("download-btn-mobile");
const resetBtn = document.getElementById("reset-btn");

function setDownloadEnabled(enabled) {
  downloadBtn.disabled = !enabled;
  if (downloadBtnMobile) downloadBtnMobile.disabled = !enabled;
}

const stainColorGroups = [
  {
    category: "General",
    colors: [
      { name: "Natural Cedar", color: "#A37033" },
      { name: "Oxford Brown", color: "#4B4036" },
      { name: "Redwood", color: "#9D4A22" },
      { name: "Leatherwood", color: "#8B572A" },
      { name: "Cedar Tone", color: "#A35E29" },
    ],
  },
  {
    category: "Semi-transparent",
    colors: [
      { name: "Chestnut", color: "#56402E" },
      { name: "Mahogany", color: "#6F2B23" },
      { name: "Pecan", color: "#A0784E" },
      { name: "Sequoia", color: "#813F2D" },
      { name: "Walnut", color: "#5D4037" },
    ],
  },
  {
    category: "Semi-solid",
    colors: [
      { name: "Auburn", color: "#7A3326" },
      { name: "Barnwood", color: "#7A6E62" },
      { name: "Black", color: "#1A1A1A" },
      { name: "Cape Cod Gray", color: "#888B85" },
      { name: "Chocolate", color: "#3F2A1F" },
      { name: "Eucalyptus", color: "#6E7B6F" },
      { name: "Palomino", color: "#C4986E" },
      { name: "Sable", color: "#3B2E27" },
      { name: "Slate Gray", color: "#5C636D" },
    ],
  },
];

let selectedColor = "#A37033";
let selectedColorName = "Natural Cedar";

const canvasLoading = document.getElementById("canvas-loading");
const canvasLoadingText = document.getElementById("canvas-loading-text");

let _canvasLoadingHideTimer = null;

function showCanvasLoading(message) {
  if (!canvasLoading) return;
  /* If a hide-timer is pending, the overlay is still visible but
   * on its way out — the text on it is stale from the just-finished
   * operation (e.g. "Recoloring fence..." after Apply Stain). Same
   * goes if the overlay is fully hidden. In both cases we want to
   * skip the 160 ms swap animation in setCanvasLoadingText so the
   * user sees the NEW label immediately instead of the previous one
   * fading out. The swap is only useful for transitions within an
   * in-progress operation (e.g. "Uploading..." → "Detecting..."
   * within a single Apply Stain). */
  const wasAboutToHide = !!_canvasLoadingHideTimer;
  if (_canvasLoadingHideTimer) {
    clearTimeout(_canvasLoadingHideTimer);
    _canvasLoadingHideTimer = null;
  }
  if (canvasLoading.hidden || wasAboutToHide) {
    if (canvasLoadingText) {
      canvasLoadingText.classList.remove("swap");
      canvasLoadingText.textContent = message;
    }
  } else {
    setCanvasLoadingText(message);
  }
  if (canvasLoading.hidden) canvasLoading.hidden = false;
  canvasLoading.setAttribute("aria-busy", "true");

  if (compareTip && !compareTip.hidden) dismissCoachTip();
}

function hideCanvasLoading() {
  if (!canvasLoading) return;
  if (_canvasLoadingHideTimer) clearTimeout(_canvasLoadingHideTimer);
  _canvasLoadingHideTimer = setTimeout(() => {
    canvasLoading.hidden = true;
    canvasLoading.setAttribute("aria-busy", "false");
    _canvasLoadingHideTimer = null;
  }, 220);
}

function setCanvasLoadingText(message) {
  if (!canvasLoadingText) return;
  if (canvasLoadingText.textContent === message) return;
  canvasLoadingText.classList.add("swap");
  setTimeout(() => {
    canvasLoadingText.textContent = message;
    canvasLoadingText.classList.remove("swap");
  }, 160);
}

function showLoading(message) {
  if (originalImage && canvasLoading) {
    showCanvasLoading(message);
  } else {
    loadingText.textContent = message;
    loadingOverlay.classList.add("active");
  }
}

function hideLoading() {
  hideCanvasLoading();
  loadingOverlay.classList.remove("active");
}

async function selectStain(item) {
  if (selectedColor === item.color) return;

  selectedColor = item.color;
  selectedColorName = item.name;

  document.querySelectorAll("#color-bar .color-bar-item").forEach((btn) => {
    const on = btn.dataset.color === item.color;
    btn.classList.toggle("is-active", on);
    btn.setAttribute("aria-checked", on ? "true" : "false");

    if (!btn.disabled) btn.tabIndex = on ? 0 : -1;
  });

  if (currentColorDisplay)
    currentColorDisplay.style.backgroundColor = item.color;
  if (colorNameDisplay) colorNameDisplay.textContent = item.name;

  if (resultState === "stained" || resultState === "cleaned_stained") {
    await recolorFence();
  }
}

function setColorChipsEnabled(enabled) {
  document.querySelectorAll("#color-bar .color-bar-item").forEach((btn) => {
    btn.disabled = !enabled;
    btn.setAttribute("aria-disabled", enabled ? "false" : "true");
    btn.tabIndex = enabled
      ? btn.classList.contains("is-active")
        ? 0
        : -1
      : -1;
  });
}

const workspaceEl = document.querySelector(".workspace");
const uploadCardSection = document.querySelector(
  'section.card[aria-labelledby="upload-card-title"]',
);
const canvasCardSection = document.querySelector(
  'section.card[aria-labelledby="canvas-card-title"]',
);
const actionBar1Div = document.getElementById("action-bar-1");
const actionBar2Div = document.getElementById("action-bar-2");
const colorBarWrapDiv = document.querySelector(".color-bar-wrap");
const downloadMobileWrapDiv = document.getElementById("download-mobile-wrap");

function _setVisible(el, visible) {
  if (!el) return;
  if (visible) {
    el.hidden = false;

    el.style.display = "";
  } else {
    el.hidden = true;
    el.style.display = "none";
  }
}

function showLoadedState() {
  if (workspaceEl) workspaceEl.classList.add("is-loaded");
  _setVisible(uploadCardSection, false);
  _setVisible(canvasCardSection, true);
  _setVisible(actionBar1Div, true);
  _setVisible(colorBarWrapDiv, true);
  _setVisible(actionBar2Div, true);
  _setVisible(downloadMobileWrapDiv, true);

  if (typeof equalizeColumns === "function") {
    requestAnimationFrame(equalizeColumns);
  }
}

function showEmptyState() {
  if (workspaceEl) workspaceEl.classList.remove("is-loaded");
  _setVisible(uploadCardSection, true);
  _setVisible(canvasCardSection, false);
  _setVisible(actionBar1Div, false);
  _setVisible(colorBarWrapDiv, false);
  _setVisible(actionBar2Div, false);
  _setVisible(downloadMobileWrapDiv, false);
}

function initColorPicker() {
  const colorBar = document.getElementById("color-bar");
  if (!colorBar) return;
  colorBar.innerHTML = "";

  const totalChips = stainColorGroups.reduce(
    (n, g) => n + g.colors.length,
    0,
  );
  const selectedInExtras = stainColorGroups
    .slice(1)
    .some((g) => g.colors.some((c) => c.color === selectedColor));

  const buildSection = (group) => {
    const section = document.createElement("div");
    section.className = "color-bar-category";

    const heading = document.createElement("div");
    heading.className = "color-bar-category-label";
    heading.textContent = group.category;
    section.appendChild(heading);

    const chips = document.createElement("div");
    chips.className = "color-bar-chips";

    group.colors.forEach((item) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "color-bar-item";
      btn.setAttribute("role", "radio");
      btn.setAttribute(
        "aria-checked",
        item.color === selectedColor ? "true" : "false",
      );
      btn.setAttribute("aria-label", item.name + " stain");
      btn.title = item.name;
      btn.dataset.color = item.color;
      btn.dataset.name = item.name;

      btn.disabled = true;
      btn.setAttribute("aria-disabled", "true");
      btn.tabIndex = -1;
      if (item.color === selectedColor) btn.classList.add("is-active");

      const swatch = document.createElement("span");
      swatch.className = "color-bar-swatch";
      swatch.style.backgroundColor = item.color;
      swatch.setAttribute("aria-hidden", "true");

      const label = document.createElement("span");
      label.className = "color-bar-label";
      label.textContent = item.name;

      btn.appendChild(swatch);
      btn.appendChild(label);
      btn.addEventListener("click", () => selectStain(item));

      chips.appendChild(btn);
    });

    section.appendChild(chips);
    return section;
  };

  colorBar.appendChild(buildSection(stainColorGroups[0]));

  const extras = document.createElement("div");
  extras.className = "color-bar-extras";
  extras.id = "color-bar-extras";
  stainColorGroups
    .slice(1)
    .forEach((group) => extras.appendChild(buildSection(group)));
  colorBar.appendChild(extras);

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "color-bar-toggle";
  toggle.setAttribute("aria-controls", "color-bar-extras");
  const extrasFlat = stainColorGroups.slice(1).flatMap((g) => g.colors);
  const PEEK_COUNT = 5;
  const peekColors =
    extrasFlat.length <= PEEK_COUNT
      ? extrasFlat
      : Array.from({ length: PEEK_COUNT }, (_, i) =>
          extrasFlat[Math.floor((i * extrasFlat.length) / PEEK_COUNT)],
        );
  const peekHTML = peekColors
    .map(
      (c) =>
        `<span class="color-bar-toggle-dot" style="background:${c.color}"></span>`,
    )
    .join("");
  const collapsedLabel = `Browse all ${totalChips} finishes`;
  const expandedLabel = "Show fewer";
  const setToggleState = (open) => {
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    toggle.innerHTML = open
      ? `<span class="color-bar-toggle-text">${expandedLabel}</span>` +
        `<i class="bi bi-chevron-up" aria-hidden="true"></i>`
      : `<span class="color-bar-toggle-peek" aria-hidden="true">${peekHTML}</span>` +
        `<span class="color-bar-toggle-text">${collapsedLabel}</span>` +
        `<i class="bi bi-chevron-down" aria-hidden="true"></i>`;
  };
  setToggleState(selectedInExtras);
  if (selectedInExtras) colorBar.classList.add("is-expanded");
  toggle.addEventListener("click", () => {
    const open = colorBar.classList.toggle("is-expanded");
    setToggleState(open);
  });
  colorBar.appendChild(toggle);

  if (!colorBar.dataset.kbBound) {
    colorBar.addEventListener("keydown", (e) => {
      const items = Array.from(
        colorBar.querySelectorAll(".color-bar-item"),
      ).filter((el) => el.offsetParent !== null);
      const cur = items.indexOf(document.activeElement);
      if (cur < 0) return;
      let next = cur;
      if (e.key === "ArrowRight" || e.key === "ArrowDown")
        next = (cur + 1) % items.length;
      else if (e.key === "ArrowLeft" || e.key === "ArrowUp")
        next = (cur - 1 + items.length) % items.length;
      else if (e.key === "Home") next = 0;
      else if (e.key === "End") next = items.length - 1;
      else return;
      e.preventDefault();
      items[next].focus();
      items[next].click();
    });
    colorBar.dataset.kbBound = "1";
  }
}

colorPickerButton.addEventListener("click", () => {
  colorPickerModal.classList.add("active");
});

colorPickerClose.addEventListener("click", () => {
  colorPickerModal.classList.remove("active");
});

colorPickerModal.addEventListener("click", (e) => {
  if (e.target === colorPickerModal) {
    colorPickerModal.classList.remove("active");
  }
});

const helpModal = document.getElementById("help-modal");
const helpModalClose = document.getElementById("help-modal-close");
const openHelpModal = () => helpModal && helpModal.classList.add("active");
const closeHelpModal = () => helpModal && helpModal.classList.remove("active");
if (helpModalClose) helpModalClose.addEventListener("click", closeHelpModal);

if (helpModal) {
  helpModal.addEventListener("click", (e) => {
    if (e.target === helpModal) closeHelpModal();
  });
}

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (helpModal && helpModal.classList.contains("active")) closeHelpModal();
  if (colorPickerModal && colorPickerModal.classList.contains("active")) {
    colorPickerModal.classList.remove("active");
  }
});

const uploadCardEl = document.querySelector(
  'section.card[aria-labelledby="upload-card-title"]',
);
const canvasCardEl = document.querySelector(
  'section.card[aria-labelledby="canvas-card-title"]',
);
const canvasStackEl = document.getElementById("canvas-stack");
const canvasBodyEl = canvasCardEl
  ? canvasCardEl.querySelector(".card-body")
  : null;
const canvasHeadEl = canvasCardEl
  ? canvasCardEl.querySelector(".card-head")
  : null;

function matchCanvasToUploadHeight() {
  if (!uploadCardEl || !canvasStackEl) return;

  if (uploadCardEl.offsetParent === null) {
    canvasStackEl.style.height = "";
    canvasStackEl.style.minHeight = "";
    return;
  }

  const target = uploadCardEl.getBoundingClientRect().height;
  const headH = canvasHeadEl ? canvasHeadEl.getBoundingClientRect().height : 0;
  const bodyCS = canvasBodyEl ? getComputedStyle(canvasBodyEl) : null;
  const bodyPad = bodyCS
    ? (parseFloat(bodyCS.paddingTop) || 0) +
      (parseFloat(bodyCS.paddingBottom) || 0)
    : 0;

  const stackH = Math.max(120, target - headH - bodyPad - 3);
  if (window.matchMedia("(max-width: 900px)").matches) {
    canvasStackEl.style.height = "";
    canvasStackEl.style.minHeight = stackH + "px";
  } else {
    canvasStackEl.style.minHeight = "";
    canvasStackEl.style.height = stackH + "px";
  }
}

const actionToolbarEl = document.getElementById("action-bar-1");
const colorCardEl = document.querySelector(".color-bar-card");

function matchColorBarToToolbarHeight() {
  if (!actionToolbarEl || !colorCardEl) return;

  if (actionToolbarEl.offsetParent === null) {
    colorCardEl.style.minHeight = "";
    return;
  }
  if (window.matchMedia("(max-width: 900px)").matches) {
    colorCardEl.style.minHeight = "";
    return;
  }
  const target = actionToolbarEl.getBoundingClientRect().height;
  colorCardEl.style.minHeight = Math.max(60, target) + "px";
}

function equalizeColumns() {
  matchCanvasToUploadHeight();
  matchColorBarToToolbarHeight();
}

window.addEventListener("resize", equalizeColumns);
window.addEventListener("load", equalizeColumns);
if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(equalizeColumns);
}
if (window.ResizeObserver) {
  const ro = new ResizeObserver(equalizeColumns);
  if (uploadCardEl) ro.observe(uploadCardEl);
  if (actionToolbarEl) ro.observe(actionToolbarEl);
}

requestAnimationFrame(() => requestAnimationFrame(equalizeColumns));

async function init() {
  // Kick off WASM module load early (concurrent with the Modal health
  // check). By the time the user clicks Apply Stain / Clean Fence, the
  // module is typically ready and the hot path uses it. If it's not
  // ready in time, the first call uses JS fallback and subsequent calls
  // pick up WASM.
  if (typeof _wasmInit === "function") {
    _wasmInit().catch(() => {});
  }

  /* Preload the SigLIP fence-gate model on page idle. ~95 MB one-time
   * download, cached in IndexedDB forever after. Using
   * requestIdleCallback (with setTimeout fallback) yields to the
   * browser's critical-path work first so the initial paint /
   * interactive aren't slowed. By the time the user uploads a photo,
   * the model is typically already cached and ready, making the gate
   * ~200 ms instead of ~10-20 s on cold network. _initFenceGate is
   * idempotent, so this can run alongside the per-upload kickoff in
   * handleImageFile with no double-work. */
  if (typeof _initFenceGate === "function") {
    const startGatePreload = () => _initFenceGate().catch(() => {});
    if (typeof requestIdleCallback === "function") {
      requestIdleCallback(startGatePreload, { timeout: 3000 });
    } else {
      setTimeout(startGatePreload, 1500);
    }
  }

  if (
    !CONFIG.MODAL_ENDPOINT ||
    CONFIG.MODAL_ENDPOINT.includes("YOUR-WORKSPACE")
  ) {
    modalReady = false;
    toast(
      "Simulator not configured. " +
        "Set CONFIG.MODAL_ENDPOINT in index3.html.",
      "error",
      8000,
    );
    return;
  }

  showLoading("Loading simulator...");

  updateStatus("Loading simulator...", "loading", { silent: true });

  const healthUrl = CONFIG.MODAL_ENDPOINT.replace(/\/detect\/?$/, "/");

  try {
    const t0 = performance.now();
    const res = await fetch(healthUrl, { method: "GET", cache: "no-store" });
    const dt = Math.round(performance.now() - t0);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const info = await res.json().catch(() => ({}));
    modalReady = true;

    updateStatus("Simulator ready", "success");
  } catch (error) {
    modalReady = false;
    console.error("Modal health check failed:", error);
    toast(
      "Simulator is starting up — your first preview may take a moment.",
      "error",
      6500,
    );
  } finally {
    hideLoading();
  }
}

// ============================================================================
// FENCE PRE-FILTER — client-side zero-shot OWLv2 object detector
// ============================================================================
// Decides whether an uploaded photo plausibly contains a wooden/cedar fence
// BEFORE the server-side detection pipeline runs. Blocks obvious non-fence
// uploads (selfies, food, interiors, vehicles, screenshots, etc.) AND the
// harder "wood-textured-but-not-a-fence" cases (coffee cup on wooden table,
// wooden stairs, wooden deck) that earlier classification-based gates could
// not distinguish.
//
// Why detection (OWLv2) instead of classification (SigLIP/CLIP):
//   Zero-shot CLASSIFICATION scores the whole image against text prompts.
//   It confuses "small fence in busy scene" with "wood texture close-up"
//   because the WHOLE-IMAGE wood-themed signal is similar.
//   Zero-shot DETECTION (OWLv2) requires the model to LOCALIZE a fence,
//   returning bounding boxes with scores. A coffee cup has no fence to
//   find; a small fence behind pool umbrellas DOES.
//
// Model: Xenova/owlv2-base-patch16-ensemble (~150 MB quantized).
// Cached in IndexedDB by transformers.js, instant on subsequent visits.
//
// See index4_dinov3.html for the full design discussion.

const _FENCE_GATE = {
  status: "uninit", // "uninit" | "loading" | "ready" | "unavailable"
  worker: null,
  workerUrl: null,
  deviceUsed: null,
  dtypeUsed: null,
  loadPromise: null,
  initStartMs: 0,
  progressCb: null,
  lastResult: null,
  /* Per-request handler dispatch for messages from the worker.
   * Each postMessage carries an id; the matching entry in
   * _pending holds the callbacks (onProgress / onReady /
   * onResult / onError). */
  _msgId: 0,
  _pending: new Map(),
};

/* Worker entry point. Defined here as a named function so we can
 * .toString() it into a Blob URL Worker. The function body runs
 * INSIDE the worker — it has its own `self`, can call `import()`,
 * owns a separate WebGPU adapter, etc. The main thread sees this
 * function as inert data. */
function _fenceGateWorkerFn() {
  let detector = null;
  self.onmessage = async (e) => {
    const msg = e.data;
    const id = msg.id;
    const reply = (type, payload) =>
      self.postMessage({ id, type, ...(payload || {}) });
    try {
      if (msg.type === "init") {
        const tx = await import(
          "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2"
        );
        const { pipeline, env } = tx;
        env.allowLocalModels = false;
        env.useBrowserCache = true;
        if (env.backends && env.backends.onnx && env.backends.onnx.wasm) {
          env.backends.onnx.wasm.numThreads = 1;
        }

        const onProgress = (p) => reply("progress", { data: p });
        const isNet = (err) => {
          const m = (err && (err.message || String(err))) || "";
          return /fetch|network|connection|ECONNRESET|aborted|timeout/i.test(
            m,
          );
        };

        /* OWL-ViT v1 base patch32 instead of OWLv2 base ensemble.
         * Tradeoffs:
         *   - ~80 MB vs ~150 MB quantized (faster first-time
         *     download)
         *   - 32×32 patches at 768×768 = 576 patch tokens vs OWLv2's
         *     16×16 = 2304 → ~4x less attention work per image
         *   - ~2.5 s vs ~7-9 s WebGPU inference, with
         *     correspondingly less GPU contention so the UI no
         *     longer stutters during gate
         *   - Slightly lower accuracy on complex scenes than OWLv2,
         *     but the binary fence/non-fence task doesn't need the
         *     extra precision. */
        const MODEL_ID = "Xenova/owlvit-base-patch32";
        let deviceUsed = "webgpu",
          dtypeUsed = "fp16",
          webgpuOk = false;
        for (let attempt = 1; attempt <= 3; attempt++) {
          try {
            detector = await pipeline(
              "zero-shot-object-detection",
              MODEL_ID,
              {
                device: "webgpu",
                dtype: "fp16",
                progress_callback: onProgress,
              },
            );
            webgpuOk = true;
            break;
          } catch (err) {
            const em = (err && (err.message || String(err))) || "";
            if (!isNet(err)) {
              reply("log", {
                level: "warn",
                message: "WebGPU unavailable (not retrying): " + em,
              });
              break;
            }
            if (attempt < 3) {
              const waitMs = attempt * 2000;
              reply("log", {
                level: "warn",
                message:
                  "WebGPU download attempt " +
                  attempt +
                  "/3 failed (" +
                  em +
                  "), retrying in " +
                  waitMs +
                  " ms...",
              });
              await new Promise((r) => setTimeout(r, waitMs));
            } else {
              reply("log", {
                level: "warn",
                message:
                  "WebGPU download failed after 3 attempts, falling back to WASM: " +
                  em,
              });
            }
          }
        }
        if (!webgpuOk) {
          deviceUsed = "wasm";
          dtypeUsed = "q8";
          detector = await pipeline(
            "zero-shot-object-detection",
            MODEL_ID,
            { device: "wasm", dtype: "q8", progress_callback: onProgress },
          );
        }
        reply("ready", { device: deviceUsed, dtype: dtypeUsed });
      } else if (msg.type === "infer") {
        if (!detector) {
          reply("error", { error: "detector not initialized" });
          return;
        }
        const { bitmap, queries, options } = msg.data;
        let url = null;
        try {
          /* transformers.js v3 RawImage.read() only reliably accepts
           * URL strings — not ImageBitmap, not Blob, not
           * OffscreenCanvas. So: draw bitmap to OffscreenCanvas
           * (worker-safe), convert to JPEG Blob (off-thread), wrap
           * in an object URL, pass that. All of these APIs work
           * inside a worker; no DOM access required. */
          const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
          canvas.getContext("2d").drawImage(bitmap, 0, 0);
          const blob = await canvas.convertToBlob({
            type: "image/jpeg",
            quality: 0.9,
          });
          url = URL.createObjectURL(blob);
          const results = await detector(url, queries, options);
          reply("result", { data: results });
        } finally {
          try {
            bitmap && bitmap.close && bitmap.close();
          } catch {
            /* already closed */
          }
          if (url)
            try {
              URL.revokeObjectURL(url);
            } catch {
              /* ignore */
            }
        }
      }
    } catch (err) {
      reply("error", {
        error: (err && (err.message || String(err))) || "unknown error",
      });
    }
  };
}

/* Two query groups. We feed BOTH to OWL-ViT in one inference call
 * (each query is a cheap text-encode), then in the decision rule
 * we compare max(fence_scores) vs max(distractor_scores). The
 * image must score the fence concept HIGHER than the strongest
 * distractor for the gate to pass.
 *
 * This is what the score threshold alone couldn't do — wooden
 * stairs score 0.065 on "wooden fence" but ALSO score highly on
 * "wooden stairs", and we let that comparison make the call. */
const _FENCE_QUERIES = [
  "wooden fence",
  "wood fence",
  "cedar fence",
  "wooden privacy fence",
  "wooden picket fence",
  "wooden plank fence",
  "stained wooden fence",
  "weathered wooden fence",
];

const _FENCE_DISTRACTOR_QUERIES = [
  "wooden stairs",
  "wooden staircase",
  "wooden railing",
  "wooden banister",
  "wooden deck",
  "wooden balcony",
  "wooden floor",
  "wooden table",
  "wooden chair",
  /* Note: "tree trunk" was removed — it matches ~99% area on any
   * outdoor scene with branches/trees in foreground, which is
   * common in real fence photos (gardens, trees near fences). It
   * dominated the distractor bucket and blocked real fences.
   * Stairs / railing / deck / banister already cover the
   * stairs-and-railing false positive. */
];

async function _initFenceGate(opts = {}) {
  if (_FENCE_GATE.status === "ready") return true;
  if (_FENCE_GATE.status === "unavailable") return false;
  if (_FENCE_GATE.status === "loading") return _FENCE_GATE.loadPromise;

  _FENCE_GATE.status = "loading";
  _FENCE_GATE.initStartMs = performance.now();
  _FENCE_GATE.progressCb = opts.onProgress || null;

  _FENCE_GATE.loadPromise = new Promise((resolve) => {
    try {
      /* Spin up a Web Worker that hosts the entire OWLv2 pipeline.
       * The model download, WebGPU init, and per-image inference
       * all happen off the main thread — UI stays smooth during
       * the ~2-10 s inference instead of freezing.
       *
       * Worker source is the stringified function above, wrapped
       * in an IIFE and served from a Blob URL. Same-origin, no
       * extra file to ship. Module worker (type: "module") so
       * dynamic import works for the transformers.js CDN load. */
      const workerSrc = "(" + _fenceGateWorkerFn.toString() + ")();";
      const workerBlob = new Blob([workerSrc], {
        type: "application/javascript",
      });
      _FENCE_GATE.workerUrl = URL.createObjectURL(workerBlob);
      const worker = new Worker(_FENCE_GATE.workerUrl, { type: "module" });
      _FENCE_GATE.worker = worker;

      worker.onerror = (e) => {
        console.warn("[fence-gate] worker error:", e.message || e);
        if (_FENCE_GATE.status === "loading") {
          _FENCE_GATE.status = "unavailable";
          resolve(false);
        }
      };

      /* Single onmessage demuxer — dispatches each reply to the
       * per-request callbacks registered in _FENCE_GATE._pending. */
      worker.onmessage = (e) => {
        const m = e.data;
        if (m && m.type === "log") {
          if (m.level === "warn") console.warn("[fence-gate]", m.message);
          return;
        }
        if (!m || m.id === undefined || !_FENCE_GATE._pending.has(m.id))
          return;
        const handlers = _FENCE_GATE._pending.get(m.id);
        if (m.type === "progress") {
          if (handlers.onProgress) handlers.onProgress(m.data);
        } else if (m.type === "ready") {
          _FENCE_GATE._pending.delete(m.id);
          if (handlers.onReady) handlers.onReady(m);
        } else if (m.type === "result") {
          _FENCE_GATE._pending.delete(m.id);
          if (handlers.onResult) handlers.onResult(m.data);
        } else if (m.type === "error") {
          _FENCE_GATE._pending.delete(m.id);
          if (handlers.onError) handlers.onError(m.error);
        }
      };

      const initId = _FENCE_GATE._msgId++;
      _FENCE_GATE._pending.set(initId, {
        onProgress: (p) => {
          if (_FENCE_GATE.progressCb) {
            try {
              _FENCE_GATE.progressCb(p);
            } catch {
              /* swallow UI errors */
            }
          }
        },
        onReady: (m) => {
          _FENCE_GATE.deviceUsed = m.device;
          _FENCE_GATE.dtypeUsed = m.dtype;
          _FENCE_GATE.status = "ready";
          resolve(true);
        },
        onError: (err) => {
          console.warn(
            "[fence-gate] init failed, gate disabled (fail-open):",
            err,
          );
          _FENCE_GATE.status = "unavailable";
          resolve(false);
        },
      });
      worker.postMessage({ id: initId, type: "init" });
    } catch (e) {
      console.warn(
        "[fence-gate] worker creation failed, gate disabled (fail-open):",
        e,
      );
      _FENCE_GATE.status = "unavailable";
      resolve(false);
    }
  });

  return _FENCE_GATE.loadPromise;
}

/* Convert any plausible "image" input (HTMLImageElement, Canvas,
 * ImageBitmap, string URL, or Blob) into a STRING URL that
 * transformers.js v2 RawImage.read() accepts.
 *
 * v2.17.2's RawImage.read() only handles strings or RawImage
 * instances — not Blobs, not DOM elements. So for anything other
 * than a string, we render to a downscaled JPEG Blob (SigLIP
 * internally rescales to 224×224, so anything beyond ~512 is
 * wasted decode work) and then wrap that Blob in an object URL.
 *
 * Returns { url, cleanup } where cleanup() revokes the object URL —
 * callers MUST invoke cleanup() after the classifier call to avoid
 * the Blob staying resident in memory. */
async function _imageToGateInput(image) {
  if (!image) return null;
  const srcW = image.naturalWidth || image.width || 0;
  const srcH = image.naturalHeight || image.height || 0;
  if (!srcW || !srcH) {
    throw new Error("image has zero dimensions");
  }
  /* createImageBitmap with resize options runs the decode + downscale
   * OFF the main thread. The bitmap is transferable, so postMessage
   * with the transfer list hands ownership to the worker with no
   * copy. transformers.js v3 accepts ImageBitmap as a pipeline input,
   * so no Blob / URL conversion is needed. */
  const MAX_DIM = 768;
  const scale = Math.min(1, MAX_DIM / Math.max(srcW, srcH));
  const outW = Math.max(1, Math.round(srcW * scale));
  const outH = Math.max(1, Math.round(srcH * scale));
  let bitmap;
  try {
    bitmap = await createImageBitmap(image, {
      resizeWidth: outW,
      resizeHeight: outH,
      resizeQuality: "medium",
    });
  } catch {
    /* Some older Safari versions ignore resize options. Fall back. */
    bitmap = await createImageBitmap(image);
  }
  return {
    bitmap,
    cleanup: () => {
      try {
        bitmap && bitmap.close && bitmap.close();
      } catch {
        /* already transferred or closed */
      }
    },
    width: outW,
    height: outH,
  };
}

async function _checkIsWoodenFence(image, opts = {}) {
  const FAIL_OPEN = (reason) => ({
    isFence: true,
    confidence: 0,
    reason,
    detections: 0,
    bestDetection: null,
    inferenceMs: 0,
  });

  if (!image) return FAIL_OPEN("no-image");

  const ready = await _initFenceGate({ onProgress: opts.onProgress });
  if (!ready) return FAIL_OPEN("gate-unavailable");

  let input;
  try {
    input = await _imageToGateInput(image);
  } catch (e) {
    console.warn("[fence-gate] image conversion failed, letting through:", e);
    return FAIL_OPEN("image-conversion-error");
  }
  if (!input) return FAIL_OPEN("no-image");

  try {
    const t0 = performance.now();
    let detections = [];
    try {
      /* Transfer the ImageBitmap directly to the worker with no
       * copy. transformers.js v3 accepts ImageBitmap as a pipeline
       * input, so no Blob / URL conversion is needed on either side.
       * The worker calls bitmap.close() after use. */
      if (!input.bitmap) {
        throw new Error("worker path requires an ImageBitmap input");
      }
      const bitmap = input.bitmap;
      detections = await new Promise((resolve, reject) => {
        const id = _FENCE_GATE._msgId++;
        _FENCE_GATE._pending.set(id, {
          onResult: (data) => resolve(data),
          onError: (err) => reject(new Error(err)),
        });
        _FENCE_GATE.worker.postMessage(
          {
            id,
            type: "infer",
            data: {
              bitmap,
              /* Feed BOTH query groups to the model in one inference
               * call — each query is a cheap text encode, and we
               * need both sets scored against the image to do the
               * fence-vs-distractor comparison in the decision
               * rule below. */
              queries: [..._FENCE_QUERIES, ..._FENCE_DISTRACTOR_QUERIES],
              /* OWL-ViT v1 sigmoid scores can be much smaller than
               * OWLv2's; set pipeline threshold near zero so the
               * model returns ALL candidates and we filter in the
               * decision rule. */
              options: { threshold: 0.001, topk: 50, percentage: true },
            },
          },
          [bitmap],
        );
      });
    } finally {
      input.cleanup();
    }
    const inferenceMs = performance.now() - t0;

    /* Two-class decision: we feed BOTH fence and distractor queries
     * to OWL-ViT, then compare the strongest fence detection
     * against the strongest distractor detection. The image passes
     * only if a fence query OUT-SCORES every distractor query by
     * at least COMPETITIVE_MARGIN.
     *
     * Why this works where score-threshold alone failed:
     *   Wooden stairs scored 0.065 on "wooden fence" — higher than
     *   the real fence's own 0.040 — so no single threshold could
     *   split them. But "wooden stairs" as a query will score even
     *   higher than 0.065 on a stairs image; "wooden fence" wins on
     *   a real fence image. The relative ranking is what we trust.
     *
     * Thresholds:
     *   SCORE_THRESHOLD     — minimum fence score for any
     *     consideration. Drops obvious noise.
     *   AREA_THRESHOLD      — minimum box area as fraction of image.
     *     Drops 1-pixel detections.
     *   COMPETITIVE_MARGIN  — fence must beat distractor by this
     *     amount. Avoids passing on close ties. */
    /* SCORE_THRESHOLD: noise floor for fence detections.
     * AREA_THRESHOLD: noise floor for any detection.
     * DISTRACTOR_AREA_MIN: stricter area requirement for
     *   distractors. Without it, OWL-ViT spuriously matches
     *   "wooden stairs" to small wooden elements like hot-tub
     *   plank surrounds (~7% area) which then wrongly beat a real
     *   fence at 16% area. Legitimate distractor objects (the
     *   photo's actual subject is stairs / table / floor / etc.)
     *   cover much more than 10% of the frame, so this filter
     *   cleanly drops the spurious cases.
     * COMPETITIVE_MARGIN: minimum gap by which fence must outscore
     *   distractor for PASS. */
    const SCORE_THRESHOLD = 0.01;
    const AREA_THRESHOLD = 0.02;
    const DISTRACTOR_AREA_MIN = 0.10;
    // Fence must out-score the strongest distractor. OWL-ViT scores here are
    // tiny (~0.01), so the margin is RELATIVE (scale-invariant) plus a small
    // absolute floor. An absolute-only 0.005 was ~40% of a 0.01 score and
    // wrongly BLOCKed real fences that merely LOOK deck-like (frontal plank
    // walls: fence 0.014 vs deck 0.012).
    const COMPETITIVE_MARGIN_REL = 0.06; // fence must beat distractor by >=6%
    const COMPETITIVE_MARGIN_ABS = 0.0005; // + floor so exact noise-ties don't pass
    const fenceSet = new Set(_FENCE_QUERIES);
    const distractorSet = new Set(_FENCE_DISTRACTOR_QUERIES);

    const computeAreaFrac = (box) => {
      const bw = (box.xmax ?? 0) - (box.xmin ?? 0);
      const bh = (box.ymax ?? 0) - (box.ymin ?? 0);
      let af = bw * bh;
      if (af > 1.5) {
        const iw = input.width || 1;
        const ih = input.height || 1;
        af = (bw * bh) / Math.max(1, iw * ih);
      }
      return af;
    };

    let bestFence = null;
    let bestDistractor = null;
    for (const det of detections) {
      if (det.score < SCORE_THRESHOLD) continue;
      const areaFrac = computeAreaFrac(det.box || {});
      if (areaFrac < AREA_THRESHOLD) continue;
      const dec = { ...det, areaFrac };
      if (fenceSet.has(det.label)) {
        if (!bestFence || det.score > bestFence.score) {
          bestFence = dec;
        }
      } else if (distractorSet.has(det.label)) {
        /* Distractors get a stricter area requirement to drop
         * spurious tiny matches that wrongly beat real fences. */
        if (areaFrac < DISTRACTOR_AREA_MIN) continue;
        if (!bestDistractor || det.score > bestDistractor.score) {
          bestDistractor = dec;
        }
      }
    }

    const distractorScore = bestDistractor?.score ?? 0;
    const isFence =
      !!bestFence &&
      bestFence.score >
        distractorScore * (1 + COMPETITIVE_MARGIN_REL) + COMPETITIVE_MARGIN_ABS;

    const verdict = {
      isFence,
      confidence: bestFence?.score || 0,
      reason: isFence ? "fence-detected" : "no-fence-detected",
      detections: detections.length,
      bestFence,
      bestDistractor,
      inferenceMs,
    };
    _FENCE_GATE.lastResult = verdict;
    return verdict;
  } catch (e) {
    console.warn("[fence-gate] inference failed, letting through:", e);
    return FAIL_OPEN("inference-error");
  }
}

async function previewStain() {
  if (!originalImage) return;
  if (!maskData) {
    /* Pre-filter: skip the server pipeline for obvious non-fence uploads. */
    showLoading("Checking image…");
    updateStatus("Checking image…", "loading");
    const verdict = await _checkIsWoodenFence(originalImage, {
      onProgress: (p) => {
        if (
          p &&
          p.status === "progress" &&
          typeof p.progress === "number"
        ) {
          const pct = Math.max(0, Math.min(100, p.progress)).toFixed(0);
          const msg = `Preparing fence detector… ${pct}%`;
          /* Route to both the visible in-canvas overlay and the hidden
           * a11y status node so screen readers and visual users both
           * see progress. */
          showLoading(msg);
          updateStatus(msg, "loading");
        }
      },
    });
    if (!verdict.isFence) {
      hideLoading();
      updateStatus(
        "No fence detected in this photo. Please upload a clear photo of a wooden fence.",
        "error",
      );
      return;
    }
    await detectFence();
    if (!maskData) return;
  }
  await recolorFence({
    loadingMessage: "Applying stain...",
    successMessage: "Stain applied!",
  });
}

fileInput.addEventListener("change", handleImageUpload);
detectBtn.addEventListener("click", previewStain);
recolorBtn.addEventListener("click", recolorFence);
cleanBtn.addEventListener("click", cleanFence);
downloadBtn.addEventListener("click", downloadResult);
if (downloadBtnMobile)
  downloadBtnMobile.addEventListener("click", downloadResult);
resetBtn.addEventListener("click", reset);

const changeImageBtn = document.getElementById("change-image-btn");
const changeImageBtnMobile = document.getElementById("change-image-btn-mobile");
const uploadBtn = document.getElementById("upload-btn");
const openFilePicker = () => fileInput.click();
if (uploadBtn) uploadBtn.addEventListener("click", openFilePicker);
if (changeImageBtn) changeImageBtn.addEventListener("click", openFilePicker);
if (changeImageBtnMobile)
  changeImageBtnMobile.addEventListener("click", openFilePicker);

opacity.addEventListener("input", () => {
  document.getElementById("opacity-value").textContent = opacity.value + "%";
});

threshold.addEventListener("input", () => {
  document.getElementById("threshold-value").textContent = threshold.value;

  if (maskData) {
    detectFence();
  }
});

edgeSmoothing.addEventListener("input", () => {
  const labels = ["None", "Light", "Medium", "Strong", "Maximum"];
  document.getElementById("smoothing-value").textContent =
    labels[edgeSmoothing.value - 1];
});

uploadSection.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadSection.classList.add("drag-over");
});

uploadSection.addEventListener("dragleave", () => {
  uploadSection.classList.remove("drag-over");
});

uploadSection.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadSection.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    handleImageFile(file);
  }
});

function handleImageUpload(event) {
  const file = event.target.files[0];
  if (file) {
    handleImageFile(file);
  }
}

function handleImageFile(file) {
  if (file.size > 10 * 1024 * 1024) {
    updateStatus("File too large! Max 10MB", "error");
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      maskData = null;
      cleanedImageData = null;
      resultState = "original";
      const mctx = maskCanvas.getContext("2d");
      const rctx = resultCanvas.getContext("2d");
      mctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      rctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
      maskCanvas.width = 800;
      maskCanvas.height = 620;
      resultCanvas.width = 800;
      resultCanvas.height = 620;
      setDownloadEnabled(false);

      originalImage = img;
      /* Kick off the SigLIP fence-gate model download in the
       * background. By the time the user clicks Apply Stain,
       * the model is usually ready and the gate adds only
       * ~200-300 ms instead of the ~10-20 s cold download.
       * Fire-and-forget — any error is swallowed inside
       * _initFenceGate. */
      _initFenceGate().catch(() => {});
      drawOriginalImage();
      drawOriginalToResult();
      canvasStack.classList.add("has-image");

      compareBtn.hidden = true;
      hideCompareUI(true);
      if (canvasLabel) canvasLabel.textContent = "Original";
      if (canvasTitleText) canvasTitleText.textContent = "Fence Preview";
      detectBtn.disabled = false;
      cleanBtn.disabled = false;
      resetBtn.disabled = false;
      setColorChipsEnabled(true);
      showLoadedState();
      updateStatus(
        'Image loaded! Click "Clean Fence" to restore, or "Apply Stain" to add color.',
        "success",
      );

      const scrollTarget = canvasCardSection;
      if (scrollTarget) {
        const isMobile = window.matchMedia("(max-width: 900px)").matches;
        requestAnimationFrame(() =>
          scrollTarget.scrollIntoView({
            behavior: "smooth",
            block: isMobile ? "start" : "center",
          }),
        );
      }
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function drawOriginalImage() {
  originalCanvas.width = originalImage.width;
  originalCanvas.height = originalImage.height;
  const ctx = originalCanvas.getContext("2d");
  ctx.drawImage(originalImage, 0, 0);
}

function drawOriginalToResult() {
  if (!originalImage) return;
  resultCanvas.width = originalImage.width;
  resultCanvas.height = originalImage.height;
  const ctx = resultCanvas.getContext("2d");
  ctx.drawImage(originalImage, 0, 0);
}

const canvasStack = document.getElementById("canvas-stack");
const canvasLabel = document.getElementById("canvas-label");
const canvasTitleText = document.getElementById("canvas-title-text");
const compareBtn = document.getElementById("compare-btn");
const compareTip = document.getElementById("compare-tip");
const compareSlider = document.getElementById("compare-slider");
const compareHandle = document.getElementById("compare-handle");

let compareMode = false;
let tipDismissTimer = null;
let tipShownThisSession = false;

function setComparePosition(pct) {
  const clamped = Math.max(0, Math.min(100, pct));
  canvasStack.style.setProperty("--compare-pct", clamped + "%");
}

function showCompareButton() {
  compareBtn.hidden = false;
}

function hideCompareUI(resetTip) {
  compareMode = false;
  canvasStack.classList.remove("compare-on");
  compareSlider.hidden = true;
  setComparePosition(50);
  if (resetTip) {
    tipShownThisSession = false;
    compareTip.hidden = true;
    compareTip.classList.remove("fading");
    if (tipDismissTimer) {
      clearTimeout(tipDismissTimer);
      tipDismissTimer = null;
    }
  }
}

function showCoachTip() {
  if (tipShownThisSession) return;
  tipShownThisSession = true;
  compareTip.hidden = false;
  compareTip.classList.remove("fading");
  if (tipDismissTimer) clearTimeout(tipDismissTimer);
  tipDismissTimer = setTimeout(dismissCoachTip, 4500);
}

function dismissCoachTip() {
  if (compareTip.hidden) return;
  compareTip.classList.add("fading");
  setTimeout(() => {
    compareTip.hidden = true;
    compareTip.classList.remove("fading");
  }, 380);
}

function animateCompare(from, to, duration, onDone) {
  const start = performance.now();
  function frame(now) {
    const t = Math.min(1, (now - start) / Math.max(1, duration));
    const eased = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    setComparePosition(from + (to - from) * eased);
    if (t < 1) requestAnimationFrame(frame);
    else if (typeof onDone === "function") onDone();
  }
  requestAnimationFrame(frame);
}

async function enterCompareMode() {
  if (!originalImage || !maskData) return;
  if (compareMode) return;
  compareMode = true;

  drawOriginalImage();
  compareSlider.hidden = false;
  canvasStack.classList.add("compare-on");

  setComparePosition(100);
  await new Promise((r) => setTimeout(r, 60));
  animateCompare(100, 0, 900, () => animateCompare(0, 50, 500));
}

function exitCompareMode() {
  compareMode = false;
  canvasStack.classList.remove("compare-on");
  compareSlider.hidden = true;
  setComparePosition(50);
}

function toggleCompareMode() {
  dismissCoachTip();
  if (compareMode) exitCompareMode();
  else enterCompareMode();
}

let dragging = false;
function pctFromEventX(clientX) {
  const rect = canvasStack.getBoundingClientRect();
  return ((clientX - rect.left) / Math.max(1, rect.width)) * 100;
}
compareHandle.addEventListener("pointerdown", (e) => {
  if (!compareMode) return;
  dragging = true;
  try {
    compareHandle.setPointerCapture(e.pointerId);
  } catch (_) {}
  e.preventDefault();
});
compareHandle.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  setComparePosition(pctFromEventX(e.clientX));
  e.preventDefault();
});
function endDrag(e) {
  if (!dragging) return;
  dragging = false;
  try {
    compareHandle.releasePointerCapture(e.pointerId);
  } catch (_) {}
}
compareHandle.addEventListener("pointerup", endDrag);
compareHandle.addEventListener("pointercancel", endDrag);
compareHandle.addEventListener("lostpointercapture", endDrag);

canvasStack.addEventListener("pointerdown", (e) => {
  if (!compareMode) return;
  if (
    e.target.closest(".compare-handle") ||
    e.target.closest(".canvas-overlay-btn") ||
    e.target.closest(".canvas-tip")
  )
    return;
  setComparePosition(pctFromEventX(e.clientX));
});

compareBtn.addEventListener("click", toggleCompareMode);

function autoLevelsCanvas(canvas, lowPct, highPct) {
  lowPct = lowPct != null ? lowPct : 0.015;
  highPct = highPct != null ? highPct : 0.985;
  const ctx = canvas.getContext("2d");
  const w = canvas.width,
    h = canvas.height;
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;
  const totalPx = w * h;

  const hist = new Uint32Array(256);
  for (let i = 0; i < data.length; i += 4) {
    const lum = ((data[i] + data[i + 1] + data[i + 2]) / 3) | 0;
    hist[lum]++;
  }

  const lowTarget = totalPx * lowPct;
  const highTarget = totalPx * highPct;
  let cum = 0,
    lo = 0,
    hi = 255,
    foundLo = false;
  for (let i = 0; i < 256; i++) {
    cum += hist[i];
    if (!foundLo && cum >= lowTarget) {
      lo = i;
      foundLo = true;
    }
    if (cum >= highTarget) {
      hi = i;
      break;
    }
  }

  if (hi - lo < 10) return;
  if (lo <= 8 && hi >= 247) return;

  const lut = new Uint8ClampedArray(256);
  const scale = 255 / (hi - lo);
  for (let i = 0; i < 256; i++) {
    if (i <= lo) lut[i] = 0;
    else if (i >= hi) lut[i] = 255;
    else lut[i] = (i - lo) * scale;
  }

  for (let i = 0; i < data.length; i += 4) {
    data[i] = lut[data[i]];
    data[i + 1] = lut[data[i + 1]];
    data[i + 2] = lut[data[i + 2]];
  }
  ctx.putImageData(imageData, 0, 0);
}

async function imageToUploadBlob(img, maxDim, quality, enhanceMode = "mild") {
  const srcW = img.naturalWidth || img.width;
  const srcH = img.naturalHeight || img.height;
  const scale = Math.min(1.0, maxDim / Math.max(srcW, srcH));
  const dstW = Math.max(1, Math.round(srcW * scale));
  const dstH = Math.max(1, Math.round(srcH * scale));

  const canvas = document.createElement("canvas");
  canvas.width = dstW;
  canvas.height = dstH;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  if (enhanceMode === "mild") {
    ctx.filter = "contrast(1.12) saturate(1.06)";
  } else if (enhanceMode === "aggressive") {
    ctx.filter = "contrast(1.35) saturate(1.18) brightness(0.88)";
  } else {
    ctx.filter = "none";
  }
  ctx.drawImage(img, 0, 0, dstW, dstH);
  ctx.filter = "none";

  if (enhanceMode !== "none") {
    if (enhanceMode === "aggressive") {
      autoLevelsCanvas(canvas, 0.03, 0.97);
    } else {
      autoLevelsCanvas(canvas, 0.015, 0.985);
    }
  }

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) =>
        blob ? resolve(blob) : reject(new Error("toBlob returned null")),
      "image/jpeg",
      quality,
    );
  });
}

async function _uploadAndInfer(image, enhanceMode) {
  const uploadBlob = await imageToUploadBlob(
    image,
    CONFIG.UPLOAD_MAX_DIM || 1024,
    CONFIG.UPLOAD_JPEG_QUALITY || 0.85,
    enhanceMode,
  );

  const formData = new FormData();
  formData.append("image", uploadBlob, "fence.jpg");
  const t0 = performance.now();
  const response = await fetch(CONFIG.MODAL_ENDPOINT, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const txt = await response.text().catch(() => "");
    throw new Error(`HTTP ${response.status}: ${txt.slice(0, 200)}`);
  }
  const totalMs = performance.now() - t0;
  const serverMs = response.headers.get("X-Inference-Ms") || "?";

  const maskBlob = await response.blob();
  const maskBitmap = await createImageBitmap(maskBlob);
  const w = maskBitmap.width,
    h = maskBitmap.height;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const cx = c.getContext("2d");
  cx.imageSmoothingEnabled = false;
  cx.drawImage(maskBitmap, 0, 0);
  const px = cx.getImageData(0, 0, w, h).data;
  const probData = new Float32Array(w * h);
  for (let i = 0; i < probData.length; i++) probData[i] = px[i * 4] / 255.0;
  return { data: probData, dims: [1, 1, w, h] };
}

function _postprocessDims() {
  const maxPpDim = CONFIG.POSTPROCESS_MAX_DIM || 1536;
  const W0 = originalImage.width;
  const H0 = originalImage.height;
  const scale = Math.min(1, maxPpDim / Math.max(W0, H0));
  const ppW = Math.max(1, Math.round(W0 * scale));
  const ppH = Math.max(1, Math.round(H0 * scale));
  return { W0, H0, ppW, ppH };
}

// Scale the fixed-pixel-radius / pixel-count CONFIG knobs so the post-process
// behavior at the downsized working resolution matches what would happen at
// the original source resolution.
//   - Pixel RADIUS / linear-pixel params scale linearly (× s).
//   - Pixel COUNT (area) params scale quadratically (× s²) because a region's
//     pixel count drops by s² when both dimensions shrink by s.
//   - Math.max(1, …) prevents collapse to 0 on extreme downscales.
//   - Caller-provided opts.X always wins (we merge our scaled defaults LAST
//     under the caller's opts).
function _scalePostprocessOpts(callerOpts, scale) {
  const px = (val, fallback) =>
    Math.max(1, Math.round((val != null ? val : fallback) * scale));
  const ct = (val, fallback) =>
    Math.max(1, Math.round((val != null ? val : fallback) * scale * scale));
  const scaledDefaults = {
    recoveryDilatePx: px(CONFIG.RECOVERY_DILATE_PX, 35),
    recovery2DilatePx: px(CONFIG.RECOVERY2_DILATE_PX, 25),
    bottomExtendMaxPx: px(CONFIG.BOTTOM_EXTEND_MAX_PX, 40),
    buildingMinCcPx: ct(CONFIG.BUILDING_MIN_CC_PX, 500),
    buildingWallMinCcPx: ct(CONFIG.BUILDING_WALL_MIN_CC_PX, 1000),
    ccOutlierMinPx: ct(CONFIG.CC_OUTLIER_MIN_PX, 300),
    ccAxisMinPx: ct(CONFIG.CC_AXIS_MIN_PX, 500),
    orientationMinCount: ct(CONFIG.ORIENTATION_MIN_COUNT, 500),
  };
  return Object.assign(scaledDefaults, callerOpts || {});
}

async function _postprocess(output, opts) {
  const smoothingKernel = parseInt(edgeSmoothing.value);
  const thresholdValue = parseFloat(threshold.value);
  const { W0, H0, ppW, ppH } = _postprocessDims();
  const scale = ppW / W0;
  const origImageData = getScaledImagePixelData(originalImage, ppW, ppH);
  const scaledOpts = _scalePostprocessOpts(opts, scale);
  const ppMask = await postprocessMask(
    output,
    ppW,
    ppH,
    smoothingKernel,
    thresholdValue,
    origImageData,
    scaledOpts,
  );
  if (ppW === W0 && ppH === H0) return ppMask;
  // Upsample the final mask back to source dims so downstream draw/recolor/
  // clean operations composite onto the original-resolution canvas without
  // size mismatches.
  return bilinearResize(ppMask, ppW, ppH, W0, H0);
}

function _hasAnyFence(mask) {
  for (let i = 0; i < mask.length; i++) if (mask[i] > 0) return true;
  return false;
}

const _MODERATE_POSTPROCESS_OPTS = {
  softLow: 0.55,
  softHigh: 0.75,

  ccMinPct: 0.05,
};

const _RELAXED_POSTPROCESS_OPTS = {
  softLow: 0.4,
  softHigh: 0.65,
  ccMinPct: 0.5,

  skipSpatialRecovery: true,

  skipOrientation: true,

  skipCcAxis: true,

  skipJunkBlobs: true,
};

async function _uploadTile(image, tx, ty, tw, th, enhanceMode) {
  const canvas = document.createElement("canvas");
  canvas.width = tw;
  canvas.height = th;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(image, tx, ty, tw, th, 0, 0, tw, th);
  const tileBitmap = await createImageBitmap(canvas);
  return await _uploadAndInfer(tileBitmap, enhanceMode);
}

async function _tiledInference(image, enhanceMode, gridX, gridY, overlap) {
  const W = image.naturalWidth || image.width;
  const H = image.naturalHeight || image.height;
  const GRID_X = gridX || 3,
    GRID_Y = gridY || 2,
    OVERLAP = overlap || 0.25;
  const stepX = Math.floor(W / GRID_X);
  const stepY = Math.floor(H / GRID_Y);
  const tileW = Math.min(W, Math.floor(stepX * (1 + OVERLAP)));
  const tileH = Math.min(H, Math.floor(stepY * (1 + OVERLAP)));

  const tiles = [];
  for (let gy = 0; gy < GRID_Y; gy++) {
    for (let gx = 0; gx < GRID_X; gx++) {
      const tx = Math.min(gx * stepX, W - tileW);
      const ty = Math.min(gy * stepY, H - tileH);
      tiles.push({ tx, ty, tw: tileW, th: tileH });
    }
  }

  const outputs = await Promise.all(
    tiles.map((t) => _uploadTile(image, t.tx, t.ty, t.tw, t.th, enhanceMode)),
  );

  const fullMask = new Float32Array(W * H);
  for (let i = 0; i < tiles.length; i++) {
    const t = tiles[i];
    const data = outputs[i].data;
    const ms = outputs[i].dims[2];
    for (let py = 0; py < t.th; py++) {
      const sy = Math.min(ms - 1, Math.floor((py * ms) / t.th));
      const dy = (t.ty + py) * W;
      const sRow = sy * ms;
      for (let px = 0; px < t.tw; px++) {
        const sx = Math.min(ms - 1, Math.floor((px * ms) / t.tw));
        const sval = data[sRow + sx];
        const idx = dy + (t.tx + px);
        if (sval > fullMask[idx]) fullMask[idx] = sval;
      }
    }
  }

  const OUT = 512;
  const downsampled = new Float32Array(OUT * OUT);
  for (let y = 0; y < OUT; y++) {
    const sy = Math.min(H - 1, Math.floor((y * H) / OUT));
    const rowOff = sy * W;
    for (let x = 0; x < OUT; x++) {
      const sx = Math.min(W - 1, Math.floor((x * W) / OUT));
      downsampled[y * OUT + x] = fullMask[rowOff + sx];
    }
  }

  let maxConf = 0,
    sumConf = 0;
  for (let i = 0; i < downsampled.length; i++) {
    if (downsampled[i] > maxConf) maxConf = downsampled[i];
    sumConf += downsampled[i];
  }
  const meanConf = sumConf / downsampled.length;
  return { data: downsampled, dims: [1, 1, OUT, OUT] };
}

async function _tiledInferenceMultiScale(image, enhanceMode) {
  const [coarse, medium, fine] = await Promise.all([
    _tiledInference(image, enhanceMode, 3, 2, 0.25),
    _tiledInference(image, enhanceMode, 4, 3, 0.25),
    _tiledInference(image, enhanceMode, 5, 4, 0.35),
  ]);
  const merged = new Float32Array(coarse.data.length);
  let maxC = 0,
    maxM = 0,
    maxF = 0,
    maxU = 0;
  for (let i = 0; i < merged.length; i++) {
    const c = coarse.data[i];
    const m = medium.data[i];
    const f = fine.data[i];
    let best = c > m ? c : m;
    if (f > best) best = f;
    merged[i] = best;
    if (c > maxC) maxC = c;
    if (m > maxM) maxM = m;
    if (f > maxF) maxF = f;
    if (best > maxU) maxU = best;
  }
  return { data: merged, dims: coarse.dims };
}

function _rgbToLab(r, g, b) {
  const lin = (c) =>
    c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  r = lin(r);
  g = lin(g);
  b = lin(b);
  const X = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
  const Y = 0.2126729 * r + 0.7151522 * g + 0.072175 * b;
  const Z = (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883;
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fX = f(X),
    fY = f(Y),
    fZ = f(Z);
  return [116 * fY - 16, 500 * (fX - fY), 200 * (fY - fZ)];
}

function _checkMaskSurroundContext(
  maskData,
  image,
  imageWidth,
  imageHeight,
  options,
) {
  const {
    ringWidthFrac = 0.08,
    greenRatioMin = 0.5,
    sampleStride = 3,
  } = options || {};

  let minX = imageWidth,
    maxX = -1,
    minY = imageHeight,
    maxY = -1;
  for (let y = 0; y < imageHeight; y++) {
    const rowOff = y * imageWidth;
    for (let x = 0; x < imageWidth; x++) {
      if (maskData[rowOff + x] > 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return true;

  const ringWidth = Math.max(
    8,
    Math.floor(Math.min(imageWidth, imageHeight) * ringWidthFrac),
  );
  const expMinX = Math.max(0, minX - ringWidth);
  const expMaxX = Math.min(imageWidth - 1, maxX + ringWidth);
  const expMinY = Math.max(0, minY - ringWidth);
  const expMaxY = Math.min(imageHeight - 1, maxY + ringWidth);

  const canvas = document.createElement("canvas");
  canvas.width = imageWidth;
  canvas.height = imageHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0, imageWidth, imageHeight);
  const rgba = ctx.getImageData(0, 0, imageWidth, imageHeight).data;

  let greenCount = 0,
    builtCount = 0,
    skyCount = 0,
    totalSampled = 0;
  for (let y = expMinY; y <= expMaxY; y += sampleStride) {
    const rowOff = y * imageWidth;
    for (let x = expMinX; x <= expMaxX; x += sampleStride) {
      if (maskData[rowOff + x] > 0) continue;

      const inRing = x < minX || x > maxX || y < minY || y > maxY;
      if (!inRing) continue;

      const pi = (rowOff + x) * 4;
      const lab = _rgbToLab(
        rgba[pi] / 255,
        rgba[pi + 1] / 255,
        rgba[pi + 2] / 255,
      );
      const L = lab[0],
        A = lab[1],
        B = lab[2];
      totalSampled++;

      if (A < -3) {
        greenCount++;
      } else if (
        (A > 8 && B > -5 && L < 70) ||
        (L > 65 && Math.abs(A) < 5 && L < 92 && B > -1)
      ) {
        builtCount++;
      } else if (L > 65 && Math.abs(A) < 8 && B < 0) {
        skyCount++;
      }
    }
  }

  if (totalSampled === 0) {
    return true;
  }

  const greenRatio = greenCount / totalSampled;
  const builtRatio = builtCount / totalSampled;
  const skyRatio = skyCount / totalSampled;

  if (greenRatio < greenRatioMin) {
    return false;
  }
  return true;
}

function _filterMaskByShape(maskData, imageWidth, imageHeight, options) {
  const {
    minAspect = 1.0,
    maxHeightFrac = 0.95,
    minDensity = 0.2,
    densityCheckMaxAspect = 2.5,
    minPxForFilter = 1000,
  } = options || {};

  let minX = imageWidth,
    minY = imageHeight;
  let maxX = -1,
    maxY = -1;
  let nonZeroCount = 0;
  for (let y = 0; y < imageHeight; y++) {
    const rowOff = y * imageWidth;
    for (let x = 0; x < imageWidth; x++) {
      if (maskData[rowOff + x] > 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        nonZeroCount++;
      }
    }
  }

  if (nonZeroCount < minPxForFilter) {
    return maskData;
  }

  const bboxWidth = maxX - minX + 1;
  const bboxHeight = maxY - minY + 1;
  const aspect = bboxWidth / bboxHeight;
  const heightFrac = bboxHeight / imageHeight;
  const density = nonZeroCount / (bboxWidth * bboxHeight);

  const shapeBad = aspect < minAspect && heightFrac > maxHeightFrac;

  const tooSparse = density < minDensity && aspect < densityCheckMaxAspect;

  if (shapeBad || tooSparse) {
    const reasons = [];
    if (shapeBad)
      reasons.push(
        `shape (aspect ${aspect.toFixed(2)}<${minAspect} AND heightFrac ${heightFrac.toFixed(2)}>${maxHeightFrac})`,
      );
    if (tooSparse)
      reasons.push(
        `density ${density.toFixed(2)}<${minDensity} AND aspect ${aspect.toFixed(2)}<${densityCheckMaxAspect}`,
      );
    return new Float32Array(maskData.length);
  }
  return maskData;
}

function _gateTilesByWholeImage(tileOutput, wholeImageOutput, options) {
  const { pixelGateThreshold = 0.08, dilatePx = 30 } = options || {};

  const N = tileOutput.length;
  if (wholeImageOutput.length !== N) {
    console.warn("[tile-gate] dimension mismatch; skipping gate");
    return tileOutput;
  }
  const size = Math.sqrt(N) | 0;
  if (size * size !== N) {
    console.warn("[tile-gate] not square; skipping gate");
    return tileOutput;
  }

  let wholeImageMax = 0;
  let allowedCount = 0;
  for (let i = 0; i < N; i++) {
    if (wholeImageOutput[i] > wholeImageMax)
      wholeImageMax = wholeImageOutput[i];
    if (wholeImageOutput[i] > pixelGateThreshold) allowedCount++;
  }

  if (allowedCount === 0) {
    return tileOutput;
  }

  const dist = new Int16Array(N);
  dist.fill(-1);
  const queue = new Int32Array(N);
  let qHead = 0,
    qTail = 0;
  for (let i = 0; i < N; i++) {
    if (wholeImageOutput[i] > pixelGateThreshold) {
      dist[i] = 0;
      queue[qTail++] = i;
    }
  }

  while (qHead < qTail) {
    const idx = queue[qHead++];
    const d = dist[idx];
    if (d >= dilatePx) continue;
    const x = idx % size;
    const y = (idx / size) | 0;
    if (x > 0 && dist[idx - 1] === -1) {
      dist[idx - 1] = d + 1;
      queue[qTail++] = idx - 1;
    }
    if (x < size - 1 && dist[idx + 1] === -1) {
      dist[idx + 1] = d + 1;
      queue[qTail++] = idx + 1;
    }
    if (y > 0 && dist[idx - size] === -1) {
      dist[idx - size] = d + 1;
      queue[qTail++] = idx - size;
    }
    if (y < size - 1 && dist[idx + size] === -1) {
      dist[idx + size] = d + 1;
      queue[qTail++] = idx + size;
    }
  }

  const gated = new Float32Array(N);
  let expandedAllowed = 0,
    kept = 0,
    droppedPx = 0;
  for (let i = 0; i < N; i++) {
    if (dist[i] >= 0) {
      gated[i] = tileOutput[i];
      expandedAllowed++;
      if (tileOutput[i] > 0.5) kept++;
    } else {
      gated[i] = 0;
      if (tileOutput[i] > 0.5) droppedPx++;
    }
  }
  return gated;
}

function _maskCompleteByColorPropagation(softMask, image, options) {
  const OUT = Math.sqrt(softMask.length) | 0;
  if (OUT * OUT !== softMask.length) {
    console.warn("[mask-prop] softMask is not square; skipping");
    return softMask;
  }

  const {
    seedThreshold = 0.6,
    candidateMin = 0.0,
    candidateMax = 0.55,
    colorStdDev = 3.0,
    spatialRadiusPx = 50,
    maxBoost = 0.5,
    minSeedCount = 200,
    colorClusters = 3,
    kmeansIterations = 6,
  } = options || {};

  const canvas = document.createElement("canvas");
  canvas.width = OUT;
  canvas.height = OUT;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(image, 0, 0, OUT, OUT);
  const rgba = ctx.getImageData(0, 0, OUT, OUT).data;

  const N = OUT * OUT;
  const labL = new Float32Array(N);
  const labA = new Float32Array(N);
  const labB = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const lab = _rgbToLab(
      rgba[i * 4] / 255,
      rgba[i * 4 + 1] / 255,
      rgba[i * 4 + 2] / 255,
    );
    labL[i] = lab[0];
    labA[i] = lab[1];
    labB[i] = lab[2];
  }

  const seeds = new Uint8Array(N);
  let seedCount = 0;

  const seedIndices = [];
  for (let i = 0; i < N; i++) {
    if (softMask[i] > seedThreshold) {
      seeds[i] = 1;
      seedCount++;
      seedIndices.push(i);
    }
  }
  if (seedCount < minSeedCount) {
    return softMask;
  }

  const K = Math.max(1, Math.min(colorClusters, 5));

  const cL = new Float32Array(K);
  const cA = new Float32Array(K);
  const cB = new Float32Array(K);

  const first = seedIndices[((seedCount * 7919) % seedCount) | 0];
  cL[0] = labL[first];
  cA[0] = labA[first];
  cB[0] = labB[first];
  for (let k = 1; k < K; k++) {
    let bestIdx = -1,
      bestDist = -1;
    for (let s = 0; s < seedCount; s++) {
      const idx = seedIndices[s];
      let minD = Infinity;
      for (let kk = 0; kk < k; kk++) {
        const dL = labL[idx] - cL[kk],
          dA = labA[idx] - cA[kk],
          dB = labB[idx] - cB[kk];
        const d = dL * dL + dA * dA + dB * dB;
        if (d < minD) minD = d;
      }
      if (minD > bestDist) {
        bestDist = minD;
        bestIdx = idx;
      }
    }
    cL[k] = labL[bestIdx];
    cA[k] = labA[bestIdx];
    cB[k] = labB[bestIdx];
  }

  const assign = new Int8Array(seedCount);
  for (let iter = 0; iter < kmeansIterations; iter++) {
    for (let s = 0; s < seedCount; s++) {
      const idx = seedIndices[s];
      let best = 0,
        bestD = Infinity;
      for (let k = 0; k < K; k++) {
        const dL = labL[idx] - cL[k],
          dA = labA[idx] - cA[k],
          dB = labB[idx] - cB[k];
        const d = dL * dL + dA * dA + dB * dB;
        if (d < bestD) {
          bestD = d;
          best = k;
        }
      }
      assign[s] = best;
    }

    const newL = new Float32Array(K),
      newA = new Float32Array(K),
      newB = new Float32Array(K);
    const counts = new Int32Array(K);
    for (let s = 0; s < seedCount; s++) {
      const k = assign[s];
      const idx = seedIndices[s];
      newL[k] += labL[idx];
      newA[k] += labA[idx];
      newB[k] += labB[idx];
      counts[k]++;
    }
    for (let k = 0; k < K; k++) {
      if (counts[k] > 0) {
        cL[k] = newL[k] / counts[k];
        cA[k] = newA[k] / counts[k];
        cB[k] = newB[k] / counts[k];
      }
    }
  }

  const stdL = new Float32Array(K),
    stdA = new Float32Array(K),
    stdB = new Float32Array(K);
  const counts = new Int32Array(K);
  for (let s = 0; s < seedCount; s++) {
    const k = assign[s];
    const idx = seedIndices[s];
    const dL = labL[idx] - cL[k],
      dA = labA[idx] - cA[k],
      dB = labB[idx] - cB[k];
    stdL[k] += dL * dL;
    stdA[k] += dA * dA;
    stdB[k] += dB * dB;
    counts[k]++;
  }
  for (let k = 0; k < K; k++) {
    if (counts[k] > 0) {
      stdL[k] = Math.sqrt(stdL[k] / counts[k]) + 1e-3;
      stdA[k] = Math.sqrt(stdA[k] / counts[k]) + 1e-3;
      stdB[k] = Math.sqrt(stdB[k] / counts[k]) + 1e-3;
    } else {
      stdL[k] = stdA[k] = stdB[k] = 1e6;
    }
  }

  const distance = new Int16Array(N);
  distance.fill(-1);

  const queue = new Int32Array(N);
  let qHead = 0,
    qTail = 0;
  for (let i = 0; i < N; i++) {
    if (seeds[i]) {
      distance[i] = 0;
      queue[qTail++] = i;
    }
  }
  while (qHead < qTail) {
    const idx = queue[qHead++];
    const d = distance[idx];
    if (d >= spatialRadiusPx) continue;
    const x = idx % OUT;
    const y = (idx / OUT) | 0;

    if (x > 0 && distance[idx - 1] === -1) {
      distance[idx - 1] = d + 1;
      queue[qTail++] = idx - 1;
    }
    if (x < OUT - 1 && distance[idx + 1] === -1) {
      distance[idx + 1] = d + 1;
      queue[qTail++] = idx + 1;
    }
    if (y > 0 && distance[idx - OUT] === -1) {
      distance[idx - OUT] = d + 1;
      queue[qTail++] = idx - OUT;
    }
    if (y < OUT - 1 && distance[idx + OUT] === -1) {
      distance[idx + OUT] = d + 1;
      queue[qTail++] = idx + OUT;
    }
  }

  const enhanced = new Float32Array(softMask);
  let propagated = 0,
    maxBoostApplied = 0;
  for (let i = 0; i < N; i++) {
    if (seeds[i]) continue;
    const cur = softMask[i];
    if (cur < candidateMin || cur > candidateMax) continue;
    const dist = distance[i];
    if (dist < 0 || dist > spatialRadiusPx) continue;

    let bestColorZ = Infinity;
    for (let k = 0; k < K; k++) {
      const zL = (labL[i] - cL[k]) / stdL[k];
      const zA = (labA[i] - cA[k]) / stdA[k];
      const zB = (labB[i] - cB[k]) / stdB[k];
      const z = Math.sqrt(zL * zL + zA * zA + zB * zB);
      if (z < bestColorZ) bestColorZ = z;
    }
    if (bestColorZ > colorStdDev) continue;

    const colorScore = 1 - bestColorZ / colorStdDev;
    const spatialScore = 1 - dist / spatialRadiusPx;

    const boost = colorScore * spatialScore * maxBoost;
    if (boost > maxBoostApplied) maxBoostApplied = boost;
    const newConf = Math.min(1, cur + boost);
    if (newConf > enhanced[i]) {
      enhanced[i] = newConf;
      propagated++;
    }
  }

  const clusterSummary = [];
  for (let k = 0; k < K; k++) {
    clusterSummary.push(
      `${k}:LAB(${cL[k].toFixed(0)},${cA[k].toFixed(0)},${cB[k].toFixed(0)})×${counts[k]}`,
    );
  }
  return enhanced;
}

async function detectFence() {
  if (!originalImage) return;
  if (!modalReady) {
    updateStatus(
      "Simulator not ready yet — please wait a moment and try again.",
      "error",
    );
    return;
  }

  showLoading("Uploading & detecting...");
  updateStatus("Uploading & detecting...", "loading");
  detectBtn.disabled = true;
  cleanBtn.disabled = true;

  try {
    let output = await _uploadAndInfer(originalImage, "mild");

    let pass1MaxConf = 0;
    for (let i = 0; i < output.data.length; i++) {
      if (output.data[i] > pass1MaxConf) pass1MaxConf = output.data[i];
    }
    maskData = await _postprocess(output);
    if (_hasAnyFence(maskData)) {
    } else {
      maskData = await _postprocess(output, _MODERATE_POSTPROCESS_OPTS);
      if (_hasAnyFence(maskData)) {
      } else {
        maskData = await _postprocess(output, _RELAXED_POSTPROCESS_OPTS);
        if (_hasAnyFence(maskData)) {
        } else {
          showLoading("Trying harder...");
          output = await _uploadAndInfer(originalImage, "aggressive");

          let pass3MaxConf = 0;
          for (let i = 0; i < output.data.length; i++) {
            if (output.data[i] > pass3MaxConf) pass3MaxConf = output.data[i];
          }
          maskData = await _postprocess(output);
          if (_hasAnyFence(maskData)) {
          } else {
            maskData = await _postprocess(output, _MODERATE_POSTPROCESS_OPTS);
            if (_hasAnyFence(maskData)) {
            } else {
              maskData = await _postprocess(output, _RELAXED_POSTPROCESS_OPTS);
              if (_hasAnyFence(maskData)) {
              } else {
                showLoading("Tiled inference (this may take longer)...");
                const tiledOutput = await _tiledInferenceMultiScale(
                  originalImage,
                  "mild",
                );

                tiledOutput.data = _gateTilesByWholeImage(
                  tiledOutput.data,
                  output.data,
                  { pixelGateThreshold: 0.04, dilatePx: 120 },
                );

                tiledOutput.data = _maskCompleteByColorPropagation(
                  tiledOutput.data,
                  originalImage,
                  { spatialRadiusPx: 30 },
                );
                tiledOutput.data = _maskCompleteByColorPropagation(
                  tiledOutput.data,
                  originalImage,
                  { seedThreshold: 0.5, spatialRadiusPx: 30 },
                );

                const _TILED_POSTPROCESS_OPTS = {
                  softLow: 0.35,
                  softHigh: 0.6,
                  ccMinPct: 0.05,
                  skipSpatialRecovery: true,
                };
                maskData = await _postprocess(tiledOutput, _TILED_POSTPROCESS_OPTS);

                maskData = _filterMaskByShape(
                  maskData,
                  originalImage.width,
                  originalImage.height,
                );

                if (_hasAnyFence(maskData)) {
                  if (
                    !_checkMaskSurroundContext(
                      maskData,
                      originalImage,
                      originalImage.width,
                      originalImage.height,
                    )
                  ) {
                    maskData = new Float32Array(maskData.length);
                  }
                }

                if (_hasAnyFence(maskData)) {
                }
              }
            }
          }
        }
      }
    }

    if (!_hasAnyFence(maskData)) {
      drawMask(maskData);
      maskData = null;
      recolorBtn.disabled = true;
      updateStatus(
        "No fence detected in this photo. Please try a clearer photo of the fence.",
        "error",
      );
      return;
    }

    drawMask(maskData);

    updateStatus("Fence detected successfully!", "success", { silent: true });
    recolorBtn.disabled = false;
  } catch (error) {
    console.error("Detection error:", error);
    updateStatus("Detection failed: " + error.message, "error");
  } finally {
    hideLoading();
    detectBtn.disabled = false;
    cleanBtn.disabled = false;
  }
}

function getOriginalImagePixelData(img) {
  const c = document.createElement("canvas");
  c.width = img.naturalWidth || img.width;
  c.height = img.naturalHeight || img.height;
  const ctx = c.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0);
  return ctx.getImageData(0, 0, c.width, c.height).data;
}

// Bilinear-downscaled pixel buffer for post-process work. Same as
// getOriginalImagePixelData but targets an explicit (w, h) so we don't pay
// the 5000x4000 = 80 MB readback cost on huge source images.
function getScaledImagePixelData(img, w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

function filterBark(mask, pixelData, width, height, satGap, brightDelta) {
  const N = width * height;

  let sumChroma = 0,
    sumL = 0,
    count = 0;
  const chromaSamples = [];
  for (let i = 0; i < N; i++) {
    if (mask[i] > 0.4) {
      const p = i * 4;
      const lab = _cleanRgbToLab(
        pixelData[p],
        pixelData[p + 1],
        pixelData[p + 2],
      );
      const chroma = Math.sqrt(lab[1] * lab[1] + lab[2] * lab[2]);
      sumChroma += chroma;
      sumL += lab[0];
      chromaSamples.push(chroma);
      count++;
    }
  }
  if (count < 100) {
    return mask;
  }
  const meanChroma = sumChroma / count;
  const meanL = sumL / count;

  let chromaSumSq = 0;
  for (let i = 0; i < chromaSamples.length; i++) {
    const d = chromaSamples[i] - meanChroma;
    chromaSumSq += d * d;
  }
  const chromaStddev = Math.sqrt(chromaSumSq / count);

  const minGap = 6;
  const adaptiveGap = Math.max(minGap, 2.0 * chromaStddev);
  const chromaCutoff = meanChroma - adaptiveGap;

  const lDelta = 15;

  const confProtect = 0.55;
  const out = new Float32Array(N);
  let dropped = 0;
  let protectedByConf = 0;
  for (let i = 0; i < N; i++) {
    if (mask[i] > 0) {
      const p = i * 4;
      const lab = _cleanRgbToLab(
        pixelData[p],
        pixelData[p + 1],
        pixelData[p + 2],
      );
      const Lstar = lab[0];
      const chroma = Math.sqrt(lab[1] * lab[1] + lab[2] * lab[2]);
      const isBarkLike =
        chroma < chromaCutoff && Math.abs(Lstar - meanL) < lDelta;
      if (isBarkLike && mask[i] <= confProtect) {
        out[i] = 0;
        dropped++;
      } else {
        out[i] = mask[i];
        if (isBarkLike) protectedByConf++;
      }
    }
  }
  return out;
}

function filterBrick(mask, pixelData, width, height) {
  const N = width * height;

  let sumA = 0,
    sumB_ = 0,
    count = 0;
  for (let i = 0; i < N; i++) {
    if (mask[i] > 0.7) {
      const p = i * 4;
      const lab = _cleanRgbToLab(
        pixelData[p],
        pixelData[p + 1],
        pixelData[p + 2],
      );
      sumA += lab[1];
      sumB_ += lab[2];
      count++;
    }
  }
  if (count < 100) {
    return mask;
  }
  const meanA = sumA / count;
  const meanB_ = sumB_ / count;

  if (meanA > 10) {
    return mask;
  }

  if (meanB_ < 15) {
    return mask;
  }

  const aThr = meanA + 8;
  const bThr = meanB_ - 3;

  let wouldDrop = 0,
    totalMasked = 0;
  for (let i = 0; i < N; i++) {
    if (mask[i] > 0) {
      totalMasked++;
      const p = i * 4;
      const lab = _cleanRgbToLab(
        pixelData[p],
        pixelData[p + 1],
        pixelData[p + 2],
      );
      if (lab[1] > aThr && lab[2] < bThr) wouldDrop++;
    }
  }
  const dropFrac = wouldDrop / Math.max(1, totalMasked);
  if (dropFrac > 0.15) {
    return mask;
  }

  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    if (mask[i] > 0) {
      const p = i * 4;
      const lab = _cleanRgbToLab(
        pixelData[p],
        pixelData[p + 1],
        pixelData[p + 2],
      );
      if (lab[1] > aThr && lab[2] < bThr) {
        out[i] = 0;
      } else {
        out[i] = mask[i];
      }
    }
  }
  return out;
}

function filterTrunks(
  mask,
  pixelData,
  width,
  height,
  hardDist,
  softDist,
  desatDelta,
) {
  const N = width * height;

  let sumA = 0,
    sumB_ = 0,
    sumSat = 0,
    count = 0;
  const trunkSatSamples = [];
  for (let i = 0; i < N; i++) {
    if (mask[i] > 0.4) {
      const r = pixelData[i * 4];
      const g = pixelData[i * 4 + 1];
      const b = pixelData[i * 4 + 2];
      const lab = _cleanRgbToLab(r, g, b);
      sumA += lab[1];
      sumB_ += lab[2];
      const max = Math.max(r, g, b),
        min = Math.min(r, g, b);
      const sat = max > 0 ? (max - min) / max : 0;
      sumSat += sat;
      trunkSatSamples.push(sat);
      count++;
    }
  }
  if (count < 100) {
    return mask;
  }
  const meanA = sumA / count;
  const meanB_ = sumB_ / count;
  const meanSat = sumSat / count;

  let sumChromaDistSq = 0;
  for (let i = 0; i < N; i++) {
    if (mask[i] > 0.4) {
      const r = pixelData[i * 4];
      const g = pixelData[i * 4 + 1];
      const b = pixelData[i * 4 + 2];
      const lab = _cleanRgbToLab(r, g, b);
      const da = lab[1] - meanA,
        db = lab[2] - meanB_;
      sumChromaDistSq += da * da + db * db;
    }
  }
  const chromaStddev = Math.sqrt(sumChromaDistSq / count);

  const hardCapChroma = 28;
  const softCapChroma = 20;
  const effHardChroma = Math.min(
    hardCapChroma,
    Math.max(22, 3.0 * chromaStddev),
  );
  const effSoftChroma = Math.min(
    softCapChroma,
    Math.max(14, 2.0 * chromaStddev),
  );
  const hardSq = effHardChroma * effHardChroma;
  const softSq = effSoftChroma * effSoftChroma;

  const meanChromaMag = Math.sqrt(meanA * meanA + meanB_ * meanB_);
  const chromaMagGap = Math.max(12, 2.5 * chromaStddev);
  const hyperChromaThr = meanChromaMag + chromaMagGap;

  const out = new Float32Array(N);
  let dropped = 0,
    droppedHyper = 0;
  for (let i = 0; i < N; i++) {
    if (mask[i] > 0) {
      const r = pixelData[i * 4];
      const g = pixelData[i * 4 + 1];
      const b = pixelData[i * 4 + 2];
      const lab = _cleanRgbToLab(r, g, b);
      const aStar = lab[1],
        bStar = lab[2];
      const da = aStar - meanA,
        db = bStar - meanB_;
      const distSq = da * da + db * db;
      const pxChromaMag = Math.sqrt(aStar * aStar + bStar * bStar);
      let drop = false;
      if (distSq > hardSq) {
        drop = true;
      } else if (distSq > softSq) {
        const max = Math.max(r, g, b),
          min = Math.min(r, g, b);
        const sat = max > 0 ? (max - min) / max : 0;
        if (sat < meanSat - desatDelta) drop = true;
      }

      if (!drop && pxChromaMag > hyperChromaThr) {
        drop = true;
        droppedHyper++;
      }
      if (drop) {
        out[i] = 0;
        dropped++;
      } else {
        out[i] = mask[i];
      }
    }
  }
  return out;
}

function filterBuildings(
  mask,
  width,
  height,
  minMeanConf,
  confRatio,
  minCcPx,
  blockyFillRatio,
  blockyConfBoost,
) {
  const N = width * height;
  const ext = _maskExtent(mask, width, height);
  if (ext.empty) return mask;
  const labels = new Int32Array(N);
  const maxLabels = (N >> 1) + 2;
  const parent = new Int32Array(maxLabels);
  let nextLabel = 1;

  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const union = (a, b) => {
    const ra = find(a),
      rb = find(b);
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
  };

  // Bound the label-assignment pass to the mask bbox -- skips empty
  // rows/cols outside the fence region.
  for (let y = ext.minY; y <= ext.maxY; y++) {
    for (let x = ext.minX; x <= ext.maxX; x++) {
      const idx = y * width + x;
      if (mask[idx] <= 0) continue;
      const left = x > 0 ? labels[idx - 1] : 0;
      const up = y > 0 ? labels[idx - width] : 0;
      if (left === 0 && up === 0) {
        labels[idx] = nextLabel;
        parent[nextLabel] = nextLabel;
        nextLabel++;
      } else if (left !== 0 && up === 0) {
        labels[idx] = left;
      } else if (left === 0 && up !== 0) {
        labels[idx] = up;
      } else {
        labels[idx] = Math.min(left, up);
        if (left !== up) union(left, up);
      }
    }
  }

  const sums = new Map();
  const counts = new Map();
  const bboxes = new Map();
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (labels[idx] === 0) continue;
      const root = find(labels[idx]);
      sums.set(root, (sums.get(root) || 0) + mask[idx]);
      counts.set(root, (counts.get(root) || 0) + 1);
      const bb = bboxes.get(root);
      if (!bb) {
        bboxes.set(root, [x, x, y, y]);
      } else {
        if (x < bb[0]) bb[0] = x;
        if (x > bb[1]) bb[1] = x;
        if (y < bb[2]) bb[2] = y;
        if (y > bb[3]) bb[3] = y;
      }
    }
  }

  let bigCount = 0;
  for (const [, cnt] of counts) {
    if (cnt >= minCcPx) bigCount++;
    if (bigCount >= 2) break;
  }
  if (bigCount <= 1) {
    return mask;
  }

  let largestRoot = -1;
  let largestCnt = 0;
  for (const [root, cnt] of counts) {
    if (cnt < minCcPx) continue;
    if (cnt > largestCnt) {
      largestCnt = cnt;
      largestRoot = root;
    }
  }
  const fenceMean = largestRoot !== -1 ? sums.get(largestRoot) / largestCnt : 0;

  const threshold =
    fenceMean > 0 ? Math.max(minMeanConf, fenceMean * confRatio) : minMeanConf;

  const dropRoots = new Set();
  let blockyHits = 0;
  for (const [root, sum] of sums) {
    const cnt = counts.get(root);
    if (cnt < minCcPx) continue;
    const mean = sum / cnt;
    const bb = bboxes.get(root);
    const bbW = bb[1] - bb[0] + 1;
    const bbH = bb[3] - bb[2] + 1;
    const bboxArea = bbW * bbH;
    const fillRatio = cnt / bboxArea;

    const bbAspect = Math.max(bbW, bbH) / Math.max(1, Math.min(bbW, bbH));
    const isBlocky = fillRatio >= blockyFillRatio && bbAspect <= 4;
    const effThr = isBlocky ? threshold + blockyConfBoost : threshold;
    if (mean < effThr) {
      dropRoots.add(root);
      if (isBlocky) blockyHits++;
    }
  }

  if (dropRoots.size === 0) {
    return mask;
  }

  const out = new Float32Array(N);
  let droppedPx = 0;
  for (let i = 0; i < N; i++) {
    if (labels[i] === 0) {
      out[i] = mask[i];
      continue;
    }
    const root = find(labels[i]);
    if (dropRoots.has(root)) {
      droppedPx++;
    } else {
      out[i] = mask[i];
    }
  }
  return out;
}

function fillHoles(mask, width, height, maxHolePct, valueScale) {
  const N = width * height;
  const maxHoleArea = Math.max(1, Math.floor((N * maxHolePct) / 100));
  const labels = new Int32Array(N);
  const maxLabels = (N >> 1) + 2;
  const parent = new Int32Array(maxLabels);
  let nextLabel = 1;

  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const union = (a, b) => {
    const ra = find(a),
      rb = find(b);
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
  };

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (mask[idx] > 0) continue;
      const left = x > 0 ? labels[idx - 1] : 0;
      const up = y > 0 ? labels[idx - width] : 0;
      if (left === 0 && up === 0) {
        labels[idx] = nextLabel;
        parent[nextLabel] = nextLabel;
        nextLabel++;
      } else if (left !== 0 && up === 0) {
        labels[idx] = left;
      } else if (left === 0 && up !== 0) {
        labels[idx] = up;
      } else {
        labels[idx] = Math.min(left, up);
        if (left !== up) union(left, up);
      }
    }
  }

  const touchesBorder = new Set();
  for (let x = 0; x < width; x++) {
    const t = labels[x];
    const b = labels[(height - 1) * width + x];
    if (t !== 0) touchesBorder.add(find(t));
    if (b !== 0) touchesBorder.add(find(b));
  }
  for (let y = 0; y < height; y++) {
    const l = labels[y * width];
    const r = labels[y * width + (width - 1)];
    if (l !== 0) touchesBorder.add(find(l));
    if (r !== 0) touchesBorder.add(find(r));
  }

  const areas = new Map();
  for (let i = 0; i < N; i++) {
    if (labels[i] === 0) continue;
    const root = find(labels[i]);
    areas.set(root, (areas.get(root) || 0) + 1);
  }

  const fillRoots = new Set();
  for (const [root, area] of areas) {
    if (!touchesBorder.has(root) && area <= maxHoleArea) {
      fillRoots.add(root);
    }
  }

  if (fillRoots.size === 0) {
    return mask;
  }

  let sumMask = 0,
    countMask = 0;
  for (let i = 0; i < N; i++) {
    if (mask[i] > 0) {
      sumMask += mask[i];
      countMask++;
    }
  }
  const fenceMean = countMask > 0 ? sumMask / countMask : 1.0;
  const fillVal = fenceMean * valueScale;

  const out = new Float32Array(N);
  let filledPx = 0;
  for (let i = 0; i < N; i++) {
    if (mask[i] > 0) {
      out[i] = mask[i];
    } else if (labels[i] !== 0 && fillRoots.has(find(labels[i]))) {
      out[i] = fillVal;
      filledPx++;
    }
  }
  return out;
}

function filterSky(mask, pixelData, width, height, topFrac, minLum, maxSat) {
  const N = width * height;
  const topRowCutoff = Math.floor(height * topFrac);
  const out = new Float32Array(N);
  let dropped = 0;
  for (let i = 0; i < N; i++) {
    if (mask[i] <= 0) continue;
    const y = (i / width) | 0;
    if (y >= topRowCutoff) {
      out[i] = mask[i];
      continue;
    }
    const r = pixelData[i * 4];
    const g = pixelData[i * 4 + 1];
    const b = pixelData[i * 4 + 2];
    const lum = (r + g + b) / 3;
    const max = Math.max(r, g, b),
      min = Math.min(r, g, b);
    const sat = max > 0 ? (max - min) / max : 0;
    if (lum >= minLum && sat < maxSat) {
      dropped++;
    } else {
      out[i] = mask[i];
    }
  }
  return out;
}

// Sliding-window max via monotonic deque. O(W*H) total regardless of radius.
// Each pixel is enqueued and dequeued at most once per axis pass — the
// inner while-loops average O(1) per pixel. Output is bit-identical to the
// naive nested-loop version. Significant on the spatiallyGuidedRecovery
// and recoverAdjacentToSurvivors paths, where radius is 20-25 and the
// naive cost was O(W*H*r).
function dilateFast(mask, width, height, radius) {
  const N = width * height;
  const tmp = new Float32Array(N);
  /* Deque holds candidate indices whose values are monotonically
   * decreasing. head and tail are MONOTONIC absolute indices (we never
   * wrap), so the backing array must be sized to the max value tail
   * can reach in a single pass -- that's `width + radius` for the row
   * pass and `height + radius` for the column pass. Sizing it to
   * (2*radius+2) was the original bug: tail overflowed for any image
   * wider/taller than the deque, silently writing past TypedArray bounds
   * and yielding 0s thereafter. */
  const dqRow = new Int32Array(width + radius);
  const dqCol = new Int32Array(height + radius);

  // Row pass: for each row, run a 1D sliding-window max along x.
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let head = 0, tail = 0;
    for (let x = 0; x < width + radius; x++) {
      if (x < width) {
        const v = mask[row + x];
        while (head < tail && mask[row + dqRow[tail - 1]] <= v) tail--;
        dqRow[tail++] = x;
      }
      const winStart = x - radius;
      const winLeft = winStart - radius;
      while (head < tail && dqRow[head] < winLeft) head++;
      if (winStart >= 0) {
        tmp[row + winStart] = mask[row + dqRow[head]];
      }
    }
  }

  // Column pass: same algorithm but stride-by-width through tmp.
  const out = new Float32Array(N);
  for (let x = 0; x < width; x++) {
    let head = 0, tail = 0;
    for (let y = 0; y < height + radius; y++) {
      if (y < height) {
        const v = tmp[y * width + x];
        while (head < tail && tmp[dqCol[tail - 1] * width + x] <= v) tail--;
        dqCol[tail++] = y;
      }
      const winStart = y - radius;
      const winLeft = winStart - radius;
      while (head < tail && dqCol[head] < winLeft) head++;
      if (winStart >= 0) {
        out[winStart * width + x] = tmp[dqCol[head] * width + x];
      }
    }
  }
  return out;
}

async function spatiallyGuidedRecovery(
  mask,
  rawProbs,
  width,
  height,
  coreThr,
  fillThr,
  dilatePx,
) {
  const N = width * height;
  const core = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    core[i] = rawProbs[i] >= coreThr ? 1.0 : 0.0;
  }
  // dilateAccelerated chooses WebGPU when available, falls back to the
  // optimized JS dilateFast otherwise. Bit-identical output.
  const zone = await dilateAccelerated(core, width, height, dilatePx);
  const out = new Float32Array(N);
  let recovered = 0;
  for (let i = 0; i < N; i++) {
    if (mask[i] > 0) {
      out[i] = mask[i];
    } else if (zone[i] > 0 && rawProbs[i] >= fillThr) {
      out[i] = rawProbs[i];
      recovered++;
    }
  }
  return out;
}

async function recoverAdjacentToSurvivors(
  mask,
  rawProbs,
  width,
  height,
  coreThr,
  fillThr,
  dilatePx,
  pixelData,
) {
  const N = width * height;

  const core = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    core[i] = mask[i] >= coreThr ? 1.0 : 0.0;
  }
  const zone = await dilateAccelerated(core, width, height, dilatePx);

  let meanA = 0,
    meanB_ = 0,
    cnt = 0;
  if (pixelData) {
    for (let i = 0; i < N; i++) {
      if (mask[i] >= coreThr) {
        const p = i * 4;
        const lab = _cleanRgbToLab(
          pixelData[p],
          pixelData[p + 1],
          pixelData[p + 2],
        );
        meanA += lab[1];
        meanB_ += lab[2];
        cnt++;
      }
    }
    if (cnt > 0) {
      meanA /= cnt;
      meanB_ /= cnt;
    }
  }

  const recoverChromaMax = 10;
  const recoverChromaMaxSq = recoverChromaMax * recoverChromaMax;

  const out = new Float32Array(mask);
  let recovered = 0,
    blockedByChroma = 0;
  for (let i = 0; i < N; i++) {
    if (mask[i] === 0 && zone[i] > 0 && rawProbs[i] >= fillThr) {
      if (pixelData && cnt > 0) {
        const p = i * 4;
        const lab = _cleanRgbToLab(
          pixelData[p],
          pixelData[p + 1],
          pixelData[p + 2],
        );
        const da = lab[1] - meanA,
          db = lab[2] - meanB_;
        if (da * da + db * db > recoverChromaMaxSq) {
          blockedByChroma++;
          continue;
        }
      }
      out[i] = rawProbs[i];
      recovered++;
    }
  }
  return out;
}

function extendFenceDown(
  mask,
  pixelData,
  width,
  height,
  maxExtendPx,
  chromaMaxDist,
) {
  let meanA = 0,
    meanB_ = 0,
    cnt = 0;
  for (let i = 0; i < width * height; i++) {
    if (mask[i] >= 0.7 && pixelData) {
      const p = i * 4;
      const lab = _cleanRgbToLab(
        pixelData[p],
        pixelData[p + 1],
        pixelData[p + 2],
      );
      meanA += lab[1];
      meanB_ += lab[2];
      cnt++;
    }
  }
  if (cnt > 0) {
    meanA /= cnt;
    meanB_ /= cnt;
  }
  const chromaMaxSq = chromaMaxDist * chromaMaxDist;

  const out = new Float32Array(mask);
  let extended = 0;
  let columnsExtended = 0;

  for (let x = 0; x < width; x++) {
    let lowestY = -1;
    for (let y = height - 1; y >= 0; y--) {
      if (mask[y * width + x] >= 0.7) {
        lowestY = y;
        break;
      }
    }
    if (lowestY === -1) continue;
    if (lowestY >= height - 1) continue;

    let addedThisCol = 0;
    for (let dy = 1; dy <= maxExtendPx; dy++) {
      const y = lowestY + dy;
      if (y >= height) break;
      const idx = y * width + x;
      if (mask[idx] > 0) continue;

      if (pixelData && cnt > 0) {
        const p = idx * 4;
        const lab = _cleanRgbToLab(
          pixelData[p],
          pixelData[p + 1],
          pixelData[p + 2],
        );
        const da = lab[1] - meanA,
          db = lab[2] - meanB_;
        if (da * da + db * db > chromaMaxSq) {
          break;
        }
      }

      out[idx] = 0.7;
      extended++;
      addedThisCol++;
    }
    if (addedThisCol > 0) columnsExtended++;
  }
  return out;
}

function filterCCColorOutliers(
  mask,
  pixelData,
  width,
  height,
  kStddev,
  minDist,
  minCcPx,
) {
  const N = width * height;
  const ext = _maskExtent(mask, width, height);
  if (ext.empty) return mask;
  const labels = new Int32Array(N);
  const maxLabels = (N >> 1) + 2;
  const parent = new Int32Array(maxLabels);
  let nextLabel = 1;

  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const union = (a, b) => {
    const ra = find(a),
      rb = find(b);
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
  };

  for (let y = ext.minY; y <= ext.maxY; y++) {
    for (let x = ext.minX; x <= ext.maxX; x++) {
      const idx = y * width + x;
      if (mask[idx] <= 0) continue;
      const left = x > 0 ? labels[idx - 1] : 0;
      const up = y > 0 ? labels[idx - width] : 0;
      if (left === 0 && up === 0) {
        labels[idx] = nextLabel;
        parent[nextLabel] = nextLabel;
        nextLabel++;
      } else if (left !== 0 && up === 0) {
        labels[idx] = left;
      } else if (left === 0 && up !== 0) {
        labels[idx] = up;
      } else {
        labels[idx] = Math.min(left, up);
        if (left !== up) union(left, up);
      }
    }
  }

  const labA = new Float32Array(N);
  const labB = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    if (labels[i] === 0) continue;
    const p = i * 4;
    const lab = _cleanRgbToLab(
      pixelData[p],
      pixelData[p + 1],
      pixelData[p + 2],
    );
    labA[i] = lab[1];
    labB[i] = lab[2];
  }

  const sumA = new Map(),
    sumB_ = new Map(),
    cnt = new Map();
  for (let i = 0; i < N; i++) {
    if (labels[i] === 0) continue;
    const root = find(labels[i]);
    sumA.set(root, (sumA.get(root) || 0) + labA[i]);
    sumB_.set(root, (sumB_.get(root) || 0) + labB[i]);
    cnt.set(root, (cnt.get(root) || 0) + 1);
  }

  const meanA = new Map(),
    meanB_ = new Map();
  for (const [root, c] of cnt) {
    if (c < minCcPx) continue;
    meanA.set(root, sumA.get(root) / c);
    meanB_.set(root, sumB_.get(root) / c);
  }

  const sumDistSq = new Map();
  for (let i = 0; i < N; i++) {
    if (labels[i] === 0) continue;
    const root = find(labels[i]);
    if (!meanA.has(root)) continue;
    const dA = labA[i] - meanA.get(root);
    const dB = labB[i] - meanB_.get(root);
    sumDistSq.set(root, (sumDistSq.get(root) || 0) + (dA * dA + dB * dB));
  }

  const minDistChroma = Math.max(8, minDist / 8);
  const ccDropDistSq = new Map();
  for (const [root, sumSq] of sumDistSq) {
    const c = cnt.get(root);
    const stddev = Math.sqrt(sumSq / c);
    const adaptive = kStddev * stddev;
    const eff = Math.max(minDistChroma, adaptive);
    ccDropDistSq.set(root, eff * eff);
  }

  const out = new Float32Array(N);
  let dropped = 0;
  for (let i = 0; i < N; i++) {
    if (labels[i] === 0) continue;
    const root = find(labels[i]);
    if (!ccDropDistSq.has(root)) {
      out[i] = mask[i];
      continue;
    }
    const dA = labA[i] - meanA.get(root);
    const dB = labB[i] - meanB_.get(root);
    if (dA * dA + dB * dB > ccDropDistSq.get(root)) {
      dropped++;
    } else {
      out[i] = mask[i];
    }
  }
  const thrSummary = [...ccDropDistSq.entries()]
    .map(([r, dsq]) => `${r}:${Math.sqrt(dsq).toFixed(1)}`)
    .slice(0, 5)
    .join(" ");
  return out;
}

function filterByOrientation(mask, width, height, ratio, minCount) {
  const N = width * height;
  const rowExtent = new Int32Array(N);
  const colExtent = new Int32Array(N);

  for (let y = 0; y < height; y++) {
    const base = y * width;
    let left = 0;
    for (let x = 0; x < width; x++) {
      left = mask[base + x] > 0 ? left + 1 : 0;
      rowExtent[base + x] = left;
    }
    let right = 0;
    for (let x = width - 1; x >= 0; x--) {
      right = mask[base + x] > 0 ? right + 1 : 0;

      if (mask[base + x] > 0) {
        rowExtent[base + x] = rowExtent[base + x] + right - 1;
      } else {
        rowExtent[base + x] = 0;
      }
    }
  }

  for (let x = 0; x < width; x++) {
    let up = 0;
    for (let y = 0; y < height; y++) {
      const i = y * width + x;
      up = mask[i] > 0 ? up + 1 : 0;
      colExtent[i] = up;
    }
    let down = 0;
    for (let y = height - 1; y >= 0; y--) {
      const i = y * width + x;
      down = mask[i] > 0 ? down + 1 : 0;
      if (mask[i] > 0) {
        colExtent[i] = colExtent[i] + down - 1;
      } else {
        colExtent[i] = 0;
      }
    }
  }

  let horizCount = 0,
    vertCount = 0;
  for (let i = 0; i < N; i++) {
    if (mask[i] <= 0) continue;
    const r = rowExtent[i],
      c = colExtent[i];
    if (r > ratio * c) horizCount++;
    else if (c > ratio * r) vertCount++;
  }
  if (horizCount + vertCount < minCount) {
    return mask;
  }

  const majorCount = Math.max(horizCount, vertCount);
  const minorCount = Math.min(horizCount, vertCount);
  if (minorCount * 3 > majorCount) {
    return mask;
  }
  const dominantHorizontal = horizCount >= vertCount;

  const out = new Float32Array(N);
  let dropped = 0;
  for (let i = 0; i < N; i++) {
    if (mask[i] <= 0) continue;
    const r = rowExtent[i],
      c = colExtent[i];
    const isHoriz = r > ratio * c;
    const isVert = c > ratio * r;
    if (dominantHorizontal && isVert) {
      dropped++;
      continue;
    }
    if (!dominantHorizontal && isHoriz) {
      dropped++;
      continue;
    }
    out[i] = mask[i];
  }
  return out;
}

function filterByCCPrincipalAxis(
  mask,
  width,
  height,
  angleToleranceDeg,
  minCcPx,
  minAspect,
) {
  const N = width * height;
  const ext = _maskExtent(mask, width, height);
  if (ext.empty) return mask;
  const labels = new Int32Array(N);
  const maxLabels = (N >> 1) + 2;
  const parent = new Int32Array(maxLabels);
  let nextLabel = 1;

  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const union = (a, b) => {
    const ra = find(a),
      rb = find(b);
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
  };

  for (let y = ext.minY; y <= ext.maxY; y++) {
    for (let x = ext.minX; x <= ext.maxX; x++) {
      const idx = y * width + x;
      if (mask[idx] <= 0) continue;
      const left = x > 0 ? labels[idx - 1] : 0;
      const up = y > 0 ? labels[idx - width] : 0;
      if (left === 0 && up === 0) {
        labels[idx] = nextLabel;
        parent[nextLabel] = nextLabel;
        nextLabel++;
      } else if (left !== 0 && up === 0) {
        labels[idx] = left;
      } else if (left === 0 && up !== 0) {
        labels[idx] = up;
      } else {
        labels[idx] = Math.min(left, up);
        if (left !== up) union(left, up);
      }
    }
  }

  const moments = new Map();
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (labels[idx] === 0) continue;
      const root = find(labels[idx]);
      let m = moments.get(root);
      if (!m) {
        m = { sx: 0, sy: 0, sxx: 0, syy: 0, sxy: 0, n: 0 };
        moments.set(root, m);
      }
      m.sx += x;
      m.sy += y;
      m.sxx += x * x;
      m.syy += y * y;
      m.sxy += x * y;
      m.n++;
    }
  }

  const ccProps = new Map();
  for (const [root, m] of moments) {
    if (m.n < minCcPx) continue;
    const cx = m.sx / m.n;
    const cy = m.sy / m.n;
    const Cxx = m.sxx / m.n - cx * cx;
    const Cyy = m.syy / m.n - cy * cy;
    const Cxy = m.sxy / m.n - cx * cy;
    const halfDiff = (Cxx - Cyy) / 2;
    const discr = Math.sqrt(halfDiff * halfDiff + Cxy * Cxy);
    const trace = Cxx + Cyy;
    const lambdaMax = trace / 2 + discr;
    const lambdaMin = trace / 2 - discr;
    const aspect = lambdaMin > 1e-6 ? lambdaMax / lambdaMin : Infinity;

    const angle = 0.5 * Math.atan2(2 * Cxy, Cxx - Cyy);
    ccProps.set(root, { angle, aspect, n: m.n });
  }

  if (ccProps.size < 2) {
    return mask;
  }

  let refRoot = null;
  let refN = 0;
  for (const [root, p] of ccProps) {
    if (p.aspect < minAspect) continue;
    if (p.n > refN) {
      refN = p.n;
      refRoot = root;
    }
  }
  if (refRoot === null) {
    return mask;
  }
  const refAngle = ccProps.get(refRoot).angle;
  const tolRad = (angleToleranceDeg * Math.PI) / 180;

  const angleDiff = (a, b) => {
    let d = Math.abs(a - b);
    if (d > Math.PI / 2) d = Math.PI - d;
    return d;
  };

  const dropRoots = new Set();
  for (const [root, p] of ccProps) {
    if (root === refRoot) continue;
    if (p.aspect < minAspect) continue;
    if (angleDiff(p.angle, refAngle) > tolRad) {
      dropRoots.add(root);
    }
  }

  if (dropRoots.size === 0) {
    return mask;
  }

  const out = new Float32Array(N);
  let droppedPx = 0;
  for (let i = 0; i < N; i++) {
    if (labels[i] === 0) {
      out[i] = mask[i];
      continue;
    }
    const root = find(labels[i]);
    if (dropRoots.has(root)) {
      droppedPx++;
    } else {
      out[i] = mask[i];
    }
  }
  return out;
}

function filterBuildingWalls(
  mask,
  pixelData,
  width,
  height,
  minCcPx,
  satGapVsLargest,
  minLargestSat,
  maxSatStddev,
) {
  const N = width * height;
  const ext = _maskExtent(mask, width, height);
  if (ext.empty) return mask;
  const labels = new Int32Array(N);
  const maxLabels = (N >> 1) + 2;
  const parent = new Int32Array(maxLabels);
  let nextLabel = 1;

  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const union = (a, b) => {
    const ra = find(a),
      rb = find(b);
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
  };

  for (let y = ext.minY; y <= ext.maxY; y++) {
    for (let x = ext.minX; x <= ext.maxX; x++) {
      const idx = y * width + x;
      if (mask[idx] <= 0) continue;
      const left = x > 0 ? labels[idx - 1] : 0;
      const up = y > 0 ? labels[idx - width] : 0;
      if (left === 0 && up === 0) {
        labels[idx] = nextLabel;
        parent[nextLabel] = nextLabel;
        nextLabel++;
      } else if (left !== 0 && up === 0) {
        labels[idx] = left;
      } else if (left === 0 && up !== 0) {
        labels[idx] = up;
      } else {
        labels[idx] = Math.min(left, up);
        if (left !== up) union(left, up);
      }
    }
  }

  const sumSat = new Map();
  const cnt = new Map();
  let totalSat = 0,
    totalSatSq = 0,
    totalCnt = 0;
  for (let i = 0; i < N; i++) {
    if (labels[i] === 0) continue;
    const root = find(labels[i]);
    const p = i * 4;
    const r = pixelData[p],
      g = pixelData[p + 1],
      b = pixelData[p + 2];
    const max = Math.max(r, g, b),
      min = Math.min(r, g, b);
    const sat = max > 0 ? (max - min) / max : 0;
    sumSat.set(root, (sumSat.get(root) || 0) + sat);
    cnt.set(root, (cnt.get(root) || 0) + 1);
    totalSat += sat;
    totalSatSq += sat * sat;
    totalCnt++;
  }

  let largestRoot = -1;
  let largestCnt = 0;
  for (const [root, c] of cnt) {
    if (c < minCcPx) continue;
    if (c > largestCnt) {
      largestCnt = c;
      largestRoot = root;
    }
  }
  if (largestRoot === -1) {
    return mask;
  }
  const largestSat = sumSat.get(largestRoot) / largestCnt;

  if (largestSat < minLargestSat) {
    return mask;
  }

  if (totalCnt > 0 && maxSatStddev != null) {
    const meanSatAll = totalSat / totalCnt;
    const varSatAll = totalSatSq / totalCnt - meanSatAll * meanSatAll;
    const stdSatAll = Math.sqrt(Math.max(0, varSatAll));
    if (stdSatAll > maxSatStddev) {
      return mask;
    }
  }

  const satCutoff = largestSat - satGapVsLargest;
  const dropRoots = new Set();
  const dropInfo = [];
  for (const [root, c] of cnt) {
    if (root === largestRoot) continue;
    if (c < minCcPx) continue;
    const ccSat = sumSat.get(root) / c;
    if (ccSat < satCutoff) {
      dropRoots.add(root);
      dropInfo.push(`${root}(${c}px, sat ${ccSat.toFixed(2)})`);
    }
  }

  if (dropRoots.size === 0) {
    return mask;
  }

  const out = new Float32Array(N);
  let droppedPx = 0;
  for (let i = 0; i < N; i++) {
    if (labels[i] === 0) continue;
    const root = find(labels[i]);
    if (dropRoots.has(root)) {
      droppedPx++;
    } else {
      out[i] = mask[i];
    }
  }
  return out;
}

function filterSmallNonElongatedBlobs(
  mask,
  width,
  height,
  maxAreaPct,
  minAspect,
) {
  const N = width * height;
  const ext = _maskExtent(mask, width, height);
  if (ext.empty) return mask;
  const maxArea = Math.floor((N * maxAreaPct) / 100);
  const labels = new Int32Array(N);
  const maxLabels = (N >> 1) + 2;
  const parent = new Int32Array(maxLabels);
  let nextLabel = 1;

  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const union = (a, b) => {
    const ra = find(a),
      rb = find(b);
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
  };

  for (let y = ext.minY; y <= ext.maxY; y++) {
    for (let x = ext.minX; x <= ext.maxX; x++) {
      const idx = y * width + x;
      if (mask[idx] <= 0) continue;
      const left = x > 0 ? labels[idx - 1] : 0;
      const up = y > 0 ? labels[idx - width] : 0;
      if (left === 0 && up === 0) {
        labels[idx] = nextLabel;
        parent[nextLabel] = nextLabel;
        nextLabel++;
      } else if (left !== 0 && up === 0) {
        labels[idx] = left;
      } else if (left === 0 && up !== 0) {
        labels[idx] = up;
      } else {
        labels[idx] = Math.min(left, up);
        if (left !== up) union(left, up);
      }
    }
  }

  const moments = new Map();
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (labels[idx] === 0) continue;
      const root = find(labels[idx]);
      let m = moments.get(root);
      if (!m) {
        m = { sx: 0, sy: 0, sxx: 0, syy: 0, sxy: 0, n: 0 };
        moments.set(root, m);
      }
      m.sx += x;
      m.sy += y;
      m.sxx += x * x;
      m.syy += y * y;
      m.sxy += x * y;
      m.n++;
    }
  }

  const dropRoots = new Set();
  for (const [root, m] of moments) {
    if (m.n > maxArea) continue;
    if (m.n < 50) {
      dropRoots.add(root);
      continue;
    }
    const cx = m.sx / m.n;
    const cy = m.sy / m.n;
    const Cxx = m.sxx / m.n - cx * cx;
    const Cyy = m.syy / m.n - cy * cy;
    const Cxy = m.sxy / m.n - cx * cy;
    const halfDiff = (Cxx - Cyy) / 2;
    const discr = Math.sqrt(halfDiff * halfDiff + Cxy * Cxy);
    const trace = Cxx + Cyy;
    const lambdaMax = trace / 2 + discr;
    const lambdaMin = trace / 2 - discr;
    const aspect = lambdaMin > 1e-6 ? lambdaMax / lambdaMin : Infinity;
    if (aspect < minAspect) {
      dropRoots.add(root);
    }
  }

  if (dropRoots.size === 0) {
    return mask;
  }

  const out = new Float32Array(N);
  let droppedPx = 0;
  for (let i = 0; i < N; i++) {
    if (labels[i] === 0) {
      out[i] = mask[i];
      continue;
    }
    const root = find(labels[i]);
    if (dropRoots.has(root)) {
      droppedPx++;
    } else {
      out[i] = mask[i];
    }
  }
  return out;
}

function filterVegetation(mask, pixelData, width, height, threshold) {
  const N = width * height;
  const out = new Float32Array(N);
  let dropped = 0;
  for (let i = 0; i < N; i++) {
    if (mask[i] > 0) {
      const r = pixelData[i * 4];
      const g = pixelData[i * 4 + 1];
      const b = pixelData[i * 4 + 2];

      if (g - Math.max(r, b) > threshold) {
        out[i] = 0;
        dropped++;
      } else {
        out[i] = mask[i];
      }
    }
  }
  return out;
}

async function postprocessMask(
  output,
  width,
  height,
  smoothingKernel = 1,
  thresholdValue = 0.5,
  origPixelData = null,
  opts = {},
) {
  const probData = output.data;
  const ms = CONFIG.INPUT_SIZE;

  let maskAtModelRes = probData;
  if (smoothingKernel > 1) {
    maskAtModelRes = morphologicalClose(probData, ms, ms, smoothingKernel);
  }

  const maskOrig = bilinearResize(maskAtModelRes, ms, ms, width, height);

  const useSoftMask = CONFIG.USE_SOFT_MASK !== false;
  const softLow =
    opts.softLow != null
      ? opts.softLow
      : CONFIG.SOFT_MASK_LOW != null
        ? CONFIG.SOFT_MASK_LOW
        : 0.3;
  const softHigh =
    opts.softHigh != null
      ? opts.softHigh
      : CONFIG.SOFT_MASK_HIGH != null
        ? CONFIG.SOFT_MASK_HIGH
        : 0.7;
  let mask;
  if (useSoftMask) {
    // WASM SIMD path: ~4x faster on this hot loop (1.9M pixels at 1536
    // cap). Output is bit-identical to the JS branch below.
    if (_WASM.status === "ready") {
      try {
        mask = _wasmSoftMaskThreshold(maskOrig, softLow, softHigh);
      } catch (e) {
        console.warn("[fsv] WASM soft mask threshold failed, fallback:", e);
        _WASM.status = "unavailable";
        mask = null;
      }
    }
    if (!mask) {
      mask = new Float32Array(width * height);
      const range = Math.max(1e-6, softHigh - softLow);
      for (let i = 0; i < maskOrig.length; i++) {
        const p = maskOrig[i];
        if (p <= softLow) {
          mask[i] = 0.0;
        } else if (p >= softHigh) {
          mask[i] = p;
        } else {
          const t = (p - softLow) / range;
          mask[i] = p * t;
        }
      }
    }
  } else {
    mask = new Float32Array(width * height);
    for (let i = 0; i < maskOrig.length; i++) {
      mask[i] = maskOrig[i] >= thresholdValue ? 1.0 : 0.0;
    }
  }

  if (CONFIG.FILTER_SPATIAL_RECOVERY && !opts.skipSpatialRecovery) {
    const coreThr =
      CONFIG.RECOVERY_CORE_THRESHOLD != null
        ? CONFIG.RECOVERY_CORE_THRESHOLD
        : 0.85;
    const fillThr =
      CONFIG.RECOVERY_FILL_THRESHOLD != null
        ? CONFIG.RECOVERY_FILL_THRESHOLD
        : 0.45;
    const dilatePx =
      opts.recoveryDilatePx != null
        ? opts.recoveryDilatePx
        : CONFIG.RECOVERY_DILATE_PX != null
          ? CONFIG.RECOVERY_DILATE_PX
          : 35;
    mask = await spatiallyGuidedRecovery(
      mask,
      maskOrig,
      width,
      height,
      coreThr,
      fillThr,
      dilatePx,
    );
  }

  const erodePx =
    opts.erodePx != null ? opts.erodePx : CONFIG.POST_ERODE_PX || 0;
  const ccMinPct =
    opts.ccMinPct != null ? opts.ccMinPct : CONFIG.CC_MIN_BLOB_AREA_PCT || 0;
  const ccTopK = CONFIG.CC_KEEP_TOP_K_BLOBS || 0;
  if (erodePx > 0 || ccMinPct > 0 || ccTopK > 0) {
    let keep = new Float32Array(width * height);
    for (let i = 0; i < mask.length; i++) {
      keep[i] = mask[i] > 0 ? 1.0 : 0.0;
    }
    if (erodePx > 0) {
      keep = erode(keep, width, height, 2 * erodePx + 1);
    }
    if (ccMinPct > 0 || ccTopK > 0) {
      const minArea = Math.max(
        1,
        Math.floor((width * height * ccMinPct) / 100.0),
      );
      keep = connectedComponentClean(keep, width, height, minArea, ccTopK);
    }

    for (let i = 0; i < mask.length; i++) {
      if (keep[i] < 0.5) mask[i] = 0.0;
    }
  }

  if (CONFIG.FILTER_VEGETATION && origPixelData && !opts.skipVeg) {
    const greenThr =
      CONFIG.VEGETATION_GREEN_DOMINANCE != null
        ? CONFIG.VEGETATION_GREEN_DOMINANCE
        : 25;
    mask = filterVegetation(mask, origPixelData, width, height, greenThr);
  }

  if (CONFIG.FILTER_SKY && origPixelData && !opts.skipSky) {
    const topFrac =
      CONFIG.SKY_TOP_FRACTION != null ? CONFIG.SKY_TOP_FRACTION : 0.4;
    const minLum =
      CONFIG.SKY_MIN_LUMINANCE != null ? CONFIG.SKY_MIN_LUMINANCE : 175;
    const maxSat =
      CONFIG.SKY_MAX_SATURATION != null ? CONFIG.SKY_MAX_SATURATION : 0.1;
    mask = filterSky(
      mask,
      origPixelData,
      width,
      height,
      topFrac,
      minLum,
      maxSat,
    );
  }

  if (CONFIG.FILTER_BARK && origPixelData && !opts.skipBark) {
    const satGap = CONFIG.BARK_SAT_GAP != null ? CONFIG.BARK_SAT_GAP : 0.1;
    const brightDelta =
      CONFIG.BARK_BRIGHT_DELTA != null ? CONFIG.BARK_BRIGHT_DELTA : 40;
    mask = filterBark(mask, origPixelData, width, height, satGap, brightDelta);
  }

  if (CONFIG.FILTER_TRUNKS && origPixelData && !opts.skipTrunk) {
    const hardDist =
      CONFIG.TRUNK_COLOR_DIST_HARD != null ? CONFIG.TRUNK_COLOR_DIST_HARD : 90;
    const softDist =
      CONFIG.TRUNK_COLOR_DIST_SOFT != null ? CONFIG.TRUNK_COLOR_DIST_SOFT : 55;
    const desatDelta =
      CONFIG.TRUNK_DESAT_DELTA != null ? CONFIG.TRUNK_DESAT_DELTA : 0.1;
    mask = filterTrunks(
      mask,
      origPixelData,
      width,
      height,
      hardDist,
      softDist,
      desatDelta,
    );
  }

  if (origPixelData && !opts.skipBrick) {
    mask = filterBrick(mask, origPixelData, width, height);
  }

  if (
    CONFIG.FILTER_CC_COLOR_OUTLIERS &&
    origPixelData &&
    !opts.skipCcOutliers
  ) {
    const kStd =
      CONFIG.CC_OUTLIER_K_STDDEV != null ? CONFIG.CC_OUTLIER_K_STDDEV : 2.8;
    const minDst =
      CONFIG.CC_OUTLIER_MIN_DIST != null ? CONFIG.CC_OUTLIER_MIN_DIST : 80;
    const minPx =
      opts.ccOutlierMinPx != null
        ? opts.ccOutlierMinPx
        : CONFIG.CC_OUTLIER_MIN_PX != null
          ? CONFIG.CC_OUTLIER_MIN_PX
          : 300;
    mask = filterCCColorOutliers(
      mask,
      origPixelData,
      width,
      height,
      kStd,
      minDst,
      minPx,
    );
  }

  if (CONFIG.FILTER_SPATIAL_RECOVERY && !opts.skipSpatialRecovery && maskOrig) {
    const coreThr2 =
      CONFIG.RECOVERY2_CORE_THRESHOLD != null
        ? CONFIG.RECOVERY2_CORE_THRESHOLD
        : 0.7;

    const fillThr2 =
      CONFIG.RECOVERY2_FILL_THRESHOLD != null
        ? CONFIG.RECOVERY2_FILL_THRESHOLD
        : 0.25;
    const dilatePx2 =
      opts.recovery2DilatePx != null
        ? opts.recovery2DilatePx
        : CONFIG.RECOVERY2_DILATE_PX != null
          ? CONFIG.RECOVERY2_DILATE_PX
          : 25;
    mask = await recoverAdjacentToSurvivors(
      mask,
      maskOrig,
      width,
      height,
      coreThr2,
      fillThr2,
      dilatePx2,
      origPixelData,
    );
  }

  if (CONFIG.FILTER_ORIENTATION && !opts.skipOrientation) {
    const ratio =
      CONFIG.ORIENTATION_RATIO != null ? CONFIG.ORIENTATION_RATIO : 1.6;
    const minCt =
      opts.orientationMinCount != null
        ? opts.orientationMinCount
        : CONFIG.ORIENTATION_MIN_COUNT != null
          ? CONFIG.ORIENTATION_MIN_COUNT
          : 500;
    mask = filterByOrientation(mask, width, height, ratio, minCt);
  }

  if (CONFIG.FILTER_CC_PRINCIPAL_AXIS && !opts.skipCcAxis) {
    const tolDeg =
      CONFIG.CC_AXIS_ANGLE_TOLERANCE_DEG != null
        ? CONFIG.CC_AXIS_ANGLE_TOLERANCE_DEG
        : 30;
    const minPx =
      opts.ccAxisMinPx != null
        ? opts.ccAxisMinPx
        : CONFIG.CC_AXIS_MIN_PX != null
          ? CONFIG.CC_AXIS_MIN_PX
          : 500;
    const minAspect =
      CONFIG.CC_AXIS_MIN_ASPECT_RATIO != null
        ? CONFIG.CC_AXIS_MIN_ASPECT_RATIO
        : 2.0;
    mask = filterByCCPrincipalAxis(
      mask,
      width,
      height,
      tolDeg,
      minPx,
      minAspect,
    );
  }

  if (CONFIG.FILTER_JUNK_BLOBS && !opts.skipJunkBlobs) {
    const maxPct =
      CONFIG.JUNK_BLOB_MAX_AREA_PCT != null
        ? CONFIG.JUNK_BLOB_MAX_AREA_PCT
        : 0.6;
    const minAspect =
      CONFIG.JUNK_BLOB_MIN_ASPECT != null ? CONFIG.JUNK_BLOB_MIN_ASPECT : 1.8;
    mask = filterSmallNonElongatedBlobs(mask, width, height, maxPct, minAspect);
  }

  if (CONFIG.FILTER_BUILDINGS && !opts.skipBuilding) {
    const minMean =
      CONFIG.BUILDING_MIN_MEAN_CONF != null
        ? CONFIG.BUILDING_MIN_MEAN_CONF
        : 0.05;
    const confRatio =
      CONFIG.BUILDING_CONF_RATIO != null ? CONFIG.BUILDING_CONF_RATIO : 0.6;
    const minPx =
      opts.buildingMinCcPx != null
        ? opts.buildingMinCcPx
        : CONFIG.BUILDING_MIN_CC_PX != null
          ? CONFIG.BUILDING_MIN_CC_PX
          : 500;
    const blockyRatio =
      CONFIG.BUILDING_BLOCKY_FILL_RATIO != null
        ? CONFIG.BUILDING_BLOCKY_FILL_RATIO
        : 0.72;
    const blockyBoost =
      CONFIG.BUILDING_BLOCKY_CONF_BOOST != null
        ? CONFIG.BUILDING_BLOCKY_CONF_BOOST
        : 0.12;
    mask = filterBuildings(
      mask,
      width,
      height,
      minMean,
      confRatio,
      minPx,
      blockyRatio,
      blockyBoost,
    );
  }

  if (
    CONFIG.FILTER_BUILDING_WALLS &&
    origPixelData &&
    !opts.skipBuildingWalls
  ) {
    const minPx =
      opts.buildingWallMinCcPx != null
        ? opts.buildingWallMinCcPx
        : CONFIG.BUILDING_WALL_MIN_CC_PX != null
          ? CONFIG.BUILDING_WALL_MIN_CC_PX
          : 1000;
    const satGap =
      CONFIG.BUILDING_WALL_SAT_GAP != null ? CONFIG.BUILDING_WALL_SAT_GAP : 0.2;
    const minLgSat =
      CONFIG.BUILDING_WALL_MIN_LARGEST_SAT != null
        ? CONFIG.BUILDING_WALL_MIN_LARGEST_SAT
        : 0.25;
    const maxSatStd =
      CONFIG.BUILDING_WALL_MAX_SAT_STDDEV != null
        ? CONFIG.BUILDING_WALL_MAX_SAT_STDDEV
        : 0.12;
    mask = filterBuildingWalls(
      mask,
      origPixelData,
      width,
      height,
      minPx,
      satGap,
      minLgSat,
      maxSatStd,
    );
  }

  if (CONFIG.FILL_HOLES && !opts.skipHoleFill) {
    const maxPct =
      CONFIG.HOLE_FILL_MAX_PCT != null ? CONFIG.HOLE_FILL_MAX_PCT : 0.3;
    const valScl =
      CONFIG.HOLE_FILL_VALUE_SCALE != null
        ? CONFIG.HOLE_FILL_VALUE_SCALE
        : 0.95;
    mask = fillHoles(mask, width, height, maxPct, valScl);
  }

  if (origPixelData && !opts.skipBottomExtend) {
    const maxExtPx =
      opts.bottomExtendMaxPx != null
        ? opts.bottomExtendMaxPx
        : CONFIG.BOTTOM_EXTEND_MAX_PX != null
          ? CONFIG.BOTTOM_EXTEND_MAX_PX
          : 40;

    const chromaMax =
      CONFIG.BOTTOM_EXTEND_CHROMA_MAX != null
        ? CONFIG.BOTTOM_EXTEND_CHROMA_MAX
        : 10;
    mask = extendFenceDown(
      mask,
      origPixelData,
      width,
      height,
      maxExtPx,
      chromaMax,
    );
  }

  return mask;
}

function connectedComponentClean(mask, width, height, minArea, keepTopK) {
  const N = width * height;
  // connectedComponentClean is called with a binary keep mask (values >= 0.5
  // treated as foreground). Use 0.5 as the threshold for extent detection
  // so the bbox matches what the labeler considers foreground.
  let minY = height, maxY = -1, minX = width, maxX = -1;
  for (let y = 0; y < height; y++) {
    const rowOff = y * width;
    for (let x = 0; x < width; x++) {
      if (mask[rowOff + x] >= 0.5) {
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
      }
    }
  }
  if (maxY < 0) return new Float32Array(N);
  const labels = new Int32Array(N);

  const maxLabels = (N >> 1) + 2;
  const parent = new Int32Array(maxLabels);
  let nextLabel = 1;

  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const union = (a, b) => {
    const ra = find(a),
      rb = find(b);
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
  };

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const idx = y * width + x;
      if (mask[idx] < 0.5) continue;
      const left = x > 0 ? labels[idx - 1] : 0;
      const up = y > 0 ? labels[idx - width] : 0;
      if (left === 0 && up === 0) {
        labels[idx] = nextLabel;
        parent[nextLabel] = nextLabel;
        nextLabel++;
      } else if (left !== 0 && up === 0) {
        labels[idx] = left;
      } else if (left === 0 && up !== 0) {
        labels[idx] = up;
      } else {
        labels[idx] = Math.min(left, up);
        if (left !== up) union(left, up);
      }
    }
  }

  const areas = new Map();
  for (let i = 0; i < N; i++) {
    if (labels[i] === 0) continue;
    const root = find(labels[i]);
    areas.set(root, (areas.get(root) || 0) + 1);
  }

  let keepSet;
  if (keepTopK > 0) {
    const sorted = [...areas.entries()].sort((a, b) => b[1] - a[1]);
    keepSet = new Set(
      sorted
        .slice(0, keepTopK)
        .filter(([_, a]) => a >= minArea)
        .map(([r, _]) => r),
    );
  } else {
    keepSet = new Set(
      [...areas.entries()].filter(([_, a]) => a >= minArea).map(([r, _]) => r),
    );
  }

  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    if (labels[i] === 0) continue;
    const root = find(labels[i]);
    if (keepSet.has(root)) out[i] = 1.0;
  }

  const totalBlobs = areas.size;
  const keptBlobs = keepSet.size;
  const droppedArea = [...areas.entries()]
    .filter(([r, _]) => !keepSet.has(r))
    .reduce((s, [_, a]) => s + a, 0);

  return out;
}

function enhanceContrast(src, width, height, factor) {
  const result = new Float32Array(width * height);

  let minVal = 1,
    maxVal = 0;
  for (let i = 0; i < src.length; i++) {
    minVal = Math.min(minVal, src[i]);
    maxVal = Math.max(maxVal, src[i]);
  }

  const range = maxVal - minVal;
  if (range > 0) {
    for (let i = 0; i < src.length; i++) {
      let normalized = (src[i] - minVal) / range;

      normalized = Math.pow(normalized, 1.0 / factor);
      result[i] = normalized;
    }
  } else {
    return src;
  }

  return result;
}

function applyThreshold(src, width, height, thresholdValue) {
  const result = new Float32Array(width * height);
  const softness = 0.1;

  for (let i = 0; i < src.length; i++) {
    const value = src[i];

    if (value < thresholdValue - softness) {
      result[i] = 0;
    } else if (value > thresholdValue + softness) {
      result[i] = value;
    } else {
      const t = (value - (thresholdValue - softness)) / (2 * softness);
      result[i] = value * t;
    }
  }

  return result;
}

function gaussianBlur(src, width, height, sigma) {
  const kernelSize = Math.ceil(sigma * 3) * 2 + 1;
  const kernel = generateGaussianKernel(kernelSize, sigma);
  const halfKernel = Math.floor(kernelSize / 2);
  const result = new Float32Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      let weightSum = 0;

      for (let ky = -halfKernel; ky <= halfKernel; ky++) {
        for (let kx = -halfKernel; kx <= halfKernel; kx++) {
          const ny = Math.max(0, Math.min(height - 1, y + ky));
          const nx = Math.max(0, Math.min(width - 1, x + kx));
          const weight =
            kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
          sum += src[ny * width + nx] * weight;
          weightSum += weight;
        }
      }

      result[y * width + x] = sum / weightSum;
    }
  }

  return result;
}

function generateGaussianKernel(size, sigma) {
  const kernel = new Float32Array(size * size);
  const center = Math.floor(size / 2);
  const sigma2 = 2 * sigma * sigma;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      kernel[y * size + x] = Math.exp(-(dx * dx + dy * dy) / sigma2);
    }
  }

  return kernel;
}

function bilateralFilter(src, width, height, diameter, sigmaColor, sigmaSpace) {
  const result = new Float32Array(width * height);
  const radius = Math.floor(diameter / 2);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      let weightSum = 0;
      const centerValue = src[y * width + x];

      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const ny = Math.max(0, Math.min(height - 1, y + dy));
          const nx = Math.max(0, Math.min(width - 1, x + dx));
          const neighborValue = src[ny * width + nx];

          const spatialDist = dx * dx + dy * dy;
          const spatialWeight = Math.exp(
            -spatialDist / (2 * sigmaSpace * sigmaSpace),
          );

          const colorDist =
            (centerValue - neighborValue) * (centerValue - neighborValue);
          const colorWeight = Math.exp(
            -colorDist / (2 * sigmaColor * sigmaColor),
          );

          const weight = spatialWeight * colorWeight;
          sum += neighborValue * weight;
          weightSum += weight;
        }
      }

      result[y * width + x] = sum / weightSum;
    }
  }

  return result;
}

function unsharpMask(src, width, height, sigma, amount) {
  const blurred = gaussianBlur(src, width, height, sigma);
  const result = new Float32Array(width * height);

  for (let i = 0; i < src.length; i++) {
    const detail = src[i] - blurred[i];
    result[i] = Math.max(0, Math.min(1, src[i] + detail * amount));
  }

  return result;
}

// ────────────────────────────────────────────────────────────────────────
// Float32Array buffer pool. Many hot functions (guided filter, dilate,
// box blur) allocate scratch buffers per call. On a 5000x4000 source
// image at 1536-cap working resolution we allocate dozens of ~7.5 MB
// arrays per detect/clean call -- 5-10% of total wall-clock goes to GC
// and zero-init.
//
// The pool keeps a per-size LRU of released buffers. Acquire returns a
// zero'd Float32Array of the requested length (or larger -- caller must
// not rely on length matching). Release returns it for reuse.
//
// Sentinel "owned by pool" check via a WeakSet: only release buffers
// that came from the pool. Buffers that escape (returned from public
// functions, stored in long-lived state) just become regular GC
// objects when no longer referenced.
const _FP_POOL = new Map();         // length -> Float32Array[]
const _FP_OWNED = new WeakSet();    // tracks pool-owned arrays
const _FP_MAX_PER_BUCKET = 4;       // cap to avoid unbounded growth
function _fpAcquire(n) {
  const bucket = _FP_POOL.get(n);
  if (bucket && bucket.length > 0) {
    const buf = bucket.pop();
    buf.fill(0);
    return buf;
  }
  const buf = new Float32Array(n);
  _FP_OWNED.add(buf);
  return buf;
}
function _fpRelease(buf) {
  if (!buf || !_FP_OWNED.has(buf)) return;
  const n = buf.length;
  let bucket = _FP_POOL.get(n);
  if (!bucket) {
    bucket = [];
    _FP_POOL.set(n, bucket);
  }
  if (bucket.length < _FP_MAX_PER_BUCKET) bucket.push(buf);
}
// Optional: clear the pool (call between very different source images
// to release memory back to the OS).
function _fpClear() {
  _FP_POOL.clear();
}

// ────────────────────────────────────────────────────────────────────────
// WebAssembly hot-loop acceleration. The .wasm artifacts ship next to
// this JS file (fsv_postprocess.wasm + fsv_postprocess_simd.wasm). The
// loader feature-detects SIMD128, fetches the appropriate variant, and
// instantiates it asynchronously. All subsequent dilate/erode/box-blur/
// Lab calls dispatch through WASM for ~2-4x wall-clock vs the optimized
// JS path. If the .wasm file is missing or instantiation fails, the
// system silently falls back to WebGPU (if available) then to JS.
//
// The WASM module exposes a fixed set of functions (see fsv_postprocess.c
// public ABI). It runs in its own linear memory, separate from JS heap.
// We use a small buffer-table layer to memoize pointer allocations
// across calls of the same length, so per-call alloc overhead is
// amortized.
//
// Build: cd fence-staining-visualizer/wasm && make && make deploy

const _WASM = {
    status: "uninit",  // "uninit" | "loading" | "ready" | "unavailable"
    module: null,
    memory: null,
    exports: null,
    /* Memoized work buffers keyed by length. Reused across calls. */
    bufs: new Map(),
    /* SIMD probe bytes -- minimal WASM module that uses one v128 op. If
     * WebAssembly.validate() returns true, the runtime supports SIMD. */
    _SIMD_PROBE: new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
        0x03, 0x02, 0x01, 0x00,
        0x0a, 0x0a, 0x01, 0x08, 0x00, 0xfd, 0x0c,
        0x00, 0x00, 0x00, 0x00, 0x0b,
    ]),
};

function _wasmSupportsSimd() {
    if (typeof WebAssembly === "undefined" || !WebAssembly.validate) return false;
    try {
        return WebAssembly.validate(_WASM._SIMD_PROBE);
    } catch (_) { return false; }
}

function _wasmUrl(filename) {
    /* Resolve the .wasm URL relative to the script location. For the
     * WordPress plugin path, the .wasm file is in the plugin folder. */
    try {
        if (typeof document !== "undefined") {
            const scripts = document.getElementsByTagName("script");
            for (const s of scripts) {
                if (s.src && (s.src.endsWith("/app.js") || s.src.includes("app.js?"))) {
                    return s.src.replace(/app\.js(\?.*)?$/, filename);
                }
            }
        }
    } catch (_) {}
    /* Fallback: relative to current page. */
    return filename;
}

async function _wasmInit() {
    if (_WASM.status === "ready") return true;
    if (_WASM.status === "unavailable") return false;
    if (_WASM.status === "loading") {
        /* Another call already kicked off init; await it. */
        while (_WASM.status === "loading") {
            await new Promise(r => setTimeout(r, 10));
        }
        return _WASM.status === "ready";
    }
    if (typeof WebAssembly === "undefined" || !WebAssembly.instantiateStreaming) {
        _WASM.status = "unavailable";
        return false;
    }
    _WASM.status = "loading";
    try {
        const useSimd = _wasmSupportsSimd();
        const filename = useSimd ? "fsv_postprocess_simd.wasm" : "fsv_postprocess.wasm";
        const url = _wasmUrl(filename);
        /* Fetch then compile, so we can inspect required imports BEFORE
         * instantiating. emcc -O3 minifies import module names (e.g.
         * "env" -> "a"), so we cannot hardcode "env" / "wasi_snapshot_
         * preview1". Inspecting WebAssembly.Module.imports() is robust
         * against any minification scheme. */
        const response = await fetch(url, { credentials: "same-origin" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const bytes = await response.arrayBuffer();
        const wasmModule = await WebAssembly.compile(bytes);
        const required = WebAssembly.Module.imports(wasmModule);
        /* Function imports get a "smart" stub that handles memory.grow
         * requests from emscripten_resize_heap (whose name we cannot
         * identify after minification). Calling memory.grow for a
         * real heap-resize request keeps emmalloc working; for wasi
         * calls (proc_exit / fd_*) whose first arg is not a heap
         * size, the stub returns 0 which is the wasi success code
         * anyway. */
        const WASM_MAX_BYTES = 536870912;
        const WASM_PAGE_BYTES = 65536;
        const smartStub = (...args) => {
            /* No-arg call: likely emscripten_get_heap_size(). Return
             * the current heap size in bytes so emmalloc sees the
             * right boundary. */
            if (args.length === 0 && _WASM.memory) {
                return _WASM.memory.buffer.byteLength;
            }
            const arg = args[0];
            /* 1-arg call with a memory-size-shaped value: likely
             * emscripten_resize_heap(requestedSize). */
            if (args.length === 1 && typeof arg === "number" && arg > WASM_PAGE_BYTES && _WASM.memory) {
                try {
                    const cur = _WASM.memory.buffer.byteLength;
                    if (arg <= cur) return 1;
                    if (arg > WASM_MAX_BYTES) return 0;
                    const newSize = Math.min(WASM_MAX_BYTES, Math.ceil(arg / WASM_PAGE_BYTES) * WASM_PAGE_BYTES);
                    _WASM.memory.grow((newSize - cur) / WASM_PAGE_BYTES);
                    return 1;
                } catch {
                    return 0;
                }
            }
            return 0;
        };
        const importObj = {};
        for (const imp of required) {
            if (!importObj[imp.module]) importObj[imp.module] = {};
            if (imp.kind === "function") {
                importObj[imp.module][imp.name] = smartStub;
            } else if (imp.kind === "memory") {
                importObj[imp.module][imp.name] = new WebAssembly.Memory({ initial: 512, maximum: 8192 });
            } else if (imp.kind === "table") {
                importObj[imp.module][imp.name] = new WebAssembly.Table({ initial: 0, element: "anyfunc" });
            } else if (imp.kind === "global") {
                importObj[imp.module][imp.name] = new WebAssembly.Global({ value: "i32", mutable: false }, 0);
            }
        }
        const inst = { module: wasmModule, instance: await WebAssembly.instantiate(wasmModule, importObj) };
        _WASM.module = inst.module;
        const rawExports = inst.instance.exports;
        /* emcc's wasm-opt pass minifies export-table names (e.g.
         * "fsv_alloc" -> "f"). Each exported function's function.name
         * is the function's INDEX in the wasm (as a string, e.g.
         * "29"). The sibling `.symbols` file from --emit-symbol-map
         * maps INDEX -> original name (e.g. "29:fsv_alloc"). Compose
         * both to get origName -> minified export key. */
        let symbolMapText = "";
        try {
            const sm = await fetch(_wasmUrl(filename + ".symbols"), { credentials: "same-origin" });
            if (sm.ok) symbolMapText = await sm.text();
        } catch { /* ignore — falls back to no remap */ }
        const indexToOrig = {};
        for (const line of symbolMapText.split("\n")) {
            const idx = line.indexOf(":");
            if (idx <= 0) continue;
            const i = line.slice(0, idx).trim();
            const orig = line.slice(idx + 1).trim();
            if (i && orig) indexToOrig[i] = orig;
        }
        const origToMin = {};
        for (const [k, v] of Object.entries(rawExports)) {
            if (typeof v !== "function") continue;
            const orig = indexToOrig[v.name];
            if (orig) origToMin[orig] = k;
        }
        /* fsv_alloc/fsv_free are thin malloc/free wrappers that emcc
         * LTO can inline so they share an index with the allocator.
         * The symbol map only records one name per index, so resolve
         * by allocator-name candidates in priority order. With
         * -sMALLOC=emmalloc the actual function is `emmalloc_malloc`
         * / `emmalloc_free`. */
        const allocCandidates = ["malloc", "emmalloc_malloc", "dlmalloc"];
        const freeCandidates  = ["free",   "emmalloc_free",   "dlfree"];
        if (!origToMin.fsv_alloc) {
            for (const c of allocCandidates) {
                if (origToMin[c]) { origToMin.fsv_alloc = origToMin[c]; break; }
            }
        }
        if (!origToMin.fsv_free) {
            for (const c of freeCandidates) {
                if (origToMin[c]) { origToMin.fsv_free = origToMin[c]; break; }
            }
        }
        const remapped = new Proxy(rawExports, {
            get(target, prop) {
                if (typeof prop !== "string") return target[prop];
                if (prop in target) return target[prop];
                const min = origToMin[prop];
                return min ? target[min] : undefined;
            },
            has(target, prop) {
                if (typeof prop !== "string") return prop in target;
                return (prop in target) || (origToMin[prop] && origToMin[prop] in target);
            },
        });
        _WASM.exports = remapped;
        _WASM.memory = null;
        for (const k of Object.keys(rawExports)) {
            if (rawExports[k] instanceof WebAssembly.Memory) {
                _WASM.memory = rawExports[k];
                break;
            }
        }
        /* Run C++ static initializers. emcc's JS glue normally calls
         * this; since we bypass the glue, we must invoke it manually
         * before ANY other export. Without it, emmalloc's freelist
         * heads and heap pointers remain zero, and the first malloc
         * walks NULL -> OOB. */
        if (_WASM.exports.__wasm_call_ctors) _WASM.exports.__wasm_call_ctors();
        if (_WASM.exports.fsv_srgb_lut_init) _WASM.exports.fsv_srgb_lut_init();
        _WASM.status = "ready";
        return true;
    } catch (e) {
        console.warn("[fsv] WASM init failed, falling back:", e);
        _WASM.status = "unavailable";
        return false;
    }
}

/* Get-or-create a WASM-side Float32 buffer of length `n`. The same
 * pointer is returned on subsequent calls with the same n (within a
 * generation). Caller writes into HEAPF32 at byteOffset = ptr>>2, length = n.
 */
function _wasmBuf(n) {
    const key = `f32:${n}`;
    let entry = _WASM.bufs.get(key);
    if (!entry) {
        const ptr = _WASM.exports.fsv_alloc(n * 4);
        entry = { ptr, byteOffset: ptr >> 2, length: n };
        _WASM.bufs.set(key, entry);
    }
    return entry;
}

function _wasmBufBytes(nBytes) {
    const key = `u8:${nBytes}`;
    let entry = _WASM.bufs.get(key);
    if (!entry) {
        const ptr = _WASM.exports.fsv_alloc(nBytes);
        entry = { ptr, byteOffset: ptr, length: nBytes };
        _WASM.bufs.set(key, entry);
    }
    return entry;
}

/* Free all memoized WASM buffers. Call between very different image
 * sizes (e.g., on reset) to release memory back to WASM heap. */
function _wasmBufsClear() {
    if (!_WASM.exports || !_WASM.exports.fsv_free) {
        _WASM.bufs.clear();
        return;
    }
    for (const entry of _WASM.bufs.values()) {
        _WASM.exports.fsv_free(entry.ptr);
    }
    _WASM.bufs.clear();
}

/* WASM dilate wrapper. Returns a Float32Array (allocated fresh, not
 * pooled, since the result outlives the call). Bit-identical to JS
 * dilateFast.
 */
function _wasmDilate(mask, width, height, radius) {
    const n = width * height;
    const heap = new Float32Array(_WASM.memory.buffer);
    const heapU8 = new Uint8Array(_WASM.memory.buffer);
    const inBuf = _wasmBuf(n);
    const outBuf = _wasmBuf(n);
    /* The "scratch" buffer can be a third memoized buffer of length n.
     * We can't use the same key as in/out -- _wasmBuf returns the same
     * pointer for same key. Use a length-tagged key. */
    let scratch = _WASM.bufs.get(`f32:scratch:${n}`);
    if (!scratch) {
        const ptr = _WASM.exports.fsv_alloc(n * 4);
        scratch = { ptr, byteOffset: ptr >> 2, length: n };
        _WASM.bufs.set(`f32:scratch:${n}`, scratch);
    }
    /* Memory may have grown -- refresh heap view after every fsv_alloc. */
    const heap2 = new Float32Array(_WASM.memory.buffer);
    heap2.set(mask, inBuf.byteOffset);
    _WASM.exports.fsv_dilate(inBuf.ptr, outBuf.ptr, scratch.ptr, width, height, radius);
    /* Output buffer might be relocated if growth happened -- refresh. */
    const heap3 = new Float32Array(_WASM.memory.buffer);
    return heap3.slice(outBuf.byteOffset, outBuf.byteOffset + n);
}

/* WASM erode wrapper. Symmetric with dilate. */
function _wasmErode(mask, width, height, radius) {
    const n = width * height;
    const inBuf = _wasmBuf(n);
    const outBuf = _wasmBuf(n);
    let scratch = _WASM.bufs.get(`f32:scratch:${n}`);
    if (!scratch) {
        const ptr = _WASM.exports.fsv_alloc(n * 4);
        scratch = { ptr, byteOffset: ptr >> 2, length: n };
        _WASM.bufs.set(`f32:scratch:${n}`, scratch);
    }
    const heap2 = new Float32Array(_WASM.memory.buffer);
    heap2.set(mask, inBuf.byteOffset);
    _WASM.exports.fsv_erode(inBuf.ptr, outBuf.ptr, scratch.ptr, width, height, radius);
    const heap3 = new Float32Array(_WASM.memory.buffer);
    return heap3.slice(outBuf.byteOffset, outBuf.byteOffset + n);
}

/* Mask-aware box blur via WASM (calls fsv_box_blur_masked). */
function _wasmBoxBlurMasked(values, mask, w, h, radius) {
    const n = w * h;
    const inBuf = _wasmBuf(n);
    let maskBuf = _WASM.bufs.get(`f32:mask:${n}`);
    if (!maskBuf) {
        const ptr = _WASM.exports.fsv_alloc(n * 4);
        maskBuf = { ptr, byteOffset: ptr >> 2, length: n };
        _WASM.bufs.set(`f32:mask:${n}`, maskBuf);
    }
    const outBuf = _wasmBuf(n);
    let heap = new Float32Array(_WASM.memory.buffer);
    heap.set(values, inBuf.byteOffset);
    heap.set(mask, maskBuf.byteOffset);
    _WASM.exports.fsv_box_blur_masked(inBuf.ptr, maskBuf.ptr, outBuf.ptr, w, h, radius);
    heap = new Float32Array(_WASM.memory.buffer);
    return heap.slice(outBuf.byteOffset, outBuf.byteOffset + n);
}

/* Soft mask threshold via WASM (SIMD-vectorized). */
function _wasmSoftMaskThreshold(src, low, high) {
    const n = src.length;
    const inBuf = _wasmBuf(n);
    const outBuf = _wasmBuf(n);
    let heap = new Float32Array(_WASM.memory.buffer);
    heap.set(src, inBuf.byteOffset);
    _WASM.exports.fsv_soft_mask_threshold(inBuf.ptr, outBuf.ptr, n, low, high);
    heap = new Float32Array(_WASM.memory.buffer);
    return heap.slice(outBuf.byteOffset, outBuf.byteOffset + n);
}

/* Batched sRGB->Lab. rgba: Uint8ClampedArray of length n_pixels*4.
 * Returns Float32Array of length n_pixels*3 (interleaved L,a,b). */
function _wasmRgbaToLab(rgba, n_pixels) {
    const bytesIn = n_pixels * 4;
    const floatsOut = n_pixels * 3;
    let inBuf = _WASM.bufs.get(`u8:${bytesIn}`);
    if (!inBuf) {
        const ptr = _WASM.exports.fsv_alloc(bytesIn);
        inBuf = { ptr, byteOffset: ptr, length: bytesIn };
        _WASM.bufs.set(`u8:${bytesIn}`, inBuf);
    }
    const outBuf = _wasmBuf(floatsOut);
    let heapU8 = new Uint8Array(_WASM.memory.buffer);
    heapU8.set(rgba.subarray ? rgba.subarray(0, bytesIn) : rgba, inBuf.byteOffset);
    _WASM.exports.fsv_rgba_to_lab(inBuf.ptr, outBuf.ptr, n_pixels);
    const heapF32 = new Float32Array(_WASM.memory.buffer);
    return heapF32.slice(outBuf.byteOffset, outBuf.byteOffset + floatsOut);
}

/* Batched Lab->sRGB. Writes RGB bytes into rgba_out (length n_pixels*4),
 * preserving the alpha bytes that were already there. */
function _wasmLabToRgba(lab, rgba_out, n_pixels) {
    const floatsIn = n_pixels * 3;
    const bytesOut = n_pixels * 4;
    const inBuf = _wasmBuf(floatsIn);
    let outBuf = _WASM.bufs.get(`u8:out:${bytesOut}`);
    if (!outBuf) {
        const ptr = _WASM.exports.fsv_alloc(bytesOut);
        outBuf = { ptr, byteOffset: ptr, length: bytesOut };
        _WASM.bufs.set(`u8:out:${bytesOut}`, outBuf);
    }
    let heapF32 = new Float32Array(_WASM.memory.buffer);
    heapF32.set(lab, inBuf.byteOffset);
    let heapU8 = new Uint8Array(_WASM.memory.buffer);
    heapU8.set(rgba_out.subarray ? rgba_out.subarray(0, bytesOut) : rgba_out, outBuf.byteOffset);
    _WASM.exports.fsv_lab_to_rgba(inBuf.ptr, outBuf.ptr, n_pixels);
    heapU8 = new Uint8Array(_WASM.memory.buffer);
    rgba_out.set(heapU8.subarray(outBuf.byteOffset, outBuf.byteOffset + bytesOut));
    return rgba_out;
}

/* SIMD pixelwise diff and mul (used by guided filter). */
function _wasmPixelwiseDiff(a, b) {
    const n = a.length;
    let aBuf = _WASM.bufs.get(`f32:opa:${n}`);
    if (!aBuf) {
        const ptr = _WASM.exports.fsv_alloc(n * 4);
        aBuf = { ptr, byteOffset: ptr >> 2, length: n };
        _WASM.bufs.set(`f32:opa:${n}`, aBuf);
    }
    let bBuf = _WASM.bufs.get(`f32:opb:${n}`);
    if (!bBuf) {
        const ptr = _WASM.exports.fsv_alloc(n * 4);
        bBuf = { ptr, byteOffset: ptr >> 2, length: n };
        _WASM.bufs.set(`f32:opb:${n}`, bBuf);
    }
    const outBuf = _wasmBuf(n);
    let heap = new Float32Array(_WASM.memory.buffer);
    heap.set(a, aBuf.byteOffset);
    heap.set(b, bBuf.byteOffset);
    _WASM.exports.fsv_pixelwise_diff(aBuf.ptr, bBuf.ptr, outBuf.ptr, n);
    heap = new Float32Array(_WASM.memory.buffer);
    return heap.slice(outBuf.byteOffset, outBuf.byteOffset + n);
}

/* Full guided filter pipeline in one WASM call. Allocates 6 scratch
 * buffers (memoized) + input/guide/mask/output. Eliminates 4 JS<->WASM
 * transitions per call vs the per-primitive path (where maskedBoxBlur
 * dispatches individually). Bit-identical to _guidedFilter. */
function _wasmGuidedFilter(input, guide, mask, w, h, radius, epsilon) {
    const n = w * h;
    const inputBuf = _wasmBuf(n);
    let guideBuf = _WASM.bufs.get(`f32:guide:${n}`);
    if (!guideBuf) {
        const ptr = _WASM.exports.fsv_alloc(n * 4);
        guideBuf = { ptr, byteOffset: ptr >> 2, length: n };
        _WASM.bufs.set(`f32:guide:${n}`, guideBuf);
    }
    let maskBuf = _WASM.bufs.get(`f32:mask:${n}`);
    if (!maskBuf) {
        const ptr = _WASM.exports.fsv_alloc(n * 4);
        maskBuf = { ptr, byteOffset: ptr >> 2, length: n };
        _WASM.bufs.set(`f32:mask:${n}`, maskBuf);
    }
    const outBuf = _wasmBuf(n);
    /* Six scratch buffers, each keyed distinctly so they don't alias. */
    const scratches = [];
    for (let k = 0; k < 6; k++) {
        const key = `f32:gfsc${k}:${n}`;
        let sc = _WASM.bufs.get(key);
        if (!sc) {
            const ptr = _WASM.exports.fsv_alloc(n * 4);
            sc = { ptr, byteOffset: ptr >> 2, length: n };
            _WASM.bufs.set(key, sc);
        }
        scratches.push(sc);
    }
    let heap = new Float32Array(_WASM.memory.buffer);
    heap.set(input, inputBuf.byteOffset);
    heap.set(guide, guideBuf.byteOffset);
    heap.set(mask, maskBuf.byteOffset);
    _WASM.exports.fsv_guided_filter(
        inputBuf.ptr, guideBuf.ptr, maskBuf.ptr, outBuf.ptr,
        scratches[0].ptr, scratches[1].ptr, scratches[2].ptr,
        scratches[3].ptr, scratches[4].ptr, scratches[5].ptr,
        w, h, radius, epsilon,
    );
    heap = new Float32Array(_WASM.memory.buffer);
    return heap.slice(outBuf.byteOffset, outBuf.byteOffset + n);
}

/* Feathered alpha-blend composite. Replaces the JS per-pixel loop in
 * cleanFence phase 3 with a single SIMD WASM call. orig, modified, out
 * are Uint8(Clamped)Arrays of length n_pixels*4 (interleaved RGBA).
 * alpha is a Float32Array of length n_pixels (0..1). Writes to `out`
 * (which may alias `orig` -- the C function handles the in-place case
 * correctly because it reads orig[i] before writing out[i] per pixel). */
function _wasmAlphaBlendRgba(orig, modified, alpha, out, n_pixels) {
    const bytesIn = n_pixels * 4;
    let origBuf = _WASM.bufs.get(`u8:blendOrig:${bytesIn}`);
    if (!origBuf) {
        const ptr = _WASM.exports.fsv_alloc(bytesIn);
        origBuf = { ptr, byteOffset: ptr, length: bytesIn };
        _WASM.bufs.set(`u8:blendOrig:${bytesIn}`, origBuf);
    }
    let modBuf = _WASM.bufs.get(`u8:blendMod:${bytesIn}`);
    if (!modBuf) {
        const ptr = _WASM.exports.fsv_alloc(bytesIn);
        modBuf = { ptr, byteOffset: ptr, length: bytesIn };
        _WASM.bufs.set(`u8:blendMod:${bytesIn}`, modBuf);
    }
    const alphaBuf = _wasmBuf(n_pixels);
    let outBuf = _WASM.bufs.get(`u8:blendOut:${bytesIn}`);
    if (!outBuf) {
        const ptr = _WASM.exports.fsv_alloc(bytesIn);
        outBuf = { ptr, byteOffset: ptr, length: bytesIn };
        _WASM.bufs.set(`u8:blendOut:${bytesIn}`, outBuf);
    }
    let heapU8 = new Uint8Array(_WASM.memory.buffer);
    heapU8.set(orig.subarray ? orig.subarray(0, bytesIn) : orig, origBuf.byteOffset);
    heapU8.set(modified.subarray ? modified.subarray(0, bytesIn) : modified, modBuf.byteOffset);
    const heapF32 = new Float32Array(_WASM.memory.buffer);
    heapF32.set(alpha, alphaBuf.byteOffset);
    _WASM.exports.fsv_alpha_blend_rgba(
        origBuf.ptr, modBuf.ptr, alphaBuf.ptr, outBuf.ptr, n_pixels,
    );
    heapU8 = new Uint8Array(_WASM.memory.buffer);
    out.set(heapU8.subarray(outBuf.byteOffset, outBuf.byteOffset + bytesIn));
    return out;
}

function _wasmPixelwiseMul(a, b) {
    const n = a.length;
    let aBuf = _WASM.bufs.get(`f32:opa:${n}`);
    if (!aBuf) {
        const ptr = _WASM.exports.fsv_alloc(n * 4);
        aBuf = { ptr, byteOffset: ptr >> 2, length: n };
        _WASM.bufs.set(`f32:opa:${n}`, aBuf);
    }
    let bBuf = _WASM.bufs.get(`f32:opb:${n}`);
    if (!bBuf) {
        const ptr = _WASM.exports.fsv_alloc(n * 4);
        bBuf = { ptr, byteOffset: ptr >> 2, length: n };
        _WASM.bufs.set(`f32:opb:${n}`, bBuf);
    }
    const outBuf = _wasmBuf(n);
    let heap = new Float32Array(_WASM.memory.buffer);
    heap.set(a, aBuf.byteOffset);
    heap.set(b, bBuf.byteOffset);
    _WASM.exports.fsv_pixelwise_mul(aBuf.ptr, bBuf.ptr, outBuf.ptr, n);
    heap = new Float32Array(_WASM.memory.buffer);
    return heap.slice(outBuf.byteOffset, outBuf.byteOffset + n);
}

// ────────────────────────────────────────────────────────────────────────
// WebGPU compute acceleration. Targets the dilate operation (the dominant
// cost in spatiallyGuidedRecovery and recoverAdjacentToSurvivors). When
// available, replaces the JS sliding-window-max with a parallel GPU
// dispatch (each output pixel = one thread). Fully falls back to the
// optimized JS dilateFast when WebGPU is unavailable or fails -- output
// is bit-identical either way.
//
// Browser support as of 2026:
//   Chrome / Edge desktop: full
//   Chrome Android: full (recent versions)
//   Chrome Linux:   requires --enable-unsafe-webgpu flag
//   Safari (macOS/iOS): preview / WebKit nightly only
//   Firefox: nightly only
//
// The detection + init is done once per page (cached). If support is
// present and init succeeds, all subsequent dilate calls go through GPU.
// If anything fails, we permanently mark GPU unavailable for the session
// and use the optimized JS path.

const _GPU_DILATE_WGSL = /* wgsl */`
struct Params {
  width: u32,
  height: u32,
  radius: u32,
  axis: u32,   // 0 = row pass (along x), 1 = col pass (along y)
};
@group(0) @binding(0) var<storage, read> inBuf: array<f32>;
@group(0) @binding(1) var<storage, read_write> outBuf: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = params.width;
  let h = params.height;
  let r = i32(params.radius);
  let axis = params.axis;
  if (gid.x >= w || gid.y >= h) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let wI = i32(w);
  let hI = i32(h);
  var m: f32 = 0.0;
  if (axis == 0u) {
    // Row pass: scan along x within radius r
    var xStart = x - r;
    var xEnd = x + r;
    if (xStart < 0) { xStart = 0; }
    if (xEnd > wI - 1) { xEnd = wI - 1; }
    let rowOff = u32(y) * w;
    for (var xi: i32 = xStart; xi <= xEnd; xi = xi + 1) {
      let v = inBuf[rowOff + u32(xi)];
      if (v > m) { m = v; }
    }
  } else {
    // Col pass: scan along y within radius r
    var yStart = y - r;
    var yEnd = y + r;
    if (yStart < 0) { yStart = 0; }
    if (yEnd > hI - 1) { yEnd = hI - 1; }
    for (var yi: i32 = yStart; yi <= yEnd; yi = yi + 1) {
      let v = inBuf[u32(yi) * w + u32(x)];
      if (v > m) { m = v; }
    }
  }
  outBuf[u32(y) * w + u32(x)] = m;
}
`;

const _GPU = {
  status: "uninit",   // "uninit" | "ready" | "unavailable"
  device: null,
  pipeline: null,
  bindGroupLayout: null,
};

async function _gpuInit() {
  if (_GPU.status === "ready") return true;
  if (_GPU.status === "unavailable") return false;
  if (typeof navigator === "undefined" || !navigator.gpu) {
    _GPU.status = "unavailable";
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      _GPU.status = "unavailable";
      return false;
    }
    _GPU.device = await adapter.requestDevice();
    const module = _GPU.device.createShaderModule({
      label: "fsv-dilate-shader",
      code: _GPU_DILATE_WGSL,
    });
    _GPU.pipeline = _GPU.device.createComputePipeline({
      label: "fsv-dilate-pipeline",
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    _GPU.status = "ready";
    return true;
  } catch (e) {
    console.warn("[fsv] WebGPU init failed, falling back to JS:", e);
    _GPU.status = "unavailable";
    return false;
  }
}

// GPU dilate. Returns a Float32Array (not pooled because of GPU readback
// timing). Caller awaits.
async function _gpuDilate(input, width, height, radius) {
  const dev = _GPU.device;
  const N = width * height;
  const byteLen = N * 4;

  const inBuf = dev.createBuffer({
    size: byteLen,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const tmpBuf = dev.createBuffer({
    size: byteLen,
    usage: GPUBufferUsage.STORAGE,
  });
  const outBuf = dev.createBuffer({
    size: byteLen,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const paramsBuf = dev.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const readBuf = dev.createBuffer({
    size: byteLen,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  dev.queue.writeBuffer(inBuf, 0, input.buffer, input.byteOffset, input.byteLength);

  function _dispatch(srcBuf, dstBuf, axis) {
    const params = new Uint32Array([width, height, radius, axis]);
    dev.queue.writeBuffer(paramsBuf, 0, params);
    const bind = dev.createBindGroup({
      layout: _GPU.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: srcBuf } },
        { binding: 1, resource: { buffer: dstBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
    });
    const cmd = dev.createCommandEncoder();
    const pass = cmd.beginComputePass();
    pass.setPipeline(_GPU.pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(
      Math.ceil(width / 8),
      Math.ceil(height / 8),
      1,
    );
    pass.end();
    dev.queue.submit([cmd.finish()]);
  }

  _dispatch(inBuf, tmpBuf, 0);  // row pass
  _dispatch(tmpBuf, outBuf, 1); // col pass

  const cmd = dev.createCommandEncoder();
  cmd.copyBufferToBuffer(outBuf, 0, readBuf, 0, byteLen);
  dev.queue.submit([cmd.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();

  inBuf.destroy(); tmpBuf.destroy(); outBuf.destroy();
  paramsBuf.destroy(); readBuf.destroy();
  return result;
}

// Top-level accelerated dilate. Tier-1 fallback chain:
//   1. WASM SIMD (if loaded; ~3-4x JS)         <- preferred for medium loads
//   2. WASM scalar (if loaded; ~1.5-2x JS)
//   3. WebGPU compute (if available; massive throughput on big images)
//   4. Optimized JS dilateFast (universal fallback)
//
// WASM is preferred over WebGPU below a threshold image size because GPU
// dispatch overhead (~5-10ms) dominates for small buffers. At 1.9M
// pixels (1536x1228 working res) GPU wins; below ~250k pixels WASM is
// faster.
//
// Output is bit-identical across all tiers.
async function dilateAccelerated(mask, width, height, radius) {
  const n = width * height;
  /* Tier 1/2: WASM (preferred if loaded; covers all sizes well). */
  if (await _wasmInit()) {
    try {
      return _wasmDilate(mask, width, height, radius);
    } catch (e) {
      console.warn("[fsv] WASM dilate failed, fallback:", e);
      _WASM.status = "unavailable";
    }
  }
  /* Tier 3: WebGPU (best on large images when WASM unavailable). */
  if (radius >= 8 && n >= 250000 && (await _gpuInit())) {
    try {
      return await _gpuDilate(mask, width, height, radius);
    } catch (e) {
      console.warn("[fsv] GPU dilate failed, fallback to JS:", e);
      _GPU.status = "unavailable";
    }
  }
  /* Tier 4: optimized JS (universal). */
  return dilateFast(mask, width, height, radius);
}

// Compute the tight bounding box of positive pixels in a soft mask. Used by
// CC labeling functions to skip empty rows/cols at the top/bottom/sides of
// the image -- avoids scanning the typical 25-40% of pixels that are
// background. Returns inclusive [minY, maxY, minX, maxX] or
// {empty: true} if nothing was found.
function _maskExtent(mask, width, height) {
  let minY = height, maxY = -1, minX = width, maxX = -1;
  for (let y = 0; y < height; y++) {
    const rowOff = y * width;
    for (let x = 0; x < width; x++) {
      if (mask[rowOff + x] > 0) {
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
      }
    }
  }
  if (maxY < 0) return { empty: true, minY: 0, maxY: -1, minX: 0, maxX: -1 };
  return { empty: false, minY, maxY, minX, maxX };
}

function bilinearResize(src, srcW, srcH, dstW, dstH) {
  const dst = new Float32Array(dstW * dstH);
  const xRatio = srcW / dstW;
  const yRatio = srcH / dstH;
  const srcWLast = srcW - 1;
  const srcHLast = srcH - 1;

  for (let y = 0; y < dstH; y++) {
    const srcY = y * yRatio;
    // `| 0` is a bitwise truncation toward zero — equivalent to Math.floor
    // for non-negative inputs (always true here: srcY = y * srcH / dstH >= 0)
    // and 10-30% faster on V8's hot loops.
    const y1 = srcY | 0;
    const y2 = y1 < srcHLast ? y1 + 1 : srcHLast;
    const dy = srcY - y1;
    const oneMinusDy = 1 - dy;
    const y1Off = y1 * srcW;
    const y2Off = y2 * srcW;
    const dstRow = y * dstW;

    for (let x = 0; x < dstW; x++) {
      const srcX = x * xRatio;
      const x1 = srcX | 0;
      const x2 = x1 < srcWLast ? x1 + 1 : srcWLast;
      const dx = srcX - x1;
      const oneMinusDx = 1 - dx;

      const v11 = src[y1Off + x1];
      const v21 = src[y1Off + x2];
      const v12 = src[y2Off + x1];
      const v22 = src[y2Off + x2];

      dst[dstRow + x] =
        v11 * oneMinusDx * oneMinusDy +
        v21 * dx * oneMinusDy +
        v12 * oneMinusDx * dy +
        v22 * dx * dy;
    }
  }

  return dst;
}

function morphologicalClose(mask, width, height, kernelSize) {
  const dilated = dilate(mask, width, height, kernelSize);

  const eroded = erode(dilated, width, height, kernelSize);
  return eroded;
}

// Public dilate -- tries WASM first, falls back to optimized JS.
// kernelSize is interpreted as (2 * radius + 1); we convert to radius.
function dilate(mask, width, height, kernelSize) {
  const radius = (kernelSize / 2) | 0;
  if (_WASM.status === "ready") {
    try { return _wasmDilate(mask, width, height, radius); }
    catch (e) {
      console.warn("[fsv] WASM dilate failed in morphological close, fallback:", e);
      _WASM.status = "unavailable";
    }
  }
  return _dilateJS(mask, width, height, kernelSize);
}

// JS sliding-window dilate (the previous implementation). Used when WASM
// is unavailable or fails. Bit-identical to dilateFast output.
function _dilateJS(mask, width, height, kernelSize) {
  const radius = (kernelSize / 2) | 0;
  const N = width * height;
  const tmp = _fpAcquire(N);
  /* See note in dilateFast: monotonic head/tail need full-pass capacity. */
  const dqRow = new Int32Array(width + radius);
  const dqCol = new Int32Array(height + radius);

  for (let y = 0; y < height; y++) {
    const row = y * width;
    let head = 0, tail = 0;
    for (let x = 0; x < width + radius; x++) {
      if (x < width) {
        const v = mask[row + x];
        while (head < tail && mask[row + dqRow[tail - 1]] <= v) tail--;
        dqRow[tail++] = x;
      }
      const winStart = x - radius;
      const winLeft = winStart - radius;
      while (head < tail && dqRow[head] < winLeft) head++;
      if (winStart >= 0) tmp[row + winStart] = mask[row + dqRow[head]];
    }
  }
  const result = new Float32Array(N);
  for (let x = 0; x < width; x++) {
    let head = 0, tail = 0;
    for (let y = 0; y < height + radius; y++) {
      if (y < height) {
        const v = tmp[y * width + x];
        while (head < tail && tmp[dqCol[tail - 1] * width + x] <= v) tail--;
        dqCol[tail++] = y;
      }
      const winStart = y - radius;
      const winLeft = winStart - radius;
      while (head < tail && dqCol[head] < winLeft) head++;
      if (winStart >= 0) result[winStart * width + x] = tmp[dqCol[head] * width + x];
    }
  }
  _fpRelease(tmp);
  return result;
}

// Public erode -- tries WASM first, falls back to optimized JS.
function erode(mask, width, height, kernelSize) {
  const radius = (kernelSize / 2) | 0;
  if (_WASM.status === "ready") {
    try { return _wasmErode(mask, width, height, radius); }
    catch (e) {
      console.warn("[fsv] WASM erode failed in morphological close, fallback:", e);
      _WASM.status = "unavailable";
    }
  }
  return _erodeJS(mask, width, height, kernelSize);
}

function _erodeJS(mask, width, height, kernelSize) {
  // Sliding-window MIN: identical algorithm, comparison reversed (<= -> >=)
  // and the "neutral" value is 1 (the upper bound of soft mask values).
  const radius = (kernelSize / 2) | 0;
  const N = width * height;
  const tmp = _fpAcquire(N);
  /* See note in dilateFast: monotonic head/tail need full-pass capacity. */
  const dqRow = new Int32Array(width + radius);
  const dqCol = new Int32Array(height + radius);

  for (let y = 0; y < height; y++) {
    const row = y * width;
    let head = 0, tail = 0;
    for (let x = 0; x < width + radius; x++) {
      if (x < width) {
        const v = mask[row + x];
        while (head < tail && mask[row + dqRow[tail - 1]] >= v) tail--;
        dqRow[tail++] = x;
      }
      const winStart = x - radius;
      const winLeft = winStart - radius;
      while (head < tail && dqRow[head] < winLeft) head++;
      if (winStart >= 0) tmp[row + winStart] = mask[row + dqRow[head]];
    }
  }
  const result = new Float32Array(N);
  for (let x = 0; x < width; x++) {
    let head = 0, tail = 0;
    for (let y = 0; y < height + radius; y++) {
      if (y < height) {
        const v = tmp[y * width + x];
        while (head < tail && tmp[dqCol[tail - 1] * width + x] >= v) tail--;
        dqCol[tail++] = y;
      }
      const winStart = y - radius;
      const winLeft = winStart - radius;
      while (head < tail && dqCol[head] < winLeft) head++;
      if (winStart >= 0) result[winStart * width + x] = tmp[dqCol[head] * width + x];
    }
  }
  _fpRelease(tmp);
  return result;
}

function drawMask(mask) {
  const width = originalImage.width;
  const height = originalImage.height;

  maskCanvas.width = width;
  maskCanvas.height = height;
  const ctx = maskCanvas.getContext("2d");
  const imageData = ctx.createImageData(width, height);

  for (let i = 0; i < mask.length; i++) {
    const value = Math.floor(mask[i] * 255);
    imageData.data[i * 4] = value;
    imageData.data[i * 4 + 1] = value;
    imageData.data[i * 4 + 2] = value;
    imageData.data[i * 4 + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
}

function deriveRecolorAlpha(softMask, fullAlphaThreshold) {
  const out = new Float32Array(softMask.length);
  const thr = Math.max(1e-6, fullAlphaThreshold);
  let core = 0,
    edge = 0;
  for (let i = 0; i < softMask.length; i++) {
    const v = softMask[i];
    if (v <= 0) {
      out[i] = 0;
    } else if (v >= thr) {
      out[i] = 1.0;
      core++;
    } else {
      out[i] = v / thr;
      edge++;
    }
  }
  return out;
}

function applyResultHeader() {
  const HEADERS = {
    original: { title: "Fence Preview", sub: "Original" },
    cleaned: { title: "Cleaned Fence", sub: "Cleaned" },
    stained: { title: "Stained Result", sub: "Stained Preview" },
    cleaned_stained: {
      title: "Cleaned & Stained Result",
      sub: "Cleaned & Stained",
    },
  };
  const h = HEADERS[resultState] || HEADERS.original;
  if (canvasTitleText) canvasTitleText.textContent = h.title;
  if (canvasLabel) canvasLabel.textContent = h.sub;
}

const CLEAN_CFG = {
  targetA: 6.0,
  targetB: 17.0,

  maxCorrection: 0.55,

  chromaCeiling: 19.0,

  greyChromaMax: 9.0,
  greyChromaSoft: 24.0,

  algaeAMax: -2.0,
  algaeAFull: -8.0,

  highlightLStar: 92.0,
};

// Precomputed LUT for sRGB (0..255 int) -> linear (float). Replaces the
// per-call Math.pow + divide. Built once at module load; each LUT hit is a
// pure indexed array read (~30x faster than the original Math.pow form).
// Used in the per-pixel Lab conversion hot loop, which fires millions of
// times per cleanFence call.
const _SRGB_TO_LINEAR_LUT = (() => {
  const lut = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    lut[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  return lut;
})();
function _srgbToLinear(c) {
  // c is sRGB byte in [0, 255]. Math.pow + divide eliminated via LUT.
  return _SRGB_TO_LINEAR_LUT[c | 0];
}
function _linearToSrgb(c) {
  const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  return Math.max(0, Math.min(255, v * 255));
}
function _cleanRgbToLab(r, g, b) {
  const rl = _SRGB_TO_LINEAR_LUT[r | 0];
  const gl = _SRGB_TO_LINEAR_LUT[g | 0];
  const bl = _SRGB_TO_LINEAR_LUT[b | 0];

  const x = (0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) / 0.95047;
  const y = (0.2126729 * rl + 0.7151522 * gl + 0.072175 * bl) / 1.0;
  const z = (0.0193339 * rl + 0.119192 * gl + 0.9503041 * bl) / 1.08883;
  // Math.cbrt is 3-5x faster than Math.pow(t, 1/3) and bit-equivalent
  // within float precision.
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(x),
    fy = f(y),
    fz = f(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}
function _cleanLabToRgb(L, a, b) {
  const fy = (L + 16) / 116;
  const fx = a / 500 + fy;
  const fz = fy - b / 200;
  const inv = (t) => {
    const t3 = t * t * t;
    return t3 > 0.008856 ? t3 : (t - 16 / 116) / 7.787;
  };
  const X = 0.95047 * inv(fx);
  const Y = 1.0 * inv(fy);
  const Z = 1.08883 * inv(fz);
  const rl = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
  const gl = -0.969266 * X + 1.8760108 * Y + 0.041556 * Z;
  const bl = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
  return [_linearToSrgb(rl), _linearToSrgb(gl), _linearToSrgb(bl)];
}

function _guidedFilter(input, guide, mask, w, h, radius, epsilon) {
  // Try the single-call WASM path first -- eliminates 4+ JS<->WASM
  // transitions and runs all arithmetic in WASM SIMD. Falls through to
  // the per-primitive path if WASM isn't ready or the all-in-one
  // function fails.
  if (_WASM.status === "ready") {
    try { return _wasmGuidedFilter(input, guide, mask, w, h, radius, epsilon); }
    catch (e) {
      console.warn("[fsv] WASM guided filter failed, falling back to per-primitive:", e);
      _WASM.status = "unavailable";
    }
  }
  const N = w * h;
  const Igi = _fpAcquire(N);
  const Igg = _fpAcquire(N);
  for (let i = 0; i < N; i++) {
    Igi[i] = guide[i] * input[i];
    Igg[i] = guide[i] * guide[i];
  }
  const meanG = maskedBoxBlur(guide, mask, w, h, radius);
  const meanI = maskedBoxBlur(input, mask, w, h, radius);
  const meanGI = maskedBoxBlur(Igi, mask, w, h, radius);
  const meanGG = maskedBoxBlur(Igg, mask, w, h, radius);
  _fpRelease(Igi); _fpRelease(Igg);
  const A = _fpAcquire(N);
  const B = _fpAcquire(N);
  for (let i = 0; i < N; i++) {
    const mg = meanG[i];
    const varG = meanGG[i] - mg * mg;
    A[i] = (meanGI[i] - mg * meanI[i]) / (varG + epsilon);
    B[i] = meanI[i] - A[i] * mg;
  }
  const meanA = maskedBoxBlur(A, mask, w, h, radius);
  const meanB = maskedBoxBlur(B, mask, w, h, radius);
  _fpRelease(A); _fpRelease(B);
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    if (mask[i] > 0) out[i] = meanA[i] * guide[i] + meanB[i];
  }
  return out;
}

// Multi-channel guided filter: when 3 channels share the same guide (the L*
// channel in cleanFence's case), meanG and meanGG don't depend on `input` --
// they're functions of the guide only. Compute them ONCE, reuse 3x.
// Saves 6 maskedBoxBlur calls per radius vs three _guidedFilter invocations.
// Returns [outL, outA, outB] in one call. Output is bit-identical to three
// separate _guidedFilter(inputN, guide, ...) calls.
function _guidedFilterMulti3(in0, in1, in2, guide, mask, w, h, radius, epsilon) {
  // When WASM is ready, dispatch each channel through the single-call
  // fsv_guided_filter. We lose the meanG/meanGG dedup (which was the JS
  // optimization for this multi-channel case), but gain a much larger
  // win from running each pass entirely in WASM SIMD with only one
  // JS<->WASM transition per channel.
  //
  // Cost analysis (1536x1228 = 1.9M pixels):
  //   JS dedup path:   2 shared box blurs + 3 x (2 unique blurs + arith)
  //                  = 8 box blurs total, all JS arithmetic
  //   WASM all-in-one: 3 x (4 box blurs + arith) = 12 box blurs total,
  //                    but all in WASM SIMD (~4x faster per blur)
  //   Net: 12 / 4 = 3 effective JS-box-blur units of work, vs 8 for JS
  //        dedup -- WASM wins by ~2.5x.
  if (_WASM.status === "ready") {
    try {
      return [
        _wasmGuidedFilter(in0, guide, mask, w, h, radius, epsilon),
        _wasmGuidedFilter(in1, guide, mask, w, h, radius, epsilon),
        _wasmGuidedFilter(in2, guide, mask, w, h, radius, epsilon),
      ];
    } catch (e) {
      console.warn("[fsv] WASM multi-channel guided filter failed, falling back to JS dedup:", e);
      _WASM.status = "unavailable";
    }
  }
  const N = w * h;
  // Guide-only stats: same across all 3 input channels.
  const Igg = _fpAcquire(N);
  for (let i = 0; i < N; i++) Igg[i] = guide[i] * guide[i];
  const meanG = maskedBoxBlur(guide, mask, w, h, radius);
  const meanGG = maskedBoxBlur(Igg, mask, w, h, radius);
  _fpRelease(Igg);

  function _onePass(input) {
    const Igi = _fpAcquire(N);
    for (let i = 0; i < N; i++) Igi[i] = guide[i] * input[i];
    const meanI = maskedBoxBlur(input, mask, w, h, radius);
    const meanGI = maskedBoxBlur(Igi, mask, w, h, radius);
    _fpRelease(Igi);
    const A = _fpAcquire(N);
    const B = _fpAcquire(N);
    for (let i = 0; i < N; i++) {
      const mg = meanG[i];
      const varG = meanGG[i] - mg * mg;
      A[i] = (meanGI[i] - mg * meanI[i]) / (varG + epsilon);
      B[i] = meanI[i] - A[i] * mg;
    }
    const meanA = maskedBoxBlur(A, mask, w, h, radius);
    const meanB = maskedBoxBlur(B, mask, w, h, radius);
    _fpRelease(A); _fpRelease(B);
    const out = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      if (mask[i] > 0) out[i] = meanA[i] * guide[i] + meanB[i];
    }
    return out;
  }
  return [_onePass(in0), _onePass(in1), _onePass(in2)];
}

function _plankJitter(x, plankWidth) {
  const plankIdx = Math.floor(x / Math.max(8, plankWidth));

  const s = plankIdx * 73 + 17;
  return [
    Math.sin(s * 0.71) * 1.2,
    Math.sin(s * 1.13) * 0.6,
    Math.cos(s * 0.89) * 0.7,
  ];
}

// Public maskedBoxBlur -- tries WASM first (~3-4x speedup), falls back
// to optimized JS. Called 16-24 times per cleanFence call, so this is
// the highest-leverage WASM wire.
function maskedBoxBlur(values, mask, w, h, radius) {
  if (_WASM.status === "ready") {
    try { return _wasmBoxBlurMasked(values, mask, w, h, radius); }
    catch (e) {
      console.warn("[fsv] WASM box blur failed, fallback:", e);
      _WASM.status = "unavailable";
    }
  }
  return _maskedBoxBlurJS(values, mask, w, h, radius);
}

function _maskedBoxBlurJS(values, mask, w, h, radius) {
  // Optimizations vs original:
  //  - `radius | 0` instead of Math.floor (10-30% faster for non-negative)
  //  - hoist y-bounded constants (y0, y1, y0stride, y1stride) out of the
  //    inner x-loop; same with rowOff, iiOff, prevOff for the integral pass
  //  - replace Math.max/Math.min with conditional expressions
  //  - cache integral lookups (y0*stride, y1*stride) once per y
  //  - the bounds clamping uses `< 0 ? 0 : v` / `> last ? last : v` form
  const N = w * h;
  const r = radius < 1 ? 1 : radius | 0;
  const stride = w + 1;
  const hLast = h - 1;
  const wLast = w - 1;
  const IIvm = new Float64Array(stride * (h + 1));
  const IIm = new Float64Array(stride * (h + 1));
  for (let y = 0; y < h; y++) {
    let rowVM = 0, rowM = 0;
    const rowOff = y * w;
    const iiOff = (y + 1) * stride;
    const prevOff = y * stride;
    for (let x = 0; x < w; x++) {
      const i = rowOff + x;
      const m = mask[i];
      rowVM += values[i] * m;
      rowM += m;
      const xPlus1 = x + 1;
      IIvm[iiOff + xPlus1] = IIvm[prevOff + xPlus1] + rowVM;
      IIm[iiOff + xPlus1] = IIm[prevOff + xPlus1] + rowM;
    }
  }
  const out = new Float32Array(N);
  for (let y = 0; y < h; y++) {
    const y0raw = y - r;
    const y1raw = y + r;
    const y0 = y0raw < 0 ? 0 : y0raw;
    const y1 = (y1raw > hLast ? hLast : y1raw) + 1;
    const y0s = y0 * stride;
    const y1s = y1 * stride;
    const outRow = y * w;
    for (let x = 0; x < w; x++) {
      const x0raw = x - r;
      const x1raw = x + r;
      const x0 = x0raw < 0 ? 0 : x0raw;
      const x1 = (x1raw > wLast ? wLast : x1raw) + 1;
      const sumW = IIm[y1s + x1] - IIm[y0s + x1] - IIm[y1s + x0] + IIm[y0s + x0];
      if (sumW <= 0) {
        out[outRow + x] = values[outRow + x];
        continue;
      }
      const sumVM = IIvm[y1s + x1] - IIvm[y0s + x1] - IIvm[y1s + x0] + IIvm[y0s + x0];
      out[outRow + x] = sumVM / sumW;
    }
  }
  return out;
}

async function cleanFence() {
  if (!originalImage) return;
  if (!maskData) {
    /* Pre-filter: skip the server pipeline for obvious non-fence uploads. */
    showLoading("Checking image…");
    updateStatus("Checking image…", "loading");
    const verdict = await _checkIsWoodenFence(originalImage, {
      onProgress: (p) => {
        if (
          p &&
          p.status === "progress" &&
          typeof p.progress === "number"
        ) {
          const pct = Math.max(0, Math.min(100, p.progress)).toFixed(0);
          const msg = `Preparing fence detector… ${pct}%`;
          /* Route to both the visible in-canvas overlay and the hidden
           * a11y status node so screen readers and visual users both
           * see progress. */
          showLoading(msg);
          updateStatus(msg, "loading");
        }
      },
    });
    if (!verdict.isFence) {
      hideLoading();
      updateStatus(
        "No fence detected in this photo. Please upload a clear photo of a wooden fence.",
        "error",
      );
      return;
    }
    await detectFence();
    if (!maskData) return;
  }

  showLoading("Cleaning fence...");
  updateStatus("Cleaning fence...", "loading");

  try {
    await new Promise((resolve) => setTimeout(resolve, 50));

    const W = originalImage.width;
    const H = originalImage.height;

    // Cap working resolution for the heavy per-pixel + guided-filter work.
    // The guided filter is O(W2*H2) per pass and we run 6 of them — at full
    // 5000x4000 res this dominates wall time. Algorithm runs at W2 x H2, then
    // the cleaned RGB is bilinear-upsampled and mask-gated into the original
    // photo so non-fence pixels stay at source resolution.
    const maxCleanDim = CONFIG.CLEAN_MAX_DIM || 1536;
    const cleanScale = Math.min(1, maxCleanDim / Math.max(W, H));
    const W2 = Math.max(1, Math.round(W * cleanScale));
    const H2 = Math.max(1, Math.round(H * cleanScale));
    const cleanDownscaled = W2 !== W || H2 !== H;

    // Destination canvas always at source resolution. Draw the original first
    // so non-fence pixels are correct without any extra compositing later.
    resultCanvas.width = W;
    resultCanvas.height = H;
    const ctx = resultCanvas.getContext("2d");
    ctx.drawImage(originalImage, 0, 0);

    // Work pixel buffer + work-res mask. `data` and `maskWork` are what the
    // algorithm operates on — at W2 x H2 (which may equal W x H).
    let data;
    let maskWork;
    let workCanvas = null;
    if (cleanDownscaled) {
      workCanvas = document.createElement("canvas");
      workCanvas.width = W2;
      workCanvas.height = H2;
      const wctx = workCanvas.getContext("2d");
      wctx.imageSmoothingEnabled = true;
      wctx.imageSmoothingQuality = "high";
      wctx.drawImage(originalImage, 0, 0, W2, H2);
      data = wctx.getImageData(0, 0, W2, H2).data;
      maskWork = bilinearResize(maskData, W, H, W2, H2);
    } else {
      // Full-res path — read the destination canvas back as-is.
      data = ctx.getImageData(0, 0, W, H).data;
      maskWork = maskData;
    }

    const useBinarized = CONFIG.RECOLOR_BINARIZE_MASK !== false;
    const fullAlphaThr =
      CONFIG.RECOLOR_FULL_ALPHA_THRESHOLD != null
        ? CONFIG.RECOLOR_FULL_ALPHA_THRESHOLD
        : 0.15;
    const fenceAlpha = useBinarized
      ? deriveRecolorAlpha(maskWork, fullAlphaThr)
      : maskWork;

    const userOpacity = parseInt(opacity.value) / 100;

    const cfg = CLEAN_CFG;

    // Pre-compute Lab for ALL pixels (regardless of mask) via batched WASM
    // when available -- this single call replaces millions of per-pixel
    // _cleanRgbToLab JS calls across the two builder loops below + the
    // main loop's fallback. ~50x faster on the conversion work alone
    // when SIMD WASM is active.
    let allLab = null;
    if (_WASM.status === "ready") {
      try {
        allLab = _wasmRgbaToLab(data, maskWork.length);
      } catch (e) {
        console.warn("[fsv] WASM rgba->lab failed, fallback to JS per-pixel:", e);
        _WASM.status = "unavailable";
        allLab = null;
      }
    }
    const _labAt = allLab
      ? (i) => [allLab[i * 3], allLab[i * 3 + 1], allLab[i * 3 + 2]]
      : (i) => {
          const idx = i * 4;
          return _cleanRgbToLab(data[idx], data[idx + 1], data[idx + 2]);
        };

    const fenceLs = [];
    const fenceLabs = [];
    for (let i = 0; i < maskWork.length; i++) {
      if (maskWork[i] > 0.5) {
        const lab = _labAt(i);
        fenceLs.push(lab[0]);
        fenceLabs.push(lab);
      }
    }
    let refL = 55,
      refA = cfg.targetA,
      refB = cfg.targetB;
    let usingDerivedRef = false;
    if (fenceLabs.length >= 100) {
      const sortedLs = [...fenceLs].sort((a, b) => b - a);
      const topN = Math.max(50, Math.floor(sortedLs.length * 0.35));
      const lThr = sortedLs[Math.min(topN - 1, sortedLs.length - 1)];
      let sL = 0,
        sA = 0,
        sB = 0,
        cnt = 0;
      for (const lab of fenceLabs) {
        if (lab[0] >= lThr) {
          sL += lab[0];
          sA += lab[1];
          sB += lab[2];
          cnt++;
        }
      }
      if (cnt > 0) {
        const candL = sL / cnt;
        const candA = sA / cnt;
        const candB = sB / cnt;
        const candChroma = Math.sqrt(candA * candA + candB * candB);

        if (candL >= 35 && candChroma >= 6) {
          refL = candL;
          refA = candA;
          refB = candB;
          usingDerivedRef = true;
        }
      }
    }

    let _modified = 0,
      _skippedHighlight = 0;

    const Lbuf = new Float32Array(W2 * H2);
    const aBuf = new Float32Array(W2 * H2);
    const bBuf = new Float32Array(W2 * H2);
    const mBuf = new Float32Array(W2 * H2);
    if (allLab) {
      // WASM batched path: read directly from the pre-computed Lab array.
      for (let i = 0; i < maskWork.length; i++) {
        if (fenceAlpha[i] > 0) {
          Lbuf[i] = allLab[i * 3];
          aBuf[i] = allLab[i * 3 + 1];
          bBuf[i] = allLab[i * 3 + 2];
          mBuf[i] = 1;
        }
      }
    } else {
      for (let i = 0; i < maskWork.length; i++) {
        if (fenceAlpha[i] > 0) {
          const idx = i * 4;
          const lab = _cleanRgbToLab(data[idx], data[idx + 1], data[idx + 2]);
          Lbuf[i] = lab[0];
          aBuf[i] = lab[1];
          bBuf[i] = lab[2];
          mBuf[i] = 1;
        }
      }
    }

    const blurRadius = Math.max(24, Math.floor(Math.min(W2, H2) / 30));
    const smallRadius = Math.max(5, Math.floor(blurRadius / 5));
    const plankWidth = Math.max(18, Math.floor(W2 / 40));
    const featherR = Math.max(2, Math.floor(blurRadius / 8));
    const sharpFactor = 1.15;

    const eps = 150.0;
    // Multi-channel guided filter: meanG and meanGG are guide-only stats
    // (same across L/a/b passes), so we compute them once per radius and
    // reuse. Output is bit-identical to 6 separate _guidedFilter calls.
    const [Lblur, aBlur, bBlur] = _guidedFilterMulti3(
      Lbuf, aBuf, bBuf, Lbuf, mBuf, W2, H2, blurRadius, eps,
    );
    const [Lsmall, aSmall, bSmall] = _guidedFilterMulti3(
      Lbuf, aBuf, bBuf, Lbuf, mBuf, W2, H2, smallRadius, eps,
    );

    const onesMask = new Float32Array(W2 * H2).fill(1);
    const alphaBlur = maskedBoxBlur(fenceAlpha, onesMask, W2, H2, featherR);
    const alphaSmooth = new Float32Array(W2 * H2);
    for (let i = 0; i < alphaSmooth.length; i++) {
      alphaSmooth[i] = Math.min(alphaBlur[i], fenceAlpha[i]);
    }

    const refChromaMag = Math.sqrt(refA * refA + refB * refB);
    const chromaCeiling = Math.max(cfg.chromaCeiling, refChromaMag);

    let _sumL = 0,
      _cntL = 0;
    for (let i = 0; i < Lbuf.length; i++) {
      if (mBuf[i] > 0) {
        _sumL += Lbuf[i];
        _cntL++;
      }
    }
    const meanFenceL = _cntL > 0 ? _sumL / _cntL : refL;
    const lShift = refL - meanFenceL;

    const lShiftCapped = Math.max(-15, Math.min(35, lShift));

    // Two-phase main loop when WASM is active:
    //   Phase 1 (per-pixel arithmetic): compute newLab[i*3+0..2], store
    //     the "touched" set in `touched`, and skip highlight/empty pixels.
    //   Phase 2 (batched Lab->RGB conversion via WASM): _wasmLabToRgba
    //     converts all newLab in one call, returning RGB bytes per pixel.
    //   Phase 3 (per-pixel composite): blend cleaned RGB into `data`
    //     using the per-pixel alphaSmooth.
    //
    // When WASM is unavailable, we fall back to the single-pass JS loop
    // that does conversion + composite inline (bit-identical output).
    if (_WASM.status === "ready") {
      const N = maskWork.length;
      const newLab = new Float32Array(N * 3);
      const touched = new Uint8Array(N);
      for (let i = 0; i < N; i++) {
        if (fenceAlpha[i] <= 0 && alphaSmooth[i] <= 0.02) continue;
        const idx = i * 4;
        const r = data[idx], g = data[idx + 1], b = data[idx + 2];
        const Lstar = Lbuf[i] || (allLab ? allLab[i * 3] : _cleanRgbToLab(r, g, b)[0]);
        if (Lstar > cfg.highlightLStar) {
          _skippedHighlight++;
          continue;
        }
        const fineL = Lbuf[i] - Lsmall[i];
        const fineA = aBuf[i] - aSmall[i];
        const fineB = bBuf[i] - bSmall[i];
        const plankA = aSmall[i] - aBlur[i];
        const plankB = bSmall[i] - bBlur[i];
        const xCol = i % W2;
        const [jL, jA, jB] = _plankJitter(xCol, plankWidth);
        let newL = Lbuf[i] + lShiftCapped + jL + fineL * (sharpFactor - 1);
        let newA = refA + jA + plankA + fineA * sharpFactor;
        let newB = refB + jB + plankB + fineB * sharpFactor;
        const outChroma = Math.sqrt(newA * newA + newB * newB);
        if (outChroma > chromaCeiling) {
          const k = chromaCeiling / outChroma;
          newA *= k;
          newB *= k;
        }
        newL = Math.max(0, Math.min(100, newL));
        newLab[i * 3] = newL;
        newLab[i * 3 + 1] = newA;
        newLab[i * 3 + 2] = newB;
        touched[i] = 1;
      }
      // Phase 2: batched Lab->RGB via WASM. Use a scratch RGBA buffer for
      // the new colors; we copy from it during the composite phase.
      const newRgba = new Uint8ClampedArray(N * 4);
      try {
        _wasmLabToRgba(newLab, newRgba, N);
        // Phase 3: composite. Try WASM-SIMD batched alpha blend first.
        // Build a per-pixel effective-alpha array: alphaSmooth[i] where
        // touched, 0 elsewhere (so untouched pixels get the original
        // unchanged).
        let blendOk = false;
        try {
          const blendAlpha = new Float32Array(N);
          for (let i = 0; i < N; i++) {
            blendAlpha[i] = touched[i] ? alphaSmooth[i] : 0;
          }
          /* Snapshot original `data` before writing -- the WASM blend
           * reads orig[i] before writing out[i] but we pass `data` as
           * both. To avoid in-place hazards we snapshot. */
          const origCopy = new Uint8ClampedArray(data);
          _wasmAlphaBlendRgba(origCopy, newRgba, blendAlpha, data, N);
          /* Count touched pixels as modified -- semantically equivalent
           * to the per-pixel _modified++ in the fallback below. */
          for (let i = 0; i < N; i++) if (touched[i]) _modified++;
          blendOk = true;
        } catch (e) {
          console.warn("[fsv] WASM alpha blend failed, fallback to per-pixel JS composite:", e);
          /* Mark WASM unavailable so subsequent calls skip it. */
          _WASM.status = "unavailable";
        }
        if (!blendOk) {
          for (let i = 0; i < N; i++) {
            if (!touched[i]) continue;
            const idx = i * 4;
            const finalAlpha = alphaSmooth[i];
            const r = data[idx], g = data[idx + 1], b = data[idx + 2];
            data[idx]     = r + (newRgba[idx]     - r) * finalAlpha;
            data[idx + 1] = g + (newRgba[idx + 1] - g) * finalAlpha;
            data[idx + 2] = b + (newRgba[idx + 2] - b) * finalAlpha;
            _modified++;
          }
        }
      } catch (e) {
        console.warn("[fsv] WASM lab->rgba failed, fallback per-pixel JS:", e);
        _WASM.status = "unavailable";
        // Fallback re-runs everything inline (cheap relative to the work
        // we just did; rare edge case).
        for (let i = 0; i < N; i++) {
          if (!touched[i]) continue;
          const idx = i * 4;
          const [nr, ng, nb] = _cleanLabToRgb(newLab[i * 3], newLab[i * 3 + 1], newLab[i * 3 + 2]);
          const finalAlpha = alphaSmooth[i];
          const r = data[idx], g = data[idx + 1], b = data[idx + 2];
          data[idx]     = r + (nr - r) * finalAlpha;
          data[idx + 1] = g + (ng - g) * finalAlpha;
          data[idx + 2] = b + (nb - b) * finalAlpha;
          _modified++;
        }
      }
    } else {
      // Pure-JS path (unchanged behavior).
      for (let i = 0; i < maskWork.length; i++) {
        if (fenceAlpha[i] <= 0 && alphaSmooth[i] <= 0.02) continue;
        const idx = i * 4;
        const r = data[idx], g = data[idx + 1], b = data[idx + 2];
        const Lstar = Lbuf[i] || _cleanRgbToLab(r, g, b)[0];
        if (Lstar > cfg.highlightLStar) {
          _skippedHighlight++;
          continue;
        }
        const fineL = Lbuf[i] - Lsmall[i];
        const fineA = aBuf[i] - aSmall[i];
        const fineB = bBuf[i] - bSmall[i];
        const plankA = aSmall[i] - aBlur[i];
        const plankB = bSmall[i] - bBlur[i];
        const xCol = i % W2;
        const [jL, jA, jB] = _plankJitter(xCol, plankWidth);
        let newL = Lbuf[i] + lShiftCapped + jL + fineL * (sharpFactor - 1);
        let newA = refA + jA + plankA + fineA * sharpFactor;
        let newB = refB + jB + plankB + fineB * sharpFactor;
        const outChroma = Math.sqrt(newA * newA + newB * newB);
        if (outChroma > chromaCeiling) {
          const k = chromaCeiling / outChroma;
          newA *= k;
          newB *= k;
        }
        newL = Math.max(0, Math.min(100, newL));
        const [nr, ng, nb] = _cleanLabToRgb(newL, newA, newB);
        const finalAlpha = alphaSmooth[i];
        data[idx]     = r + (nr - r) * finalAlpha;
        data[idx + 1] = g + (ng - g) * finalAlpha;
        data[idx + 2] = b + (nb - b) * finalAlpha;
        _modified++;
      }
    }

    if (cleanDownscaled) {
      // Write the cleaned low-res RGB back to the work canvas, then have the
      // browser bilinear-upscale it onto resultCanvas (which already holds
      // the full-res original photo). We do NOT replace the whole canvas —
      // instead use destination-over compositing only on the fence pixels:
      //   1. Build a full-res mask alpha image (white = fence, transparent =
      //      non-fence) from the source-res maskData.
      //   2. Upscale the cleaned work canvas to a full-res offscreen canvas.
      //   3. Apply the mask as a destination-in clip on that upscaled image.
      //   4. drawImage that clipped image onto resultCanvas — non-fence pixels
      //      retain the sharp original.
      workCanvas.getContext("2d").putImageData(
        new ImageData(data, W2, H2),
        0,
        0,
      );

      // Step 1: full-res mask alpha image
      const maskAlphaCanvas = document.createElement("canvas");
      maskAlphaCanvas.width = W;
      maskAlphaCanvas.height = H;
      const mctx = maskAlphaCanvas.getContext("2d");
      const maskImgData = mctx.createImageData(W, H);
      const md = maskImgData.data;
      for (let i = 0; i < maskData.length; i++) {
        // Soft alpha: scale mask to 0-255 with a low-threshold boost so the
        // feathered border (mask 0.02-0.15) gets nearly-full alpha. This keeps
        // the composite seamless even with a soft mask.
        const v = maskData[i];
        const a = v <= 0 ? 0 : v >= 0.15 ? 255 : Math.round((v / 0.15) * 255);
        const p = i * 4;
        md[p] = 255;
        md[p + 1] = 255;
        md[p + 2] = 255;
        md[p + 3] = a;
      }
      mctx.putImageData(maskImgData, 0, 0);

      // Step 2 + 3: upscale cleaned to full res, then mask it via dst-in
      const upCanvas = document.createElement("canvas");
      upCanvas.width = W;
      upCanvas.height = H;
      const uctx = upCanvas.getContext("2d");
      uctx.imageSmoothingEnabled = true;
      uctx.imageSmoothingQuality = "high";
      uctx.drawImage(workCanvas, 0, 0, W, H);
      uctx.globalCompositeOperation = "destination-in";
      uctx.drawImage(maskAlphaCanvas, 0, 0);

      // Step 4: drawImage clipped cleaned onto the original-bearing
      // resultCanvas. Default 'source-over' blends by the cleaned canvas's
      // own alpha (which is the mask we just applied) — fence pixels get
      // replaced, non-fence pixels stay original.
      ctx.drawImage(upCanvas, 0, 0);

      cleanedImageData = ctx.getImageData(0, 0, W, H);
    } else {
      // Full-res path — `data` IS the source-resolution pixel buffer we read
      // from ctx earlier. Wrap it back in an ImageData and write it out.
      ctx.putImageData(new ImageData(data, W, H), 0, 0);
      cleanedImageData = new ImageData(new Uint8ClampedArray(data), W, H);
    }
    setDownloadEnabled(true);
    updateStatus("Cleaning complete!", "success");
    resultState = "cleaned";
    applyResultHeader();
    showCompareButton();
    showCoachTip();
  } finally {
    hideLoading();
  }
}

async function recolorFence({
  loadingMessage = "Recoloring fence...",
  successMessage = "Recoloring complete!",
} = {}) {
  if (!originalImage || !maskData) return;

  showLoading(loadingMessage);
  updateStatus(loadingMessage, "loading");

  try {
    await new Promise((resolve) => setTimeout(resolve, 50));

    resultCanvas.width = originalImage.width;
    resultCanvas.height = originalImage.height;
    const ctx = resultCanvas.getContext("2d");

    let imageData;
    if (
      cleanedImageData &&
      cleanedImageData.width === originalImage.width &&
      cleanedImageData.height === originalImage.height
    ) {
      imageData = new ImageData(
        new Uint8ClampedArray(cleanedImageData.data),
        cleanedImageData.width,
        cleanedImageData.height,
      );
      ctx.putImageData(imageData, 0, 0);
    } else {
      ctx.drawImage(originalImage, 0, 0);
      imageData = ctx.getImageData(
        0,
        0,
        originalImage.width,
        originalImage.height,
      );
    }
    const data = imageData.data;

    const rgb = hexToRgb(selectedColor);
    const alpha = parseInt(opacity.value) / 100;
    const mode = blendMode.value;

    const useBinarized = CONFIG.RECOLOR_BINARIZE_MASK !== false;
    const fullAlphaThr =
      CONFIG.RECOLOR_FULL_ALPHA_THRESHOLD != null
        ? CONFIG.RECOLOR_FULL_ALPHA_THRESHOLD
        : 0.15;
    const recolorAlpha = useBinarized
      ? deriveRecolorAlpha(maskData, fullAlphaThr)
      : maskData;

    // Optimizations: hoist constant blend params (rgb.r/g/b), replace
    // Math.max/min with branchless conditionals, cache pixel-channel reads,
    // skip the function-call overhead for the "blend" call entirely for
    // the smart-blend hot path. Bit-identical output.
    const rgbR = rgb.r, rgbG = rgb.g, rgbB = rgb.b;
    const maskLen = maskData.length;

    let _lSum = 0, _sSum = 0, _count = 0;
    for (let i = 0; i < maskLen; i++) {
      if (maskData[i] > 0.5) {
        const idx = i * 4;
        const r = data[idx], g = data[idx + 1], b = data[idx + 2];
        const mx = r > g ? (r > b ? r : b) : (g > b ? g : b);
        const mn = r < g ? (r < b ? r : b) : (g < b ? g : b);
        _lSum += (mx + mn) * 0.5;
        _sSum += mx > 0 ? (mx - mn) / mx : 0;
        _count++;
      }
    }
    const fenceMeanL = _count > 0 ? _lSum / _count / 255 : 0.5;
    const fenceMeanS = _count > 0 ? _sSum / _count : 0.3;

    for (let i = 0; i < maskLen; i++) {
      const maskVal = recolorAlpha[i];
      if (maskVal > 0) {
        const idx = i * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];

        const blended = applyBlendMode(
          r, g, b, rgbR, rgbG, rgbB, mode, fenceMeanL, fenceMeanS,
        );

        const finalAlpha = alpha * maskVal;
        data[idx] = r + (blended.r - r) * finalAlpha;
        data[idx + 1] = g + (blended.g - g) * finalAlpha;
        data[idx + 2] = b + (blended.b - b) * finalAlpha;
      }
    }

    ctx.putImageData(imageData, 0, 0);
    setDownloadEnabled(true);
    updateStatus(successMessage, "success");

    resultState = cleanedImageData ? "cleaned_stained" : "stained";
    applyResultHeader();
    showCompareButton();
    showCoachTip();
  } finally {
    hideLoading();
  }
}

function _rgbToHsl(r, g, b) {
  r /= 255;
  g /= 255;
  b /= 255;
  const max = Math.max(r, g, b),
    min = Math.min(r, g, b);
  const l = (max + min) / 2;
  let h, s;
  if (max === min) {
    h = 0;
    s = 0;
  } else {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (max === g) h = ((b - r) / d + 2) / 6;
    else h = ((r - g) / d + 4) / 6;
  }
  return [h, s, l];
}
function _hslToRgb(h, s, l) {
  if (s === 0) {
    const v = l * 255;
    return [v, v, v];
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 0.5) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  return [
    hue2rgb(p, q, h + 1 / 3) * 255,
    hue2rgb(p, q, h) * 255,
    hue2rgb(p, q, h - 1 / 3) * 255,
  ];
}

function applyBlendMode(r, g, b, cr, cg, cb, mode, fenceMeanL, fenceMeanS) {
  const meanL = fenceMeanL != null ? fenceMeanL : 0.5;
  const meanS = fenceMeanS != null ? fenceMeanS : 0.3;
  switch (mode) {
    case "smart": {
      const baseHSL = _rgbToHsl(r, g, b);
      const tgtHSL = _rgbToHsl(cr, cg, cb);
      const baseL = baseHSL[2];
      const baseS = baseHSL[1];
      const tgtL = tgtHSL[2];
      const tgtS = tgtHSL[1];
      const texRangeL = 0.75;
      const texRangeS = 0.55;
      let newL = tgtL + (baseL - meanL) * texRangeL;
      let newS = tgtS + (baseS - meanS) * texRangeS;
      newL = Math.max(0, Math.min(1, newL));
      newS = Math.max(0, Math.min(1, newS));
      const [nr, ng, nb] = _hslToRgb(tgtHSL[0], newS, newL);
      return { r: nr, g: ng, b: nb };
    }
    case "multiply":
      return {
        r: (r / 255) * (cr / 255) * 255,
        g: (g / 255) * (cg / 255) * 255,
        b: (b / 255) * (cb / 255) * 255,
      };
    case "overlay":
      return {
        r:
          r < 128
            ? (2 * r * cr) / 255
            : 255 - (2 * (255 - r) * (255 - cr)) / 255,
        g:
          g < 128
            ? (2 * g * cg) / 255
            : 255 - (2 * (255 - g) * (255 - cg)) / 255,
        b:
          b < 128
            ? (2 * b * cb) / 255
            : 255 - (2 * (255 - b) * (255 - cb)) / 255,
      };
    case "screen":
      return {
        r: 255 - ((255 - r) * (255 - cr)) / 255,
        g: 255 - ((255 - g) * (255 - cg)) / 255,
        b: 255 - ((255 - b) * (255 - cb)) / 255,
      };
    case "color":
      return { r: cr, g: cg, b: cb };
    default:
      return { r, g, b };
  }
}

function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : { r: 139, g: 69, b: 19 };
}

async function downloadResult() {
  showLoading("Preparing download...");

  try {
    await new Promise((resolve) => setTimeout(resolve, 300));
    const link = document.createElement("a");

    const FNAME = {
      original: "fence_original.jpg",
      cleaned: "fence_cleaned.jpg",
      stained: "fence_stained.jpg",
      cleaned_stained: "fence_cleaned_and_stained.jpg",
    };
    link.download = FNAME[resultState] || "fence_result.jpg";
    // JPEG @ 0.92 keeps file size roughly comparable to typical phone-camera
    // uploads (2-4 MB on a 5000x4000 canvas) and is visually indistinguishable
    // from lossless PNG, which on the same canvas runs 20-30 MB.
    link.href = resultCanvas.toDataURL("image/jpeg", 0.92);
    link.click();
    updateStatus("Downloaded!", "success");
  } finally {
    hideLoading();
  }
}

async function reset() {
  showLoading("Resetting...");

  try {
    await new Promise((resolve) => setTimeout(resolve, 200));

    originalImage = null;
    maskData = null;
    cleanedImageData = null;
    resultState = "original";

    const ctx1 = originalCanvas.getContext("2d");
    const ctx2 = maskCanvas.getContext("2d");
    const ctx3 = resultCanvas.getContext("2d");

    ctx1.clearRect(0, 0, originalCanvas.width, originalCanvas.height);
    ctx2.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    ctx3.clearRect(0, 0, resultCanvas.width, resultCanvas.height);

    originalCanvas.width = 800;
    originalCanvas.height = 620;
    maskCanvas.width = 800;
    maskCanvas.height = 620;
    resultCanvas.width = 800;
    resultCanvas.height = 620;

    ctx1.clearRect(0, 0, originalCanvas.width, originalCanvas.height);
    ctx2.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    ctx3.clearRect(0, 0, resultCanvas.width, resultCanvas.height);

    detectBtn.disabled = true;
    cleanBtn.disabled = true;
    recolorBtn.disabled = true;
    setDownloadEnabled(false);
    resetBtn.disabled = true;
    setColorChipsEnabled(false);
    showEmptyState();
    fileInput.value = "";

    compareBtn.hidden = true;
    hideCompareUI(true);
    canvasStack.classList.remove("has-image");
    if (canvasLabel) canvasLabel.textContent = "Upload a photo";
    if (canvasTitleText) canvasTitleText.textContent = "Fence Preview";
    updateStatus(
      modalReady ? "Simulator ready" : "Simulator not ready",
      modalReady ? "success" : "error",
    );
  } finally {
    hideLoading();
  }
}

const toastContainer = document.getElementById("toast-container");

function toast(message, type = "info", durationMs) {
  if (!toastContainer || !message) return;
  if (!durationMs) durationMs = type === "error" ? 5500 : 3500;

  const el = document.createElement("div");
  el.className = "toast toast-" + type;
  el.setAttribute("role", type === "error" ? "alert" : "status");

  const iconHTML =
    type === "success"
      ? '<i class="bi bi-check-lg"></i>'
      : type === "error"
        ? '<i class="bi bi-exclamation-lg"></i>'
        : '<i class="bi bi-info-circle"></i>';
  el.innerHTML =
    '<span class="toast-icon" aria-hidden="true">' +
    iconHTML +
    "</span>" +
    '<span class="toast-msg"></span>';
  el.querySelector(".toast-msg").textContent = message;

  let dismissed = false;
  const dismiss = () => {
    if (dismissed) return;
    dismissed = true;
    el.classList.add("fading");
    setTimeout(() => el.remove(), 340);
  };
  el.addEventListener("click", dismiss);
  toastContainer.appendChild(el);
  setTimeout(dismiss, durationMs);
}

function updateStatus(message, type = "", opts) {
  if (status) {
    status.textContent = message;
    status.className = "status " + type;
  }
  if (opts && opts.silent) return;
  if (type === "success" || type === "error") {
    toast(message, type);
  }
}

  // Eagerly initialize — the loader calls initFenceSimulator(shadowRoot) after
  // the shadow DOM has been populated, so we don't need to wait for window.load.
  init();
  initColorPicker();
}

window.FSV_initFenceSimulator = initFenceSimulator;
