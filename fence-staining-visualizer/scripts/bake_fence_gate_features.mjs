#!/usr/bin/env node
/**
 * bake_fence_gate_features.mjs
 *
 * Offline-precompute the SigLIP text embeddings for the fence-gate
 * classifier. Run once whenever the prompt arrays in
 * `index4_dinov3.html` change.
 *
 * Why this exists:
 *   The browser fence-gate (in index4_dinov3.html + wordpress/app.js)
 *   needs ~36 s on a first visit to run SigLIP's text encoder over all
 *   ~56 prompts. Pre-computing the embeddings offline, then serving
 *   them as a static ~170 KB file, drops the gate-ready time to
 *   ~50 ms on EVERY visit (first or repeat).
 *
 * How:
 *   1. Reads the canonical prompt arrays from
 *      `../index4_dinov3.html` (single source of truth).
 *   2. Loads the same SigLIP model the browser uses
 *      (Xenova/siglip-base-patch16-224, quantized).
 *   3. Calls the unified model with (real_text_inputs + dummy_pixel_
 *      values), extracts text_embeds, L2-normalizes.
 *   4. Writes the embeddings as a raw Float32Array .bin file plus a
 *      .json sidecar with version + dims + prompts-hash metadata.
 *      Both files are copied to the standalone HTML directory AND
 *      the WordPress plugin directory so each deployment can serve
 *      them relative to its own JS.
 *
 * Outputs:
 *   ../fence_gate_text_features_v1.bin
 *   ../fence_gate_text_features_v1.json
 *   ../wordpress/fence_gate_text_features_v1.bin
 *   ../wordpress/fence_gate_text_features_v1.json
 *
 * Run:
 *   cd fence-staining-visualizer/scripts
 *   npm install
 *   npm run bake
 *
 * Re-run when:
 *   - You change _FENCE_POSITIVE_PROMPTS or _FENCE_NEGATIVE_PROMPTS
 *     in index4_dinov3.html. The client validates the prompt-hash in
 *     the .json sidecar against its in-code prompts and falls back to
 *     in-browser compute if there's a mismatch — but you want a hit
 *     in production, so re-bake.
 *   - You change the MODEL_ID below.
 *   - You bump the version (bump VERSION here AND in both files'
 *     client loaders).
 */

import {
    AutoTokenizer,
    AutoModel,
    Tensor,
    env,
} from "@xenova/transformers";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(__dirname, "..");

const VERSION = 1;
const MODEL_ID = "Xenova/siglip-base-patch16-224";
const QUANTIZED = true;
const SOURCE_HTML = resolve(REPO, "index4_dinov3.html");

const BIN_NAME = `fence_gate_text_features_v${VERSION}.bin`;
const JSON_NAME = `fence_gate_text_features_v${VERSION}.json`;
const OUTPUT_DIRS = [REPO, resolve(REPO, "wordpress")];

// ───────────────────────────────────────────────────────────────────────
// 1. Extract prompts from the canonical source (index4_dinov3.html).
// ───────────────────────────────────────────────────────────────────────

function extractPrompts(src, varName) {
    const re = new RegExp(`const ${varName} = \\[([\\s\\S]*?)\\];`);
    const m = src.match(re);
    if (!m) throw new Error(`${varName} not found in source`);
    /* Strip /* ... *​/ and // ... comments from the array body BEFORE
     * extracting string literals, so any example strings inside the
     * explanatory comments aren't counted as prompts. */
    const body = m[1]
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/\/\/.*$/gm, "");
    const items = Array.from(body.matchAll(/"((?:[^"\\]|\\.)*)"/g)).map(
        (x) => x[1],
    );
    if (items.length === 0) {
        throw new Error(`${varName} parsed to 0 items — regex bug?`);
    }
    return items;
}

console.log(`Reading prompts from ${SOURCE_HTML}`);
const html = readFileSync(SOURCE_HTML, "utf8");
const positives = extractPrompts(html, "_FENCE_POSITIVE_PROMPTS");
const negatives = extractPrompts(html, "_FENCE_NEGATIVE_PROMPTS");
const allPrompts = [...positives, ...negatives];
console.log(
    `Found ${positives.length} positive + ${negatives.length} negative = ${allPrompts.length} prompts`,
);

const promptsHash = createHash("sha256")
    .update(JSON.stringify(allPrompts))
    .digest("hex");
console.log(`Prompts SHA-256: ${promptsHash}`);

// ───────────────────────────────────────────────────────────────────────
// 2. Load SigLIP via the same runtime the browser uses.
// ───────────────────────────────────────────────────────────────────────

env.allowLocalModels = false;
env.useBrowserCache = false; // Node has no IndexedDB.
/* Scope the model cache to this script directory so a half-finished
 * download from a previous run is easy to nuke (just delete the
 * folder). Default in transformers.js Node is `./.cache/` relative
 * to CWD, which is too easy to share with other tools. */
env.cacheDir = resolve(__dirname, ".transformers_cache");

/* HuggingFace CDN occasionally drops connections mid-stream
 * (ECONNRESET). Wrap the model + tokenizer load in retries with
 * exponential backoff so a single network hiccup doesn't kill the
 * bake. The downloaded bytes that DID complete are cached on disk,
 * so retried loads only re-fetch what was missing. */
async function withRetries(fn, label, maxRetries = 4) {
    let lastErr;
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return await fn();
        } catch (e) {
            lastErr = e;
            const msg = e?.cause?.message || e?.message || String(e);
            console.warn(
                `  [retry] ${label} attempt ${attempt}/${maxRetries} failed: ${msg}`,
            );
            if (attempt < maxRetries) {
                const delaySec = attempt * 2;
                console.warn(`  [retry] waiting ${delaySec}s before retry...`);
                await new Promise((r) => setTimeout(r, delaySec * 1000));
            }
        }
    }
    throw lastErr;
}

console.log(`\nLoading ${MODEL_ID} (quantized=${QUANTIZED})...`);
console.log(`Cache dir: ${env.cacheDir}`);
const tokenizer = await withRetries(
    () => AutoTokenizer.from_pretrained(MODEL_ID),
    "tokenizer load",
);
const model = await withRetries(
    () => AutoModel.from_pretrained(MODEL_ID, { quantized: QUANTIZED }),
    "model load",
);

// ───────────────────────────────────────────────────────────────────────
// 3. Tokenize prompts + run forward pass with dummy pixel_values.
// ───────────────────────────────────────────────────────────────────────

console.log("Tokenizing prompts...");
const textInputs = tokenizer(allPrompts, {
    padding: "max_length",
    truncation: true,
});

console.log("Running forward pass to extract text_embeds...");
const dummyPixels = new Float32Array(1 * 3 * 224 * 224);
const dummyPixelValues = new Tensor("float32", dummyPixels, [1, 3, 224, 224]);

const t0 = Date.now();
const out = await model({
    input_ids: textInputs.input_ids,
    attention_mask: textInputs.attention_mask,
    pixel_values: dummyPixelValues,
});
const forwardMs = Date.now() - t0;
console.log(`Forward pass: ${(forwardMs / 1000).toFixed(1)} s`);

if (!out.text_embeds) {
    throw new Error(
        `output has no text_embeds. available keys: ${Object.keys(out).join(",")}`,
    );
}

const textEmbeds = out.text_embeds;
const rawData = textEmbeds.data;
const dims = Array.from(textEmbeds.dims);
const N = dims[0];
const D = dims[1];
console.log(
    `Text features: ${dims.join("×")} float32 (${rawData.length} floats, ${(rawData.byteLength / 1024).toFixed(1)} KB)`,
);

/* L2-normalize each row manually. transformers.js v2's Tensor wrapper
 * in Node doesn't expose .normalize(), and using .data + a JS loop
 * is bit-identical to PyTorch's F.normalize(x, p=2, dim=-1). */
const data = new Float32Array(N * D);
for (let i = 0; i < N; i++) {
    const offset = i * D;
    let sumSq = 0;
    for (let d = 0; d < D; d++) {
        const v = rawData[offset + d];
        sumSq += v * v;
    }
    const norm = Math.sqrt(sumSq);
    const inv = norm > 0 ? 1 / norm : 0;
    for (let d = 0; d < D; d++) {
        data[offset + d] = rawData[offset + d] * inv;
    }
}

// Sanity check — first row should have L2 norm ≈ 1.0 after normalize.
let norm0 = 0;
for (let d = 0; d < D; d++) norm0 += data[d] * data[d];
console.log(`First-row L2 norm: ${Math.sqrt(norm0).toFixed(6)} (should be ≈ 1.0)`);

// ───────────────────────────────────────────────────────────────────────
// 4. Write outputs (binary + metadata sidecar) to both deployment dirs.
// ───────────────────────────────────────────────────────────────────────

const metadata = {
    version: VERSION,
    model: MODEL_ID,
    quantized: QUANTIZED,
    promptCount: allPrompts.length,
    embeddingDim: dims[1],
    dims: dims,
    promptsHash: promptsHash,
    bakedAt: new Date().toISOString(),
};

const binBuffer = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
const jsonText = JSON.stringify(metadata, null, 2);

for (const dir of OUTPUT_DIRS) {
    mkdirSync(dir, { recursive: true });
    const binPath = resolve(dir, BIN_NAME);
    const jsonPath = resolve(dir, JSON_NAME);
    writeFileSync(binPath, binBuffer);
    writeFileSync(jsonPath, jsonText);
    console.log(`Wrote ${binPath}`);
    console.log(`Wrote ${jsonPath}`);
}

console.log(
    `\nDone. The fence-gate will load these files at runtime instead of running the text encoder.`,
);
