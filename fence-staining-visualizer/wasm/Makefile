# ============================================================================
# fsv_postprocess WASM build
#
# Produces two .wasm artifacts:
#   1. fsv_postprocess.wasm        - portable scalar build (universal support)
#   2. fsv_postprocess_simd.wasm   - SIMD128 build (~2-3x faster where available)
#
# The JS-side loader feature-detects SIMD support via WebAssembly.validate()
# with a small probe module, then loads the appropriate .wasm.
#
# Prerequisites
# -------------
#   - Emscripten SDK (https://emscripten.org/docs/getting_started/downloads.html)
#       git clone https://github.com/emscripten-core/emsdk.git
#       cd emsdk && ./emsdk install latest && ./emsdk activate latest
#       source ./emsdk_env.sh
#   - GNU Make
#
# Build
# -----
#   make           - build both scalar and SIMD .wasm files
#   make scalar    - build only the scalar variant
#   make simd      - build only the SIMD variant
#   make clean     - remove build artifacts
#   make deploy    - copy built .wasm into the JS hosting directories
#
# Output paths (in addition to ./build/):
#   fsv_postprocess.wasm and fsv_postprocess_simd.wasm
#   These need to be hosted alongside the JS at runtime. The deploy target
#   copies them into:
#     ../wordpress/        (WordPress plugin folder; auto-included on plugin upload)
#     ../                  (root of fence-staining-visualizer/ for the standalone HTML)
# ============================================================================

EMCC ?= emcc
CFLAGS_COMMON = -O3 -flto -fno-rtti -fno-exceptions
# --profiling-funcs preserves function symbol names in the .wasm so our
# manual loader can look up exports by their original C names (e.g.
# `fsv_alloc`) instead of post-link-minified short names (e.g. `a`).
# Import module names may still be minified by emcc; the JS loader uses
# WebAssembly.Module.imports() to inspect and stub whatever names emcc
# emits, so no compile-time flag is needed for that side.
EMFLAGS_COMMON = \
    --profiling-funcs \
    --emit-symbol-map \
    -sEXPORTED_FUNCTIONS=_fsv_alloc,_fsv_free,_fsv_srgb_lut_init,_fsv_dilate,_fsv_erode,_fsv_box_blur_masked,_fsv_rgba_to_lab,_fsv_lab_to_rgba,_fsv_pixelwise_diff,_fsv_pixelwise_mul,_fsv_guided_filter,_fsv_alpha_blend_rgba,_fsv_soft_mask_threshold,_malloc,_free \
    -sENVIRONMENT=web \
    -sFILESYSTEM=0 \
    -sASSERTIONS=0 \
    -sALLOW_MEMORY_GROWTH=1 \
    -sINITIAL_MEMORY=268435456 \
    -sMAXIMUM_MEMORY=536870912 \
    -sSTACK_SIZE=1048576 \
    -sMALLOC=emmalloc \
    --no-entry

# WASM-only output (no glue JS) so we can integrate via our own loader.
EMFLAGS_BARE = $(EMFLAGS_COMMON) -sSINGLE_FILE=0 -sWASM=1 -sSTANDALONE_WASM=0

SRC = fsv_postprocess.c
BUILD_DIR = build

all: scalar simd

scalar: $(BUILD_DIR)/fsv_postprocess.wasm
simd:   $(BUILD_DIR)/fsv_postprocess_simd.wasm

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Scalar build: portable across all browsers with WebAssembly support.
$(BUILD_DIR)/fsv_postprocess.wasm: $(SRC) | $(BUILD_DIR)
	$(EMCC) $(CFLAGS_COMMON) $(EMFLAGS_BARE) \
	    -o $(BUILD_DIR)/fsv_postprocess.js $(SRC)

# SIMD build: requires v128 support (Chrome 91+, Firefox 89+, Safari 16.4+).
# We pass a distinct -o basename so emcc writes fsv_postprocess_simd.wasm
# directly; no post-build rename needed.
$(BUILD_DIR)/fsv_postprocess_simd.wasm: $(SRC) | $(BUILD_DIR)
	$(EMCC) $(CFLAGS_COMMON) -msimd128 $(EMFLAGS_BARE) \
	    -o $(BUILD_DIR)/fsv_postprocess_simd.js $(SRC)

deploy: all
	cmd /c copy /Y $(subst /,\,$(BUILD_DIR)/fsv_postprocess.wasm)         ..\wordpress\fsv_postprocess.wasm
	cmd /c copy /Y $(subst /,\,$(BUILD_DIR)/fsv_postprocess_simd.wasm)    ..\wordpress\fsv_postprocess_simd.wasm
	cmd /c copy /Y $(subst /,\,$(BUILD_DIR)/fsv_postprocess.wasm)         ..\fsv_postprocess.wasm
	cmd /c copy /Y $(subst /,\,$(BUILD_DIR)/fsv_postprocess_simd.wasm)    ..\fsv_postprocess_simd.wasm
	cmd /c copy /Y $(subst /,\,$(BUILD_DIR)/fsv_postprocess.js.symbols)      ..\wordpress\fsv_postprocess.wasm.symbols
	cmd /c copy /Y $(subst /,\,$(BUILD_DIR)/fsv_postprocess_simd.js.symbols) ..\wordpress\fsv_postprocess_simd.wasm.symbols
	cmd /c copy /Y $(subst /,\,$(BUILD_DIR)/fsv_postprocess.js.symbols)      ..\fsv_postprocess.wasm.symbols
	cmd /c copy /Y $(subst /,\,$(BUILD_DIR)/fsv_postprocess_simd.js.symbols) ..\fsv_postprocess_simd.wasm.symbols

clean:
	powershell -NoProfile -Command "Remove-Item -Recurse -Force $(BUILD_DIR) -ErrorAction SilentlyContinue"

.PHONY: all scalar simd deploy clean
