/**
 * fsv-loader.js — defines the <fence-simulator> custom element.
 *
 * Renders the entire simulator inside a Shadow DOM root so WordPress theme
 * CSS cannot reach in. Reads asset URLs from window.FSV_ASSETS (injected by
 * PHP via wp_localize_script).
 *
 * Dependencies (must load before this script):
 *   - app.js          (defines window.FSV_initFenceSimulator)
 *
 * About fonts/icons inside Shadow DOM:
 *   - @font-face declared at document level DOES propagate into shadow trees
 *     in modern browsers. That's enough for plain text fonts (Inter, Plus
 *     Jakarta Sans) — their CSS contains only @font-face.
 *   - Bootstrap Icons is different: its CSS contains @font-face AND class
 *     rules like `.bi-cloud-arrow-up::before { content: "\f1d8"; }`. Class
 *     rules at document level CANNOT reach elements inside a shadow root —
 *     that's exactly the isolation we want. So we MUST inject the icons CSS
 *     <link> INSIDE the shadow root, otherwise icon glyphs render as empty.
 *   - We also inject the Google Fonts <link> inside the shadow root for
 *     redundancy (browser caches the URL, so no extra HTTP cost).
 */
(function () {
  "use strict";

  if (window.customElements && window.customElements.get("fence-simulator")) {
    return;
  }

  let _cachedAssets = null;

  async function loadAssets() {
    if (_cachedAssets) return _cachedAssets;
    const A = window.FSV_ASSETS || {};
    if (!A.css || !A.html) {
      throw new Error(
        "[fence-simulator] window.FSV_ASSETS missing css/html URLs",
      );
    }
    const [cssText, htmlText] = await Promise.all([
      fetch(A.css, { cache: "default" }).then((r) => {
        if (!r.ok) throw new Error("CSS fetch failed: " + r.status);
        return r.text();
      }),
      fetch(A.html, { cache: "default" }).then((r) => {
        if (!r.ok) throw new Error("HTML fetch failed: " + r.status);
        return r.text();
      }),
    ]);
    _cachedAssets = { css: cssText, html: htmlText };
    return _cachedAssets;
  }

  class FenceSimulator extends HTMLElement {
    constructor() {
      super();
      this._initialized = false;
    }

    async connectedCallback() {
      if (this._initialized) return;
      this._initialized = true;

      const shadow = this.attachShadow({ mode: "open" });

      let assets;
      try {
        assets = await loadAssets();
      } catch (err) {
        console.error("[fence-simulator] asset load failed:", err);
        shadow.innerHTML =
          '<div style="padding:16px;font-family:sans-serif;color:#dc2626;">' +
          "Fence simulator failed to load. Please refresh." +
          "</div>";
        return;
      }

      // Inject font + icon <link> tags + the stylesheet + body markup into
      // the shadow root. <link> and <style> both work inside Shadow DOM, and
      // theme CSS cannot reach in either.
      const A = window.FSV_ASSETS || {};
      const fontsLink = A.fonts
        ? '<link rel="stylesheet" href="' + A.fonts + '">'
        : "";
      const iconsLink = A.icons
        ? '<link rel="stylesheet" href="' + A.icons + '">'
        : "";
      shadow.innerHTML =
        fontsLink + iconsLink + "<style>" + assets.css + "</style>" + assets.html;

      // Hand control to app.js. Wait one frame so the shadow tree is laid out
      // before sizing/equalize logic queries getBoundingClientRect.
      await new Promise((r) => requestAnimationFrame(r));

      if (typeof window.FSV_initFenceSimulator !== "function") {
        console.error(
          "[fence-simulator] window.FSV_initFenceSimulator missing — " +
            "did app.js load before fsv-loader.js?",
        );
        return;
      }
      try {
        window.FSV_initFenceSimulator(shadow);
      } catch (err) {
        console.error("[fence-simulator] init failed:", err);
      }
    }
  }

  customElements.define("fence-simulator", FenceSimulator);
})();
