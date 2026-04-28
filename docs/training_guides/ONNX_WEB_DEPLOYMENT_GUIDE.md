# SegFormer ONNX Web Deployment Guide 🚀

Complete guide to export your trained SegFormer model to ONNX and deploy it on the web.

---

## 📋 Prerequisites

```bash
# Install required packages
pip install torch transformers onnx onnxruntime safetensors
```

---

## 🔄 Step 1: Export Model to ONNX

### Option A: Using the Export Script (Recommended)

```bash
# Run the export script
python export_segformer_to_onnx.py
```

This will:
- ✅ Load your trained model from `checkpoints/segformer/best_model/`
- ✅ Export to ONNX format
- ✅ Optimize the model
- ✅ Validate outputs (PyTorch vs ONNX)
- ✅ Save to `onnx_models/segformer_fence_detector.onnx`

**Expected Output:**
```
======================================================================
SEGFORMER MODEL EXPORT TO ONNX
======================================================================
Loading model from checkpoints\segformer\best_model...
✓ Model loaded successfully
  - Parameters: 3,716,226
  - Classes: 2

Exporting to ONNX...
  - Input size: 512x512
  - Batch size: 1
  - Opset version: 14
✓ ONNX export complete: onnx_models\segformer_fence_detector.onnx
  - File size: 14.15 MB

Optimizing ONNX model...
✓ Optimization complete: onnx_models\segformer_fence_detector_optimized.onnx
  - Original: 14.15 MB
  - Optimized: 14.15 MB

Validating ONNX export...
✓ Validation complete
  - Max difference: 0.000012
  - Mean difference: 0.000003
  ✓ PASSED (max_diff < 0.001)

✓ Metadata saved: onnx_models\model_metadata.json
======================================================================
EXPORT COMPLETE!
======================================================================
```

### Option B: Manual Export (Advanced)

```python
from transformers import SegformerForSemanticSegmentation
import torch

# Load model
model = SegformerForSemanticSegmentation.from_pretrained(
    './checkpoints/segformer/best_model'
)
model.eval()

# Export
dummy_input = torch.randn(1, 3, 512, 512)
torch.onnx.export(
    model,
    dummy_input,
    'segformer_fence_detector.onnx',
    input_names=['pixel_values'],
    output_names=['logits'],
    dynamic_axes={'pixel_values': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=14
)
```

---

## 🌐 Step 2: Test Locally with Web Server

### Start the Server

```bash
# Start simple HTTP server
python start_web_server.py
```

**Output:**
```
======================================================================
SEGFORMER WEB SERVER
======================================================================
Server starting on port 8000...
Serving from: D:\Ubuntu\TECHNOTAU (2)\...

📂 Make sure you have:
   ✓ index_segformer_web.html
   ✓ onnx_models/segformer_fence_detector.onnx

🌐 Open in browser: http://localhost:8000/index_segformer_web.html

Press Ctrl+C to stop server
======================================================================
```

### Open in Browser

1. Open: **http://localhost:8000/index_segformer_web.html**
2. Upload a fence image
3. Click "Detect Fence"
4. Choose color and click "Recolor Fence"
5. Download result

---

## 📁 File Structure

After export, your directory should look like:

```
training/
├── checkpoints/
│   └── segformer/
│       └── best_model/
│           ├── config.json
│           ├── model.safetensors (or pytorch_model.bin)
│           └── preprocessor_config.json
├── onnx_models/                    # ← Created by export script
│   ├── segformer_fence_detector.onnx
│   ├── segformer_fence_detector_optimized.onnx  (optional)
│   └── model_metadata.json
├── export_segformer_to_onnx.py     # ← Export script
├── index_segformer_web.html        # ← Web interface
├── start_web_server.py             # ← HTTP server
└── ONNX_WEB_DEPLOYMENT_GUIDE.md    # ← This file
```

---

## 🔧 Configuration

### Update Model Path in HTML

If your ONNX model is in a different location, edit `index_segformer_web.html`:

```javascript
// Line ~471
const CONFIG = {
    MODEL_PATH: './onnx_models/segformer_fence_detector.onnx',  // ← Update this
    INPUT_SIZE: 512,  // Match your training size
    MEAN: [0.485, 0.456, 0.406],
    STD: [0.229, 0.224, 0.225],
    NUM_CLASSES: 2
};
```

### Adjust Input Size

If you trained with different input size (e.g., 384):

1. **In export script** (`export_segformer_to_onnx.py`):
   ```python
   INPUT_SIZE = 384  # Line 27
   ```

2. **In HTML** (`index_segformer_web.html`):
   ```javascript
   INPUT_SIZE: 384,  // Line 473
   ```

---

## 🚀 Deployment Options

### Option 1: Local File (Simplest)

1. Copy files to a folder:
   ```
   web_app/
   ├── index.html (rename from index_segformer_web.html)
   ├── onnx_models/
   │   └── segformer_fence_detector.onnx
   ```

2. Open `index.html` directly in browser (may have CORS issues)

### Option 2: Static Web Hosting

Deploy to any static hosting service:

**GitHub Pages:**
```bash
# Create gh-pages branch
git checkout -b gh-pages
git add index.html onnx_models/
git commit -m "Deploy web app"
git push origin gh-pages

# Access at: https://yourusername.github.io/repo-name/
```

**Netlify:**
```bash
# Drag and drop folder to: https://app.netlify.com/drop
```

**Vercel:**
```bash
vercel --prod
```

### Option 3: Custom Server

**Node.js (Express):**
```javascript
const express = require('express');
const app = express();
app.use(express.static('.'));
app.listen(8000, () => console.log('Server on http://localhost:8000'));
```

**Python (Flask):**
```python
from flask import Flask, send_from_directory
app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index_segformer_web.html')

app.run(port=8000)
```

---

## 🐛 Troubleshooting

### Issue 1: Model Not Loading

**Error:** `Failed to load model: 404 Not Found`

**Solution:**
- Check file path in `CONFIG.MODEL_PATH`
- Ensure ONNX file exists in `onnx_models/` folder
- Use relative path: `./onnx_models/model.onnx`

### Issue 2: CORS Error

**Error:** `CORS policy: No 'Access-Control-Allow-Origin' header`

**Solution:**
- Use `start_web_server.py` (has CORS enabled)
- Or use browser extension: "Allow CORS"
- Or deploy to real web server

### Issue 3: Memory Error

**Error:** `RangeError: Array buffer allocation failed`

**Solution:**
- Model too large for browser
- Enable quantization in export script:
  ```python
  QUANTIZE = True  # Line 34
  ```
- Reduce input size (512 → 384)

### Issue 4: Slow Inference

**Problem:** Detection takes >5 seconds

**Solution:**
- Use optimized model (created during export)
- Update HTML to use WebGL backend:
  ```javascript
  executionProviders: ['webgl', 'wasm']  // Line 484
  ```
- Consider quantization (smaller, faster)

### Issue 5: Wrong Predictions

**Problem:** Mask doesn't match training results

**Solution:**
- Verify input size matches training (512)
- Check preprocessing (mean/std values)
- Adjust detection threshold slider
- Validate export outputs match PyTorch

---

## 📊 Performance Benchmarks

| Device | Backend | Model Size | Inference Time |
|--------|---------|------------|----------------|
| Desktop (Chrome) | WebGL | 14 MB | ~200ms |
| Desktop (Chrome) | WASM | 14 MB | ~800ms |
| Desktop (Chrome) | WASM (Quantized) | 4 MB | ~600ms |
| Mobile (Safari) | WASM | 14 MB | ~2000ms |

---

## 🎨 Customization

### Change UI Colors

Edit CSS in `index_segformer_web.html`:
```css
/* Line 16 - Main gradient */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Line 46 - Badge color */
background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
```

### Add More Blend Modes

In HTML JavaScript section:
```javascript
// Line 489 - Add to select options
<option value="colordodge">Color Dodge</option>

// Line 690 - Add blend mode function
case 'colordodge':
    return [
        Math.min(255, r * 255 / (255 - cr)),
        Math.min(255, g * 255 / (255 - cg)),
        Math.min(255, b * 255 / (255 - cb))
    ];
```

### Add Metrics Display

Show IoU/Dice after detection:
```javascript
// After detection completes
const iou = calculateIoU(maskData, groundTruth);
updateStatus(`✓ Detected! IoU: ${iou.toFixed(2)}`, 'success');
```

---

## 📚 Additional Resources

- **ONNX Runtime Web:** https://onnxruntime.ai/docs/tutorials/web/
- **Transformers ONNX Export:** https://huggingface.co/docs/transformers/serialization
- **SegFormer Paper:** https://arxiv.org/abs/2105.15203

---

## ✅ Checklist

Before deploying to production:

- [ ] Model exported successfully
- [ ] ONNX validation passed (max_diff < 0.001)
- [ ] Tested locally with sample images
- [ ] Performance acceptable (<1s inference)
- [ ] UI responsive on mobile
- [ ] Error handling works
- [ ] Download function works
- [ ] CORS configured correctly
- [ ] Files organized properly
- [ ] Documentation updated

---

## 🎯 Quick Start Commands

```bash
# 1. Export model
python export_segformer_to_onnx.py

# 2. Start server
python start_web_server.py

# 3. Open browser
http://localhost:8000/index_segformer_web.html

# 4. Test with image
# Upload → Detect → Recolor → Download
```

---

## 💡 Tips

1. **Optimize for Mobile:** Reduce input size to 384 or 256
2. **Improve Speed:** Enable WebGL backend
3. **Reduce Size:** Use quantization (INT8)
4. **Better UX:** Add progress bar during inference
5. **Debugging:** Open browser console (F12) for errors

---

## 📝 License & Credits

- **Model:** SegFormer (NVIDIA)
- **Framework:** Transformers (Hugging Face)
- **Runtime:** ONNX Runtime Web
- **UI:** Custom HTML/CSS/JS

---

**Need Help?** Check browser console (F12) for detailed error messages.

**Good luck with your deployment! 🚀**
