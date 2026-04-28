# Future Data Enhancement Plan - Complex Scene Handling
**Date:** November 14, 2025  
**Current Model:** Mask2Former + SegFormer-B5  
**Training Data:** 804 images, 27 scene types (clean fences)

---

## Current Model Limitations

### **What It Does Well (92%+ IoU):**
✅ Clean fences with lawn/grass backgrounds  
✅ Isolated fence structures  
✅ Standard fence types (picket, privacy, chain-link, etc.)  
✅ Various lighting conditions (trained augmentations)  
✅ Different fence materials (wood, vinyl, metal)  

### **What It Will Struggle With (40-75% IoU):**
❌ Trees/large plants behind or near fences  
❌ Wooden decks, sheds, furniture near fences  
❌ Pets/humans touching or standing near fences  
❌ Firewood/logs stacked against fences  
❌ Dense vegetation (vines, ivy) growing on fences  
❌ Multiple wooden structures in same scene  

**Root Cause:** Training data contains only clean, isolated fence scenes without distractors or occlusions.

---

## Phase 2: Data Collection Requirements

### **Target: +200-300 Images with Complex Scenarios**

#### **Category 1: Occlusions (60-80 images)**
- Fences with trees partially blocking view
- Bushes/shrubs growing in front of fence
- Parked vehicles near fence line
- Garden furniture obscuring parts of fence
- Multiple fence segments with gaps

**Annotation Rule:** Only mark the FENCE pixels, not the occluding objects.

#### **Category 2: Wooden Distractors (40-60 images)**
- Fence + wooden deck in same frame
- Fence + garden shed (wooden walls)
- Fence + wooden benches/tables nearby
- Fence + playground equipment (wooden)
- Fence + firewood piles/logs against fence
- Fence + wooden pergola/trellis

**Annotation Rule:** Mark ONLY fence structure, not other wooden objects.

#### **Category 3: Living Subjects (30-40 images)**
- People standing/leaning on fence
- People walking near fence
- Dogs/cats near fence
- Birds sitting on fence rails
- People working on fence (painting, repairs)

**Annotation Rule:** Exclude humans/animals from fence mask, even if touching.

#### **Category 4: Vegetation Interference (50-70 images)**
- Vines/ivy growing on fence
- Dense grass touching fence base
- Flower beds at fence line
- Overhanging tree branches on fence
- Climbing plants on fence posts
- Moss/lichen on wooden fences

**Annotation Rule:** Mark only fence structure, not vegetation (even if attached).

#### **Category 5: Complex Backgrounds (30-50 images)**
- Fences with dense tree backgrounds
- Fences with buildings in background
- Multiple fence types in one scene
- Fences with open/closed gates
- Partial fences (under construction)
- Damaged/broken fence sections

**Annotation Rule:** Focus on precise boundary detection despite background clutter.

---

## Phase 2: Fine-Tuning Strategy

### **Once Enhanced Dataset Ready:**

**1. Create New Dataset Directory:**
```
data_v2/
  images/          # 804 original + 200-300 new complex scenes
  masks/           # Updated masks
  annotations.json # COCO format (optional)
```

**2. Fine-Tuning Configuration:**
```python
# In train_Mask2Former.py Config class:
IMAGES_DIR = PROJECT_ROOT / "data_v2" / "images"
MASKS_DIR = PROJECT_ROOT / "data_v2" / "masks"
EPOCHS = 100  # Fewer epochs for fine-tuning
LEARNING_RATE = 1e-5  # Lower LR (10x smaller)
FREEZE_BACKBONE_EPOCHS = 0  # Keep all layers trainable
```

**3. Load Phase 1 Checkpoint:**
```python
# Load best model from current training
checkpoint = torch.load('checkpoints/mask2former/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

**4. Fine-Tune on Enhanced Dataset:**
- Train for 100 epochs (not 200)
- Monitor IoU on complex scenes specifically
- Target: 80%+ IoU on complex scenes

---

## Expected Performance Improvements

### **Before Phase 2 (Current Model):**
| Scenario | IoU |
|----------|-----|
| Clean fence + lawn | 92% |
| Fence + tree background | 60-75% |
| Fence + wooden deck | 40-60% |
| Fence + vegetation | 65-80% |

### **After Phase 2 (Fine-Tuned Model):**
| Scenario | IoU |
|----------|-----|
| Clean fence + lawn | 92% (maintained) |
| Fence + tree background | 85-90% |
| Fence + wooden deck | 75-85% |
| Fence + vegetation | 80-90% |

---

## Data Collection Tips

### **DIY Data Collection:**
1. **Take photos in your neighborhood:**
   - Walk around with smartphone
   - Capture 300-500 raw images of complex fence scenes
   - Focus on scenarios listed above

2. **Use Google Street View:**
   - Screenshot fence images with trees, buildings, etc.
   - Ensure variety in backgrounds and distractors

3. **Synthetic Data Generation:**
   - Use DALL-E/MidJourney/Stable Diffusion
   - Prompts: "wooden fence with dog nearby", "fence with tree behind it", etc.
   - Generates diverse scenarios quickly

### **Annotation Tools:**
- **LabelMe** (current tool) - Good for polygons
- **CVAT** - Better for complex scenes, faster annotation
- **Label Studio** - Best for large-scale annotation projects

### **Time Estimate:**
- Data collection: 2-4 hours (300 images)
- Annotation: 10-15 hours (300 images × 2-3 min each)
- Training/fine-tuning: 3-4 hours (100 epochs)
- **Total:** ~20-25 hours for Phase 2

---

## Validation Strategy for Phase 2

### **Create Test Sets:**

**Test Set 1: Clean Fences (current scenarios)**
- 50 images similar to current training data
- Ensures Phase 1 performance is maintained

**Test Set 2: Complex Scenes (new scenarios)**
- 50 images with trees, distractors, vegetation
- Measures improvement on target scenarios

**Test Set 3: Edge Cases**
- 20 images with extreme challenges
- Multiple distractors, heavy occlusion, etc.

### **Success Criteria:**
- Test Set 1: IoU ≥ 90% (maintain current performance)
- Test Set 2: IoU ≥ 80% (significant improvement)
- Test Set 3: IoU ≥ 70% (acceptable on hard cases)

---

## Alternative: Active Learning Approach

**Instead of collecting 300 images upfront:**

1. **Deploy Current Model** (after Phase 1 training completes)
2. **Collect 50-100 images** where model fails
3. **Annotate failure cases only**
4. **Fine-tune on failure cases**
5. **Repeat until performance acceptable**

**Benefits:**
- More efficient (focus on actual failures)
- Faster iteration cycles
- Less annotation effort

**Trade-offs:**
- Requires deployment infrastructure
- Longer overall timeline
- May miss edge cases not encountered in deployment

---

## Recommended Next Steps

### **After Current Training Completes (6.7 hours):**

**Week 1:**
1. ✅ Evaluate Phase 1 model on validation set
2. ✅ Test on 10-20 real-world images (with distractors)
3. ✅ Identify specific failure modes
4. ✅ Document which scenarios need most improvement

**Week 2-3:**
5. 📸 Collect 200-300 images focusing on failure modes
6. 🏷️ Annotate new images (allocate 10-15 hours)
7. 🔀 Merge with original dataset (1,000+ total images)

**Week 4:**
8. 🔧 Configure fine-tuning (lower LR, fewer epochs)
9. 🚀 Fine-tune for 100 epochs (~3-4 hours)
10. ✅ Validate on all 3 test sets

**Production Deployment:**
11. 🎯 A/B test Phase 1 vs Phase 2 models
12. 📊 Monitor real-world performance
13. 🔄 Continue iterative improvements

---

## Cost-Benefit Analysis

### **Phase 1 Only (Current Plan):**
**Cost:** ~7 hours training  
**Benefit:** 92% IoU on clean fences  
**Limitation:** 40-75% IoU on complex scenes  
**Use Case:** Prototype, proof-of-concept, controlled environments  

### **Phase 1 + Phase 2 (Recommended for Production):**
**Cost:** ~7 hours (Phase 1) + ~25 hours (Phase 2) = 32 hours total  
**Benefit:** 80-90% IoU on ALL scenarios  
**Use Case:** Production deployment, real-world fencing applications  

### **ROI Consideration:**
- If staining app used on **clean lawns only** → Phase 1 sufficient
- If staining app used on **typical backyards** → Phase 2 required
- If staining app needs **commercial deployment** → Phase 2 + ongoing monitoring

---

## Final Recommendation

**For MVP/Prototype:** ✅ **Current Phase 1 is EXCELLENT**  
**For Production:** ⚠️ **Phase 2 required within 1-2 months**  

**Hybrid Approach (Best):**
1. Deploy Phase 1 model to beta users (clean environments)
2. Collect real failure cases from beta deployment
3. Use failure cases to guide Phase 2 data collection
4. Fine-tune and upgrade to production model

---

**STATUS:** Phase 1 training ready to start (EARLY_STOPPING=False verified)  
**NEXT:** Run `python train_Mask2Former.py` for 200 epochs (~6.7 hours)  
**THEN:** Evaluate and plan Phase 2 based on actual results
