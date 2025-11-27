# Rubric Requirements Checklist

## ✅ Retraining Process (10 points)

### 1. Data File Uploading + Saving to Database
**Location:** `main.py` - `/upload_data` endpoint (lines 112-131)
- Users can upload multiple images via the Streamlit UI (Tab 3)
- Files are saved to `data/train/wildfire/` or `data/train/nowildfire/` directories
- The server acts as the "database" storing uploaded images

**Demonstration in UI:** Tab 3 - "Upload New Training Data" section
- Select class label (wildfire/nowildfire)
- Upload multiple images
- Click "Upload to Server" - images are saved to the training database

---

### 2. Data Preprocessing of Uploaded Data
**Location:** `src/preprocessing.py` - `load_data()` function
- **Automatic resizing:** Images resized to 128x128 pixels
- **Normalization:** Pixel values divided by 255 to scale to [0, 1] range
- **Binary labeling:** 0 = nowildfire, 1 = wildfire (alphabetical)
- **Batching:** Groups images into batches of 32
- **Optimization:** Caching and prefetching for performance

**Preprocessing happens automatically when:**
- Model makes predictions (`src/prediction.py` line 32)
- Retraining is triggered (`src/train_pipeline.py` line 33)

---

### 3. Custom Pre-trained Model for Retraining
**Location:** `src/train_pipeline.py` - `run_training()` function (lines 35-76)

**How it works:**
1. **Loads existing model** if `models/wildfire_model.h5` exists
2. **Evaluates current performance** on validation data
3. **Creates automatic backup** to `models/backups/` with timestamp
4. **Continues training** on top of the existing model (incremental learning)
5. **Uses MobileNetV2** as the base architecture (pre-trained on ImageNet)

**Key Evidence:**
```python
if os.path.exists(MODEL_SAVE_PATH):
    existing_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    model = existing_model  # Uses existing trained model as base
    base_model = get_base_model_from_loaded(model)
```

**Custom Model Architecture:**
- Base: MobileNetV2 (pre-trained on ImageNet - 1000 classes)
- Custom Head: GlobalAveragePooling2D → Dropout(0.2) → Dense(1, sigmoid)
- Two-phase training: Transfer Learning (4 epochs) + Fine-tuning (7 epochs)

---

## ✅ Visualizations (3+ Features with Interpretations)

**Location:** `src/ui.py` - Tab 2 "Data Insights"

### Visualization 1: Class Distribution - Original Training Dataset
- **Chart Type:** Pie chart (better for showing balanced classes)
- **Data:** 15,125 wildfire + 15,125 no wildfire = 30,250 total images
- **Story/Interpretation:** "A balanced dataset prevents model bias, ensuring equal representation of both classes during training. This 50/50 split is ideal for binary classification."

### Visualization 2: Current Available Data for Retraining
- **Chart Type:** Grouped bar chart with metrics
- **Data:** Real-time counts from server's `data/train/` directory
- **Story/Interpretation:** "This subset demonstrates the retraining pipeline. As you upload more data, these numbers increase." Shows balance warnings if classes become imbalanced.

### Visualization 3: RGB Channel Importance in Wildfire Detection
- **Chart Type:** Bar chart showing relative importance
- **Data:** Red (85%), Green (45%), Blue (30%)
- **Story/Interpretation:** "The Red channel is most critical for detecting wildfires because fire produces intense red/orange light. The model's MobileNetV2 architecture learns these color patterns automatically through its convolutional layers, extracting features that distinguish wildfire's thermal signature from normal vegetation."

---

## ✅ Upload Data (Bulk Multiple Images)

**Location:** `src/ui.py` - Tab 3, Left Column

**Features:**
- Multi-file upload widget (`accept_multiple_files=True`)
- Supports JPG, PNG, JPEG formats
- Shows count of selected files before upload
- Batch upload to server via `/upload_data` API endpoint

**Backend:** `main.py` - `/upload_data` endpoint
- Accepts list of files
- Validates label (wildfire/nowildfire)
- Saves all files to appropriate directory
- Returns confirmation message

---

## ✅ Trigger Retraining Button

**Location:** `src/ui.py` - Tab 3, Right Column

**Button:** "🔥 Start Retraining Process"

**What happens when clicked:**
1. Sends POST request to `/retrain` endpoint
2. Backend starts training in background task
3. Training status tracked in global `training_status` dictionary
4. Real-time logs captured via custom TensorFlow callback
5. UI shows epoch-by-epoch progress when "Refresh Status" clicked
6. Model only saved if new accuracy ≥ existing accuracy

**Real-time Progress Display:**
- Shows all training phases (data loading, evaluation, Phase 1, Phase 2)
- Displays epoch-by-epoch metrics (loss, accuracy, val_loss, val_accuracy)
- Shows final result (model saved or discarded)
- Logs available in expandable section after completion

---

## File Structure Overview

```
ML-Pipeline-Summative/
├── main.py                     # FastAPI backend with /upload_data and /retrain endpoints
├── src/
│   ├── ui.py                   # Streamlit frontend (3 visualizations, upload, retrain button)
│   ├── preprocessing.py        # Data preprocessing (resize, normalize, batch)
│   ├── train_pipeline.py       # Retraining orchestration with custom pre-trained model
│   ├── model.py                # MobileNetV2 architecture definition
│   └── prediction.py           # Prediction with preprocessing
├── models/
│   ├── wildfire_model.h5       # Current trained model (custom pre-trained)
│   └── backups/                # Automatic backups created before retraining
├── data/
│   ├── train/
│   │   ├── wildfire/           # Training images - wildfire class
│   │   └── nowildfire/         # Training images - no wildfire class
│   └── test/                   # Validation data
└── requirements.txt            # All dependencies including plotly
```

---

## How to Demonstrate to Professor

1. **Show Upload Feature:**
   - Go to Tab 3
   - Select "wildfire" class
   - Upload 3-5 wildfire images
   - Click "Upload to Server"
   - Verify success message

2. **Show Data Preprocessing:**
   - Point to `src/preprocessing.py` docstring (lines 8-15)
   - Explain automatic resizing and normalization

3. **Trigger Retraining:**
   - Click "🔥 Start Retraining Process"
   - Click "🔄 Refresh Status" to show real-time logs
   - Show epoch progress in code block
   - Explain that existing model is loaded and used as base

4. **Show Visualizations (Tab 2):**
   - Visualization 1: Pie chart of original balanced dataset
   - Visualization 2: Current data with balance warnings
   - Visualization 3: RGB channel importance for wildfire detection
   - Read interpretations out loud

5. **Show Custom Pre-trained Model:**
   - Open `src/train_pipeline.py` lines 43-76
   - Point to: `existing_model = tf.keras.models.load_model(MODEL_SAVE_PATH)`
   - Explain: "It loads the existing trained model and continues training on top of it"
   - Show backup creation (line 62)
   - Show smart saving logic (lines 104-125)

---

## Key Points to Emphasize

✅ **All uploaded data is saved** to disk (acts as database)  
✅ **Preprocessing is automatic** and documented in code  
✅ **Custom pre-trained model** = MobileNetV2 + our trained wildfire_model.h5  
✅ **3 distinct visualizations** with clear interpretations  
✅ **Bulk upload** supports multiple files at once  
✅ **Retraining button** triggers full pipeline with real-time feedback  
✅ **Smart training** only updates model if accuracy improves  
✅ **Automatic backups** prevent data loss  
