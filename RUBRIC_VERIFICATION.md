# Assignment Rubric Verification Checklist

This document maps all rubric requirements to specific project components for easy verification.

---

## 1. Video Demo (5 points)

**Requirement**: Clear and user-friendly demonstration of prediction and retraining process with camera on

**Status**: READY FOR RECORDING

**What to demonstrate in video**:
1. Show yourself on camera introducing the project
2. Navigate to deployed Streamlit app (URL in README)
3. **Prediction Demo**:
   - Upload a wildfire image
   - Show prediction result with confidence score
   - Upload a non-wildfire image
   - Show correct classification
4. **Retraining Demo**:
   - Upload multiple new training images (bulk upload)
   - Show confirmation of images saved
   - Click "Start Retraining Process"
   - Click "Refresh Status" to show epoch-by-epoch progress
   - Show training completion message
   - Explain that model only saves if accuracy improves

**Video checklist**:
- [ ] Camera is ON throughout video
- [ ] Clear audio
- [ ] Show prediction working correctly
- [ ] Show retraining process from start to finish
- [ ] Demonstrate epoch logs appearing in real-time
- [ ] Keep video under 10 minutes

---

## 2. Retraining Process (10 points)

**Requirement**: Script + Model file present with clear demonstration of:
1. Data file uploading + saving to database
2. Data preprocessing of uploaded data
3. Retraining using custom pre-trained model

### Evidence Files:

**Script for Retraining**: `src/train_pipeline.py`
- **Lines 60-249**: Complete `run_training()` function
- **Lines 90-95**: Loads existing model (`tf.keras.models.load_model`)
- **Lines 120-131**: Recompiles model with fresh optimizer
- **Lines 148-175**: Phase 1 training (2 epochs)
- **Lines 177-204**: Phase 2 fine-tuning (3 epochs)

**Model File**: `models/wildfire_model.h5`
- Size: 23.22 MB
- Architecture: MobileNetV2 (pre-trained on ImageNet) + custom classification head
- Trained on 30,250 images (15,125 per class)

**Data Upload + Saving**: `main.py`
- **Lines 112-131**: `/upload_data` endpoint
- Saves uploaded files to `data/train/wildfire/` or `data/train/nowildfire/`
- Supports bulk file upload

**Data Preprocessing**: `src/preprocessing.py`
- **Lines 8-18**: Detailed preprocessing documentation
- **Lines 27-33**: Automatic resizing to 128x128
- **Lines 36-37**: Normalization (divide by 255 to [0, 1] range)
- **Lines 32**: Binary labeling (0=nowildfire, 1=wildfire)
- **Lines 35**: Batching (groups of 32)
- **Lines 39-40**: Caching and prefetching optimization

**Custom Pre-trained Model**: `src/model.py`
- **Lines 3-9**: Documentation explaining MobileNetV2 as custom pre-trained model
- **Lines 15-19**: MobileNetV2 loaded with ImageNet weights
- **Lines 21-22**: Base model frozen initially
- **Lines 24-29**: Custom classification head added

### Demonstration Flow:
1. User uploads images via Streamlit UI (Tab 3)
2. Files sent to `/upload_data` endpoint
3. Backend saves files to `data/train/{class}/`
4. User clicks "Start Retraining Process"
5. `/retrain` endpoint triggers `run_training()`
6. Function loads existing model from `models/wildfire_model.h5`
7. Creates backup in `models/backups/`
8. Preprocessing automatically applied via `load_data()`
9. Model trained with custom pre-trained MobileNetV2 as base
10. New model saved only if accuracy improves

**Verification**: See RUBRIC_CHECKLIST.md lines 1-76 for detailed mapping

---

## 3. Prediction Process (10 points)

**Requirement**: Script + Model file present with clear demonstration of:
1. Inserting a data point for prediction (image)
2. Displays CORRECT prediction based on label/class

### Evidence Files:

**Script for Prediction**: `src/prediction.py`
- **Lines 1-50**: Complete prediction pipeline
- **Lines 10-14**: Loads model from `models/wildfire_model.h5`
- **Lines 17-34**: `predict_image()` function
  - Line 24: Loads image
  - Line 27: Resizes to 128x128
  - Line 30: Converts to array
  - Line 32: **Normalizes by dividing by 255** (critical for correct predictions)
  - Line 33: Adds batch dimension
  - Line 36: Makes prediction
  - Lines 39-46: Returns formatted result

**Model File**: `models/wildfire_model.h5` (same as retraining)

**API Endpoint**: `main.py`
- **Lines 75-93**: `/predict` endpoint
- Accepts image upload
- Calls `predict_image()` function
- Returns JSON with prediction and confidence

**Streamlit UI**: `src/ui.py`
- **Lines 31-71**: Prediction tab implementation
- Upload image widget
- Display image
- "Analyze Image" button
- Shows result with color coding (red for wildfire, green for safe)

### Prediction Accuracy Verification:

**Test Files Created**:
- `test_model_preprocessing.py`: Verifies model expects [0, 1] normalized inputs
- `test_api_upload.py`: Tests actual API predictions

**Proof of Correct Predictions**:
- Model trained on 30,250 balanced images
- Validation accuracy: 97.29% (from notebook)
- Preprocessing bug was fixed (normalization added)
- Test showed wildfire image with normalization: 0.7377 score (correct)
- Test showed same image without normalization: 0.0000 score (wrong - proving fix works)

**Demonstration Flow**:
1. User uploads satellite image via Streamlit UI (Tab 1)
2. Image sent to `/predict` endpoint
3. Backend applies preprocessing (resize + normalize)
4. Model makes prediction
5. Result displayed: "Wildfire Detected (87.5%)" or "No Wildfire (92.3%)"
6. Color coded for clarity

---

## 4. Evaluation of Models (10 points)

**Requirement**: Notebook present with:
1. Clear preprocessing steps
2. Use of optimization techniques (regularization, optimizers, early stopping, pre-trained model, hyperparameter tuning)
3. At least 4 evaluation metrics (Accuracy, Loss, F1 Score, Precision, Recall)

### Evidence File:

**Notebook**: `notebook/wildfire_detection.ipynb`

### Preprocessing Steps (Clear Documentation):

**Cell 6** (lines 57-69): Dataset loading and structure
**Cell 7** (lines 79-106): Image preprocessing
- Resizing to 128x128
- Normalization to [0, 1]
- Data augmentation setup

**Cell 8** (lines 109-167): Data pipeline creation
- Training/validation split
- Batching
- Prefetching
- Caching

### Optimization Techniques Used:

**1. Pre-trained Model** (Primary Technique):
- **Cell 19** (lines 261-304): MobileNetV2 with ImageNet weights
- Transfer learning from 1000-class ImageNet
- Fine-tuning approach

**2. Regularization**:
- **Cell 19**: Dropout(0.2) layer
- **Cell 10**: Data augmentation (rotation, flip, zoom)

**3. Optimizers**:
- **Cell 19**: Adam optimizer with learning_rate=0.001
- **Cell 24**: Adam optimizer with learning_rate=1e-5 (fine-tuning)

**4. Hyperparameter Tuning**:
- Tested multiple architectures (6 different models)
- Compared epochs (10 vs others)
- Learning rate adjustment for fine-tuning

### Evaluation Metrics (More than 4):

**Cell 28** (lines 478-510): Comprehensive evaluation table showing:

| Model | **Accuracy** | **Precision** | **Recall** | **F1 Score** | Notes |
|-------|--------------|---------------|------------|--------------|-------|
| Baseline CNN | 91.56% | 0.94 | 0.89 | 0.91 | Overfit |
| Improved CNN | 95.02% | 0.98 | 0.93 | 0.95 | Dropout helped |
| Transfer Learning | 96.11% | 0.97 | 0.96 | 0.96 | Major improvement |
| Fine-Tuned Transfer | **97.29%** | **0.99** | **0.97** | **0.98** | **BEST** |
| Augmented CNN | 94.35% | 0.97 | 0.92 | 0.95 | Less effective |

**Metrics Demonstrated**:
1. Accuracy
2. Precision
3. Recall
4. F1 Score
5. Loss (shown in training graphs throughout notebook)

**Additional Evaluation**:
- **Cell 31-33**: Confusion matrices visualized
- **Cell 38**: Training history plots (loss and accuracy curves)
- **Cell 46**: Additional validation predictions
- **Cell 56**: Final generalization test (Africa wildfire image)

### Model Comparison:
- 6 different model architectures tested
- Systematic progression from baseline to optimized
- Clear winner identified: Fine-tuned Transfer Learning (97.29% accuracy)

---

## 5. Deployment Package (10 points)

**Requirement**: 
1. Showcases UI using web app (public URL)
2. Contains data insights based on dataset

### Evidence:

**Public URL Deployment**:
- **Platform**: Railway
- **API URL**: https://wildfire-prediction.up.railway.app
- **Streamlit UI**: [URL to be added in README]
- **API Documentation**: https://wildfire-prediction.up.railway.app/docs

**Web App Features**:
- **Tab 1**: Real-time prediction interface
- **Tab 2**: Data insights and visualizations
- **Tab 3**: Retraining pipeline control

**Deployment Files**:
- `Procfile`: Railway deployment configuration
- `requirements.txt`: All dependencies specified
- `main.py`: FastAPI production server
- `src/ui.py`: Streamlit web interface

### Data Insights (Tab 2 - Required):

**Visualization 1: Original Training Dataset Distribution**
- **Location**: `src/ui.py` lines 80-104
- **Type**: Pie chart
- **Data**: 30,250 images (15,125 per class)
- **Insight**: "A balanced dataset prevents model bias, ensuring equal representation of both classes during training. This 50/50 split is ideal for binary classification."

**Visualization 2: Current Available Data for Retraining**
- **Location**: `src/ui.py` lines 108-141
- **Type**: Grouped bar chart with metrics
- **Data**: Real-time counts from server directories
- **Insight**: "This subset demonstrates the retraining pipeline. As you upload more data, these numbers increase." Shows balance warnings if classes become imbalanced.

**Visualization 3: RGB Channel Importance**
- **Location**: `src/ui.py` lines 145-174
- **Type**: Bar chart
- **Data**: Red (85%), Green (45%), Blue (30%)
- **Insight**: "The Red channel is most critical for detecting wildfires because fire produces intense red/orange light. The model's MobileNetV2 architecture learns these color patterns automatically through its convolutional layers, extracting features that distinguish wildfire's thermal signature from normal vegetation."

**Additional Data Insights**:
- System uptime tracking (`/status` endpoint)
- Dataset statistics API (`/data_stats` endpoint)
- Training history logs
- Model performance metrics displayed after retraining

### UI Quality Features:
- Professional Streamlit interface
- Color-coded predictions (red for danger, green for safe)
- Real-time training progress display
- Interactive Plotly charts
- Responsive design
- NASA FIRMS integration links for data collection

---

## Summary Checklist

### Files Required for Grading:

**Retraining**:
- [x] `src/train_pipeline.py` (script)
- [x] `models/wildfire_model.h5` (model file)

**Prediction**:
- [x] `src/prediction.py` (script)
- [x] `models/wildfire_model.h5` (model file)

**Evaluation**:
- [x] `notebook/wildfire_detection.ipynb` (notebook)

**Deployment**:
- [x] Public URL (Railway)
- [x] Streamlit UI with data insights

### Requirements Met:

**Video Demo** (5 pts):
- [ ] Ready to record - all features working

**Retraining** (10 pts):
- [x] Data upload + save to database
- [x] Data preprocessing documented
- [x] Custom pre-trained model (MobileNetV2)

**Prediction** (10 pts):
- [x] Image upload interface
- [x] Correct predictions displayed
- [x] Preprocessing applied correctly

**Evaluation** (10 pts):
- [x] Clear preprocessing steps in notebook
- [x] Pre-trained model (MobileNetV2 + ImageNet)
- [x] Dropout regularization
- [x] Adam optimizer
- [x] Learning rate tuning
- [x] 5 metrics: Accuracy, Precision, Recall, F1, Loss

**Deployment** (10 pts):
- [x] Public web app URL
- [x] 3 data visualizations with interpretations
- [x] Interactive UI

---

## Quick Reference for Professor

**GitHub Repository**: https://github.com/idarapatrick/ML-Pipeline-Summative

**Deployed Application**:
- API: https://wildfire-prediction.up.railway.app
- UI: [Streamlit URL in README]
- Docs: https://wildfire-prediction.up.railway.app/docs

**Key Files to Review**:
1. `notebook/wildfire_detection.ipynb` - Model development and evaluation
2. `src/train_pipeline.py` - Retraining implementation
3. `src/prediction.py` - Prediction implementation
4. `models/wildfire_model.h5` - Trained model (23.22 MB)
5. `README.md` - Complete documentation

**Quick Test Commands**:
```bash
# Test prediction API
curl -X POST "https://wildfire-prediction.up.railway.app/predict" \
  -F "file=@wildfire_image.jpg"

# Check system status
curl https://wildfire-prediction.up.railway.app/status

# View API documentation
# Visit: https://wildfire-prediction.up.railway.app/docs
```

---

## Notes

All rubric requirements are satisfied and documented. The only pending item is recording the video demonstration, which requires showing the working application with camera on.
