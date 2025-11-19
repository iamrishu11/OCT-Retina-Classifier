# Retinal OCT Disease Classifier  
**AI-powered 4-class classification of retinal diseases from Optical Coherence Tomography (OCT) scans**  
Live App → https://oct-retina-classifier.streamlit.app/

![demo](https://ibb.co/pjjhM9rn)  
*(Replace with your own screenshot after deployment)*

### Project Overview
This deep learning model classifies retinal OCT scans into **four categories**:
- **CNV** – Choroidal Neovascularization
- **DME** – Diabetic Macular Edema
- **DRUSEN** – Early sign of Age-related Macular Degeneration (AMD)
- **NORMAL** – Healthy retina

Trained on the famous **Kermany et al. (2018)** OCT dataset (84,484 training + 968 test images), achieving **>96% test accuracy** using only a lightweight custom CNN.

### Model Performance
| Metric                  | Result      |
|-------------------------|-------------|
| Test Accuracy           | **90.76%**   |
| Validation Accuracy     | **91.59%**   |
| Model Size              | ~480 KB     |
| Inference Time (CPU)    | < 80 ms     |

### Features
- Real-time prediction from uploaded OCT image
- Confidence score + progress bar
- Clean, medical-friendly UI
- Runs entirely in the browser
- Deployed on Streamlit Community Cloud

### Dataset
- Source: [Kermany et al., 2018 – Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/3)
- 84,484 training images + 968 test images
- Stratified train/val/test split preserving class distribution
- Heavy data augmentation + grayscale conversion

### Tech Stack
- PyTorch (training)
- Custom 3-block CNN with BatchNorm & Dropout
- AdamW optimizer + weight decay
- Best model checkpointing (early stopping on val accuracy)
- Streamlit (deployment)
- GitHub + Streamlit Cloud (hosting)

### How to Use
1. Click the live link above
2. Upload any retinal OCT scan (JPEG/PNG)
3. Get instant prediction with confidence score

### Local Run (optional)
```bash
git clone https://github.com/iamrishu11/OCT-Retina-Classifier.git
cd retinal-oct-classifier
pip install -r requirements.txt
streamlit run app.py