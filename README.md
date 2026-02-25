# 🛰️ Building Damage Assessment -- xView2

Streamlit web application for automatic **building damage assessment**
from pre- and post-disaster satellite imagery using deep learning
(xView2 / xBD dataset).

The application allows users to upload or provide satellite image pairs
and obtain:

-   🏢 Building segmentation mask\
-   🔥 Damage severity classification (No / Minor / Major / Destroyed)\
-   🎨 Color-coded visualization of predictions

------------------------------------------------------------------------

## 📌 Project Overview

Natural disasters require rapid and scalable damage assessment.\
This project leverages deep learning to analyze **satellite
imagery** (before and after a disaster) and automatically estimate
building damage severity.

System architecture:

User → Streamlit App → FastAPI Backend → Deep Learning Model →
Prediction Images → Streamlit Display

-   **Frontend:** Streamlit\
-   **Backend:** FastAPI\
-   **Model:** U-Net-based architecture trained on xView2/xBD dataset\
-   **Task:** Semantic segmentation + damage classification

------------------------------------------------------------------------


## 📂 Repository Structure

Project_xView2/ 
│ └── app/                                \# Streamlit application\
│     ├── ds_project_homepage.py\
│     ├── overview.py\
│     ├── damage_estimator.py\
│     ├── model_page.py\
│     ├── past_disasters.py\
│     ├── config.toml\
│     └── imgs/                           \# Images used for the application\
│ └── src/                                \# Core scripts\
│     └── main.ipynb\
│ ├── data/                               \# Training data not included in repository !!\
│ └── utils/
│     ├── augmenter.ipynb\                \# Data augmentation script to increase training data quantity\
│     ├── create_labels_from_json.ipynb\  \# Script to convert geospatial building annotations into a segmentation mask image for training your damage model.\
│     ├── read_training_log.ipynb\        \# Script to read output training logs and plot figures\
│     └── figures/
│         ├── f1_score.png\
│         ├── accuracy.png\
│         └── loss.png\
│ └── models/                             \# Saved model weights (U-net v11)\
│     ├── unet_v11_multiclass_epoch_19.h5
│     ├── unet_v11_multiclass_epoch_20.h5
│     └── unet_v11_training_log.txt\
├── requirements.txt\
├── .gitignore\
└── README.md

------------------------------------------------------------------------

## 📊 Dataset

Based on the **xView2 Challenge dataset (xBD)**.

-   Pre-disaster and post-disaster satellite image pairs\
-   Building footprints\
-   4 damage severity levels:
    -   No damage\
    -   Minor damage\
    -   Major damage\
    -   Destroyed

Reference:\
Gupta et al., *xBD: A Dataset for Assessing Building Damage from
Satellite Imagery*, 2019.

Dataset is **not included** in this repository.

------------------------------------------------------------------------


## 🖥️ Application Features

### 🔹 Load from file

Upload: - Pre-disaster image (.png) - Post-disaster image (.png)

Click **Predict** to receive: - Damage severity map - Building mask -
Color legend

### 🔹 Load from URL

Provide direct image URLs for: - Pre-disaster - Post-disaster

### 🔹 Load dual image

If both images are stitched together: - Split horizontally - Split
vertically

------------------------------------------------------------------------

## 🧠 Model Description

The backend model:

-   Takes pre- and post-disaster images\
-   Classifies building damage into 4 categories\
-   Outputs a color-coded segmentation map

Damage severity classes:

  Severity       Color
  -------------- --------
  No damage      Cyan
  Minor damage   Yellow
  Major damage   Orange
  Destroyed      Red


------------------------------------------------------------------------

## 📈 Future Improvements

-   Overlay prediction over original imagery\
-   Confidence maps\
-   Batch inference\
-   Dockerized deployment\
-   GPU acceleration

