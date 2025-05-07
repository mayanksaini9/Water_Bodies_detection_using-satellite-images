# 🌊 Water Bodies Detection from Satellite Images using DeepLabV3+

This project performs **semantic segmentation** to detect water bodies from satellite images using a **DeepLabV3+** model with an **Xception backbone** in TensorFlow/Keras.

![Water Body Segmentation](https://img.shields.io/badge/model-DeepLabV3%2B-blue)  
![License](https://img.shields.io/github/license/mayanksaini9/water-body-segmentation)

## 📁 Dataset

The dataset is from [Kaggle - Satellite Images of Water Bodies](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies). It contains:
- `images/` folder with satellite images.
- `masks/` folder with corresponding binary masks (1 for water body, 0 for background).
- Total image-mask pairs: **2841**

## 🧠 Model

We used the **DeepLabV3+** architecture with the **Xception** backbone, a powerful deep CNN designed for segmentation tasks.

### Key Features:
- **Data Augmentation**: Rotation, zoom, width/height shift, flipping, etc.
- **Batch Normalization**: Added after convolution layers for training stability.
- **Dice Loss**: Custom loss function for imbalanced segmentation data.
- **EarlyStopping**, **ModelCheckpoint**, **ReduceLROnPlateau** used during training.

## 🛠️ Setup Instructions

### 1. Clone the Repository
bash
git clone https://github.com/yourusername/water-body-segmentation.git
cd water-body-segmentation


2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt


4. Prepare Dataset
Download the dataset from Kaggle.

Place the images and masks folders in the project directory.

5. Train the Model
python
Copy
Edit
python train.py
📊 Results & Evaluation
Final validation Dice score: ~0.90


Confusion Matrix
A confusion matrix is plotted to evaluate pixel-wise performance.

Visualizations
python
Copy
Edit
# Visualization examples included in notebook
📁 Project Structure
bash
Copy
Edit
.
├── images/                  # Input images
├── masks/                  # Binary masks
├── model/                  # Saved models
├── train.py                # Training script
├── predict.py              # Prediction script
├── utils.py                # Utility functions
├── nkt2.ipynb              # Jupyter notebook (full workflow)
└── README.md               # This file
📌 To-Do
 Add support for test-time augmentation

 Deploy model with Streamlit or Flask

 Export model to ONNX or TFLite for mobile use

👨‍💻 Author
Mayank Saini
Mohd Umar Siddiqui
GitHub https://github.com/mayanksaini9
LinkedIn https://www.linkedin.com/in/mayank-saini-a3b566257/



📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

yaml
Copy
Edit

