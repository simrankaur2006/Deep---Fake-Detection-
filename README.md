# 📌 Overview

This project implements a deep learning-based image classification system to detect whether an image is real or AI-generated (deepfake).

# 🚀 Features
✅ Binary classification (Real vs Fake)
✅ Works with custom datasets 
✅ Train from scratch or fine-tune pretrained models
✅ Evaluation metrics:
Accuracy
Precision , Recall
F1-score
✅ Simple prediction script
✅ Optional web interface Flask 
✅ GPU support (Colab and CUDA)

# 🧠 Model Architecture
Input Image → Feature Extractor  → Fully Connected Layers → Output (Real/Fake)

# 📂 Project Structure
deepfake-detection/
│── dataset/
│   ├── real/
│   └── fake/
│
│── models/
│   └── model.pth
│
│── train.py        # Training script
│── predict.py      # Inference script
│── app.py          # Web app 
│── utils.py        # Helper functions
│── requirements.txt
│── README.md

# ⚙️ Installation
git clone 
cd deepfake-detection

pip install -r requirements.txt

# 📊 Dataset Format
dataset/
├── real/
│   ├── img1.jpg
│   ├── img2.jpg
│
├── fake/
│   ├── img1.jpg
│   ├── img2.jpg

# 🏋️ Training

Run the training script:
python train.py

# 🔍 Prediction
python predict.py --image path/to/image.jpg

# 🌐 Web App (Optional)

Run the UI:   python app.py

# 📈 Evaluation Metrics
Accuracy
Precision
Recall
F1 Score
Confusion Matrix

# 🛠 Tech Stack
1. Python
2. PyTorch
3. Hugging Face Transformers
4. OpenCV
5. NumPy 
6. Pandas
7. Flask 

# 🤝 Contributing

      Contributions are welcome!

             Fork the repository
             Create a new branch
             Make your changes
             Submit a pull request
