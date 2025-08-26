**🛡️ Multimodal-Phishing-Detection-using-Gen-AI **

📌 Overview

The Phishing Detector is an AI-powered system that identifies phishing attempts in online content such as emails, webpages, and images. It combines Natural Language Processing (NLP) with Optical Character Recognition (OCR) to extract and analyze text, then classifies the content as either phishing or legitimate.

This project leverages:

-EasyOCR for extracting text from images and screenshots.

-RoBERTa Transformer Model for advanced text embeddings.

-Custom PyTorch Classifier for phishing detection.

-Evaluation metrics (Accuracy, Precision, Recall, F1-Score) for performance tracking.

---
✨ Features

📖 Text extraction from emails, images, and HTML pages.

🔍 Deep NLP embeddings using RoBERTa.

🤖 Custom PyTorch model for phishing classification.

📊 Performance metrics to evaluate model accuracy.

⚡Gpu acceleration with CUDA (if available). 
---

**⚙️ Installation**

Clone the repository and install dependencies:

->git clone https://github.com/your-username/phishing_detector.git
cd phishing_detector
pip install -r requirements.txt


->Or install manually inside Jupyter Notebook:

->%pip install transformers easyocr tqdm scikit-learn pandas pillow
%pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


->System dependencies:

apt install -y poppler-utils tesseract-ocr
---
**🚀 Usage**

1)Open the Jupyter Notebook:

jupyter notebook phishing_detector.ipynb


2)Update the paths inside the notebook if needed:

.TrainingSet_Modified.xlsx → dataset file

.images/ → folder with email/webpage screenshots

.Alloutputs/ → HTML outputs

3)Run all cells to:

.Extract text features from dataset & images.

.Train the phishing detection model.

.Evaluate the model with test data.
---
**📊 Results**

The project reports:

.Accuracy

.Precision

.Recall

.F1-Score

These metrics ensure balanced detection between phishing and legitimate cases.
---
**🔮 Future Improvements**

.Add support for real-time email/webpage scanning.

.Improve dataset size for better generalization.

.Deploy as a web app / API for integration with mail clients.

.Use hybrid models (NLP + URL feature engineering) for more robust phishing detection.
