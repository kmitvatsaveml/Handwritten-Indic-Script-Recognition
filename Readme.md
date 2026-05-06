# ✍️ Handwritten Telugu Vowel Recognition

A deep learning project that classifies **handwritten Telugu vowels** using a lightweight CNN (~286K parameters) trained from scratch.

---

## 📌 Overview

This project builds and deploys a Convolutional Neural Network (CNN) to recognize 6 Telugu vowel characters:

**A, Aa, Ai, E, Ee, U**

- Dataset: 1200 images (200 per class)
- Model: 3-layer CNN
- Framework: PyTorch
- Deployment: Streamlit Web App

---

## 🚀 Features

- 🎨 Draw Telugu characters on an interactive canvas
- 🤖 Real-time predictions using trained CNN
- 📊 Confidence scores + probability visualization
- 🎯 Practice mode with feedback system

---

## 🧠 Model Details

- Architecture: 3-layer CNN (16 → 32 → 64 filters)
- Parameters: ~286,000
- Input: 64×64 grayscale images
- Optimizer: Adam
- Loss: CrossEntropy

### 📈 Performance

- **Validation Accuracy:** 99.44%
- **Test Accuracy:** 98.33%

---

## 📊 Ablation Highlights

| Experiment    | Best Choice |
| ------------- | ----------- |
| Dropout       | 0.25        |
| Augmentation  | Basic       |
| Learning Rate | 1e-3        |
| Optimizer     | Adam        |

---

## 📂 Project Structure

```
├── app.py                     # Streamlit app
├── telugu_vowel_model.pth    # Trained model
├── training_notebook.ipynb   # Training + experiments
├── images/                   # Saved plots (EDA + ablation)
│   ├── eda_samples.png
│   ├── eda_dimensions.png
│   ├── eda_class_dist.png
│   ├── training_history.png
│   ├── ablation_a2_dropout.png
│   ├── ablation_a3_augmentation.png
│   ├── ablation_a4_lr.png
│   ├── ablation_a5_optimizer.png
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone YOUR_GITHUB_LINK
cd telugu-vowel-recognition

pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 🧪 Training

To retrain the model:

```bash
jupyter notebook training_notebook.ipynb
```

---

## 🌐 Deployment

- Streamlit Cloud / HuggingFace Spaces supported

👉 Replace with your links:

- **Live App:** [YOUR_DEPLOYMENT_LINK](https://huggingface.co/spaces/Akulareddy/telugu-vowel-recognizer)
- **GitHub Repo:** [YOUR_GITHUB_LINK](https://github.com/kmitvatsaveml/Handwritten-Indic-Script-Recognition.git)

---

## ⚠️ Limitations

- Small dataset (1200 samples)
- Only 6 Telugu vowels
- Sensitive to drawing quality

---

## 🔮 Future Work

- Extend to full Telugu alphabet
- Improve robustness to noisy inputs
- Explore deeper CNNs / Transformers

---

## 📚 References

- Telugu 6 Vowel Dataset (Kaggle)
- PyTorch Documentation
- Optuna Hyperparameter Optimization

---

## 👨‍💻 Author

Akula Amanaganti 2025202034
Poreddy Srivatsav Reddy 2025121016
