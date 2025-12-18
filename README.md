# Fake News Detection Using Machine Learning and Deep Learning

## 📌 Project Overview

The rapid spread of misinformation online has made fake news detection a critical challenge in modern society. This project focuses on building an end-to-end **Fake News Detection system** using both **traditional Machine Learning** and **Deep Learning** techniques to automatically classify news articles as *Fake* or *Real*.

This project was completed as part of an **academic course final** and was later **extended with additional experimentation, performance comparison, and real-world deployment** using **Streamlit**.

The project covers the complete ML lifecycle — from data cleaning and exploratory analysis to model training, evaluation, and real-world deployment using **Streamlit**.

---

## 🎯 Project Objective

To design and deploy a robust text classification system that can accurately detect fake news articles by leveraging linguistic patterns and contextual representations learned through ML and DL models.

---

## 📂 Dataset

* **Source**: Kaggle – Fake and Real News Dataset (clmentbisaillon)
* **Classes**:

  * `0` → Fake News
  * `1` → Real News
* **Features Used**:

  * News title
  * News article text

After preprocessing and deduplication, the dataset was balanced and cleaned to ensure reliable training and evaluation.

---

## 🧹 Text Preprocessing Pipeline

Key preprocessing steps include:

* Removal of duplicates and empty samples
* Regex-based noise cleaning (URLs, metadata, social handles)
* Numeric normalization
* Title and body text merging
* Lowercasing and whitespace normalization

This pipeline ensures consistency across ML models, DL models, and deployment.

---

## 🤖 Models Implemented

### 🔹 Machine Learning Models (TF-IDF based)

* Logistic Regression ✅ *(Best ML model)*
* Multinomial Naive Bayes
* Random Forest

**Best ML Accuracy**: **98.1%** (Logistic Regression)

---

### 🔹 Deep Learning Models (GloVe embeddings)

* LSTM
* BiLSTM
* GRU ✅ *(Best DL model & deployed)*

**Best DL Accuracy**: **99.16%** (GRU)

---

## 📊 Model Evaluation

Models were evaluated using:

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix
* Training & validation curves (DL models)

Both ML and DL models demonstrate strong generalization, with GRU achieving the highest overall performance.

---

## 🚀 Deployment

The best-performing **GRU model** was deployed as a web application using **Streamlit**.

### 🔗 Live Demo

👉 [https://fake-news-system-app.streamlit.app/](https://fake-news-system-app.streamlit.app/)

The deployed app allows users to input custom news text and instantly receive a fake/real prediction.

### 🔗 Deployment Repository

A separate lightweight repository is used for deployment:

* Contains only inference-related code and artifacts
* Optimized for Streamlit Cloud execution

---

## 🗂️ Project Structure

```
fake-news-detection-ml-dl/
│
├── notebooks/
│   └── ml_dl_fake_news_detection.ipynb
│
├── reports/
│   └── 2165_A_MLL_Project_Report.pdf
│
├── slides/
│   └── 2165_A_MLL_Project_Slide.pptx
│
├── requirements.txt
├── README.md
└── .gitignore

```

---

## How to Run Locally

Follow these steps to run the project and explore the experiments:

1. **Clone the repository**

```bash
git clone https://github.com/bobibarua/fake-news-detection-ml-dl.git
cd fake-news-detection-ml-dl
```

2. **Install dependencies**
   Make sure you have Python 3.x installed. Then run:

```bash
pip install -r requirements.txt
```

> Ensure that `requirements.txt` includes all necessary packages such as `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`, `wordcloud`, etc.

3. **Open and run the notebook**

```bash
jupyter notebook notebooks/ml_dl_fake_news_detection.ipynb
```

This notebook contains the full workflow:

* Data loading and preprocessing
* ML & DL model training and evaluation
* Visualizations such as word clouds and confusion matrices

4. **View the deployed app (optional)**
   The Streamlit app is deployed separately and can be accessed here:
   [Live Demo](https://fake-news-system-app.streamlit.app/)

> Note: This repository contains the training notebook, report, and slides. The deployment files for Streamlit are maintained in a separate repository.

---

## 🧪 Reproducibility & Engineering Practices

* Models and vectorizers saved using `joblib` and Keras `.h5` format
* Hyperparameters stored in JSON metadata
* Tokenizer and label mapping reused across training and deployment
* Clean separation between experimentation and production code


## 🔮 Future Improvements

* Transformer-based models (BERT, RoBERTa)
* Multilingual fake news detection
* Explainability using SHAP or LIME
* Continuous model updates with new data

---

## 👤 Author

**Bobi Barua**
GitHub: [https://github.com/bobibarua](https://github.com/bobibarua)

---

⭐ If you find this project useful, consider giving it a star!

