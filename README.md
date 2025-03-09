# 🧪 Toxicity Analyzer

**Toxicity Analyzer** is a deep learning-powered tool that detects different forms of toxic language in text. It uses a **Bidirectional LSTM** model to classify input text into **six toxicity categories**:  
✔️ Toxic  
✔️ Severe Toxic  
✔️ Obscene  
✔️ Threat  
✔️ Insult  
✔️ Identity Hate  

The model is deployed on **Hugging Face** and can be accessed through a **Streamlit web app**.

---

## 🚀 Features

- **Real-time Toxicity Detection**: Enter any text and get a toxicity breakdown.
- **Multi-label Classification**: Detects multiple toxic traits in a single comment.
- **Probability-Based Scores**: Shows **toxicity likelihood** for each category.
- **Interactive UI**: Intuitive interface for easy analysis.
- **Deployed Model**: Runs on **Hugging Face Spaces**, eliminating the need for local setup.

---

## 🎯 How It Works

1. The user enters text into the **Toxicity Analyzer**.
2. The **pre-trained LSTM model** processes the input.
3. The model **predicts toxicity scores** across six categories.
4. The app displays:
   - **Binary results** (Yes/No for each category).
   - **Percentage-based toxicity scores**.
   - **Visual progress bars** to indicate severity.

---

## 📊 Model Details

The **Bidirectional LSTM** model has the following architecture:
- **Embedding Layer**: Transforms words into dense vectors.
- **Bidirectional LSTM**: Captures contextual meaning from both directions.
- **Fully Connected Layers**: Deep layers with ReLU activations.
- **Sigmoid Output Layer**: Generates probability scores for toxicity.

The model was trained on the **Jigsaw Toxic Comment Dataset**, ensuring a well-balanced classification.

---

## 🔍 Evaluation Metrics

The model is evaluated using:
- **Precision** (Correctness of toxic predictions)
- **Recall** (Ability to detect toxic comments)
- **Categorical Accuracy** (Overall correctness)

---

## 🌐 Try It Online

The app is **live on Hugging Face**! Try it out here:  
🔗 **[Toxicity Analyzer - Live Demo](https://huggingface.co/spaces/eemas/Toxicity-Analyzer)**  
