## 📌 Overview

This project performs **sentiment analysis** on a dataset of comments from the **Digikala website** using **Support Vector Machine (SVM)** and **Random Forest Classifier (RFC)**. The dataset is stored in `sample_dataset.json`, and stopwords are loaded from `stopwords.txt`. The pipeline includes data preprocessing, text vectorization using **TF-IDF**, and training/testing of classification models.

## 📂 Files and Structure

- **`sample_dataset.json`** - JSON file containing comments and their sentiment labels from **Digikala**.
- **`stopwords.txt`** - List of stopwords to be removed from the text.

## 🛠️ How It Works

1. **Loading Data**: Reads the dataset from `sample_dataset.json` and extracts `comment` and `sentiment` fields using `nested_lookup`.
2. **Stopwords Removal**: (Optional) Removes stopwords from comments.
3. **Text Vectorization**: Converts text data into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
4. **Splitting Data**: The dataset is split into training (70%) and testing (30%) sets.
5. **Model Training & Evaluation**:
   - **SVM (Support Vector Machine)**: Trains an SVM classifier and evaluates its performance.
   - **Random Forest Classifier**: Trains an RFC model to handle imbalanced data.
6. **Performance Metrics**: Displays **confusion matrix, precision, recall, f1-score, and accuracy** for each model.

## 📈 Handling Imbalanced Data

By addressing the **class imbalance issue**, the accuracy improved by **8%**, showing that balancing the dataset significantly impacts the performance of sentiment analysis models.

## 🔍 Key Functions Explained

- **`remove_stopwords(txt)`**: Removes stopwords from the text.
- **`vectorize_tfidf(txt)`**: Converts text into a TF-IDF matrix.
- **`split_data(X, y, test_size)`**: Splits data into train/test sets.
- **`define_run_model(model, X_train, X_test, y_train)`**: Trains a machine learning model and returns predictions.
- **`check_result(y_test, y_pred)`**: Evaluates model performance with classification metrics.