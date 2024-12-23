# Breast Cancer Classification with SVM using scikit-learn

## Project Overview

This project explores the **Support Vector Machine (SVM)** algorithm for breast cancer classification using the **scikit-learn** library in Python. The goal is to predict whether a breast tumor is **malignant** or **benign** based on features extracted from digitized images of breast cancer biopsies.

---

## Project Structure

-  Script containing the implementation.
- File listing dependencies (optional).

---

## Dependencies


Install the required dependencies with:

```bash
pip install -r requirements.txt
```

---

## Steps

### 1. Load the Dataset

- Stores features and target labels in a **Pandas DataFrame**.

### 2. Split the Dataset

- Ensures a **20% split** for testing using `test_size=0.2`.
- Employs `random_state` for reproducibility.

### 3. Preprocess the Data


- Scales both training and testing data using the same scaler object.

### 4. Train the SVM Model

- Creates an SVM model with:
  - **Linear kernel**
  - Regularization parameter **C=1**
- Trains the model on the training data using the `` method.

### 5. Evaluate the Model


- Calculates performance metrics using:
  - **Accuracy score**
  - **Classification report**
- Reports **precision**, **recall**, **F1-score**, and **support** for each class.

### 6. Make Predictions

- Allows predictions on new data points using the trained model.
- Demonstrates how to reshape a single data point for prediction.

---

## Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/sarwat-mumtaz/breast-cancer-classification-svm.git
   cd breast-cancer-classification-svm
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python svm_breast_cancer.py
   ```

---

## Conclusion

This project provides a basic implementation of SVM for breast cancer classification. You can extend this by:

- Trying different SVM kernels (e.g., **polynomial**, **RBF**).
- Tuning hyperparameters for potentially better performance.
- Exploring feature engineering techniques.
- Building more complex models for classification.

---

## License

This project is licensed under the [MIT License](LICENSE).

