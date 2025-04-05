# Assignment 3 - Multinomial Logistic Regression with CI/CD

This project is part of Assignment 3 for the Machine Learning course.  
The goal is to implement a multinomial logistic regression model from scratch, evaluate it, and apply CI/CD practices using GitHub Actions and MLflow.

---

## Project Objectives

1. **Model Implementation**  
   - Implemented a custom Multinomial Logistic Regression class in `LogisticRegression.py`
   - Supported options for batch, mini-batch, and stochastic gradient descent
   - Added L2 Regularization (Ridge) as an option

2. **Evaluation**  
   - Dataset was preprocessed and selling price was discretized into 4 classes
   - Metrics evaluated: Accuracy, Precision, Recall, F1-score, Macro and Weighted metrics

3. **CI/CD & MLflow**  
   - Unit tests created in `test_model.py` to verify model input and output shape  
   - GitHub Actions configured in `.github/workflows/test.yml`  
     - Automatically runs tests and logs model to MLflow  
   - Final model is logged to [MLflow Tracking Server](https://mlflow.ml.brain.cs.ait.ac.th/)  
     - Registered as: `st124879-a3-model`

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/ML_A3_Arisa.git
cd ML_A3_Arisa

# Install dependencies
pip install -r requirements.txt

# Run unit tests
python -m unittest discover -s . -p "test_*.py"

# Train and log model
python train.py

```

---

## Author

Arisa Phanmaneelak
Student ID: st124879
Asian Institute of Technology