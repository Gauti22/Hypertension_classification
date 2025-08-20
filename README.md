# Hypertension Classification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![DVC](https://img.shields.io/badge/MLOps-DVC-success)
![MLflow](https://img.shields.io/badge/Experiment%20Tracking-MLflow-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-brightgreen)

## 📌 Project Overview

This project aims to **classify whether a patient has Hypertension** based on health-related features. The pipeline integrates **data preprocessing, model training, experiment tracking (MLflow), data versioning (DVC)**, and **deployment with Streamlit**. The goal is to provide a reproducible and interactive ML workflow for healthcare classification tasks.

---

## 📂 Repository Structure

```
hypertension_classification/
│── data/                # Raw and processed data (DVC-tracked)
│── files/               # DVC metadata files
│── mlartifacts/         # MLflow tracked models & experiments
│── streamlit_app.py     # Streamlit web application
│── params.yaml          # Configurations and hyperparameters
│── requirements.txt     # Python dependencies
│── dvc.yaml             # DVC pipeline definition
│── dvc.lock             # DVC lock file
│── .dvcignore           # Ignore patterns for DVC
│── .gitignore           # Git ignore rules
```

---

## 🚀 Features

* Automated **data preprocessing & feature engineering**
* **Classification models** (baseline + advanced)
* **Experiment tracking with MLflow**
* **DVC integration** for reproducible pipelines
* **Interactive Streamlit app** for predictions
* Modular and scalable MLOps design

---

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/hypertension_classification.git
   cd hypertension_classification
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install DVC and MLflow:

   ```bash
   pip install dvc mlflow
   ```

---

## 📊 Usage

### 1. Run ML pipeline

```bash
dvc repro
```

### 2. Train and track experiments with MLflow

```bash
mlflow ui
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) to explore experiments.

### 3. Run Streamlit app

```bash
streamlit run streamlit_app.py
```

---

## ⚙️ Configuration

Modify model settings in `params.yaml`. Example:

```yaml
train:
  test_size: 0.2
  random_state: 42
model:
  type: random_forest
  n_estimators: 200
  max_depth: 10
```

---

## 📈 Results

* Baseline: Logistic Regression
* Advanced: Random Forest, XGBoost
* Metrics tracked: Accuracy, Precision, Recall, F1-score, ROC-AUC

> Check MLflow UI and `mlartifacts/` for detailed experiment results.

---

## 🔮 Future Work

* Hyperparameter optimization with Optuna
* Deploy model as REST API (FastAPI/Flask)
* Integrate CI/CD workflows with GitHub Actions
* Add SHAP/Explainability dashboards

---

## 🤝 Contribution

Contributions are welcome! To contribute:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature-name`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**.
