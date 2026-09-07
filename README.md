# 🚀 Spaceship Titanic — Supervised Learning Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Kaggle-Spaceship%20Titanic-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle Competition">
  <img src="https://img.shields.io/badge/Validation%20Accuracy-81.4%25-2ea44f?style=flat-square" alt="Validation Accuracy">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="License: MIT">
</p>

<p align="center">
  Predicting which passengers were transported to an alternate dimension after the <em>Spaceship Titanic</em> collided with a spacetime anomaly — an end-to-end supervised learning pipeline built with scikit-learn.
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Key Techniques](#-key-techniques)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🛰️ Overview

This repository contains a full binary classification workflow for the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition — predicting the boolean target `Transported` for each passenger.

The project is split into two notebooks with distinct purposes:

| Notebook | Purpose |
|---|---|
| `01_exploration_and_visualization.ipynb` | Open-ended EDA — distributions, missingness, correlation, mutual information |
| `02_supervised_learning_pipeline.ipynb` | Clean, reproducible, leakage-safe path from raw data to Kaggle submission |

---

## 📁 Project Structure

```
spaceship-titanic/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── 01_exploration_and_visualization.ipynb
│   └── 02_supervised_learning_pipeline.ipynb
├── submission.csv
└── README.md
```

---

## 🗂️ Dataset

The data describes ~8,700 passengers aboard the *Spaceship Titanic*, including:

- **Demographics** — `Age`, `HomePlanet`, `Destination`, `VIP`
- **Travel details** — `Cabin` (Deck/Num/Side), `CryoSleep`, `PassengerId` (encodes travel group)
- **Onboard spending** — `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`
- **Target** — `Transported` (boolean, ~50/50 balanced)

---

## 🔬 Approach

1. **EDA** — distributions, missing-value audit, target balance, correlation, and mutual information analysis
2. **Feature engineering** — `Cabin` split into `Deck`/`CabinNum`/`Side`, `TotalSpend` (sum of all onboard spending), `GroupSize` (from `PassengerId`)
3. **Leakage-safe split** — train/validation split performed *before* fitting any preprocessing
4. **Preprocessing pipeline** — `ColumnTransformer` with median imputation + scaling for numeric features, most-frequent imputation + one-hot encoding for categorical features
5. **Model comparison** — 5-fold stratified cross-validation across Logistic Regression, Random Forest, HistGradientBoosting, and SVC
6. **Hyperparameter tuning** — `GridSearchCV` on the winning model
7. **Final evaluation** — validation accuracy, precision/recall/F1, confusion matrix
8. **Feature importance** — permutation importance on the held-out validation set
9. **Refit & submit** — retrained on 100% of labeled data, predictions written to `submission.csv`

---

## 📊 Results

| Model | CV Accuracy (mean) |
|---|---|
| Logistic Regression (baseline) | 0.793 |
| Random Forest | 0.799 |
| SVC | 0.801 |
| **HistGradientBoosting (selected)** | **0.810** |

**Final tuned model — validation set:**

| Metric | Not Transported | Transported |
|---|---|---|
| Precision | 0.82 | 0.81 |
| Recall | 0.80 | 0.83 |
| F1-score | 0.81 | 0.82 |

**Validation accuracy: `0.8143`**

---

## ⚙️ Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/spaceship-titanic.git
cd spaceship-titanic

# Install dependencies
pip install pandas numpy scikit-learn matplotlib jupyter

# Place train.csv and test.csv from Kaggle into data/
# Run the pipeline
jupyter notebook notebooks/02_supervised_learning_pipeline.ipynb
```

Running all cells in `02_supervised_learning_pipeline.ipynb` produces `submission.csv`, ready to upload to Kaggle.

---

## 🧠 Key Techniques

- ✅ Leakage-safe preprocessing (`Pipeline` + `ColumnTransformer`, fit only on training folds)
- ✅ Stratified k-fold cross-validation for reliable model comparison
- ✅ Systematic hyperparameter search via `GridSearchCV`
- ✅ Model-agnostic feature importance via permutation importance
- ✅ Refit on full data before final prediction

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `Jupyter`

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
  <sub>Built as a reference template for supervised learning classification projects.</sub>
</p>
