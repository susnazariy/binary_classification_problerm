# Breast Cancer Classification

A machine learning project for classifying breast tumors as malignant or benign using the Wisconsin Breast Cancer dataset. This project explores multiple classification algorithms with hyperparameter tuning to find the optimal model.

## Overview

This project aims to predict whether a breast tumor is malignant (cancerous) or benign based on features computed from digitized images of fine needle aspirates (FNA). It follows a complete ML pipeline from data exploration to model comparison and selection.

## Dataset

The Wisconsin Breast Cancer dataset contains **569 samples** with **30 features** computed from cell nuclei characteristics:

| Feature Category | Features (mean, SE, worst) |
|-----------------|---------------------------|
| Size | radius, perimeter, area |
| Texture | texture (std dev of gray-scale values) |
| Shape | smoothness, compactness, concavity, concave points, symmetry, fractal dimension |

**Target Variable:**
| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| Malignant | 0 | 212 | 37.3% |
| Benign | 1 | 357 | 62.7% |

## Project Pipeline

### 1. Data Cleaning
- Verified no missing values
- Detected outliers using IQR method

### 2. Exploratory Data Analysis
- Target distribution analysis
- Feature distributions by class (histograms, boxplots)
- Correlation analysis with target
- Pairplot visualization of top features

### 3. Feature Engineering
- **StandardScaler**: Normalized all features
- **PCA Analysis**: Dimensionality reduction (10 components for 95% variance)
- Identified highly correlated feature pairs (r > 0.9)

### 4. Model Training & Comparison

Implemented a custom `Classifier` class for streamlined training. Models tested:

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear baseline classifier |
| Decision Tree | Tree-based classifier |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential ensemble method |
| SVM | Support Vector Machine |
| KNeighbors | Instance-based learning |
| Naive Bayes | Probabilistic classifier |
| AdaBoost | Adaptive boosting |

### 5. Hyperparameter Tuning
- Used `GridSearchCV` with 5-fold cross-validation
- Compared default vs. tuned model performance

## Results

**Final Model Comparison (GridSearchCV):**

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| KNeighbors | 0.9825 | 0.9730 | 1.0000 | 0.9863 | 0.9835 |
| SVM | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9937 |
| Logistic Regression | 0.9737 | 0.9726 | 0.9861 | 0.9793 | 0.9957 |
| Gradient Boosting | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9904 |
| AdaBoost | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9818 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9932 |
| Naive Bayes | 0.9298 | 0.9444 | 0.9444 | 0.9444 | 0.9868 |
| Decision Tree | 0.9211 | 0.9565 | 0.9167 | 0.9362 | 0.9163 |

**Best Models:** KNeighbors and SVM achieved the highest F1-Score (~0.986)

## Key Findings

1. **Dataset Quality**: Clean dataset with no missing values
2. **Class Imbalance**: Slightly imbalanced (62.7% Benign, 37.3% Malignant)
3. **Feature Correlation**: Many features highly correlated, especially area-related measurements
4. **Best Features**: Worst concave points, worst perimeter, and worst radius are most predictive
5. **Model Performance**: Most models achieve >95% accuracy due to well-separated classes

## Feature Importance (Random Forest)

Top predictive features:
1. worst concave points
2. worst perimeter
3. worst radius
4. mean concave points
5. worst area

## Tech Stack

- **Python 3.13**
- **pandas** — Data manipulation
- **NumPy** — Numerical computing
- **scikit-learn** — ML algorithms, preprocessing & metrics
- **Matplotlib & Seaborn** — Visualization

## Project Structure

```
├── BreastCancerML.ipynb    # Main notebook with complete analysis
├── README.md               # Project documentation
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-classification.git
   cd breast-cancer-classification
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook BreastCancerML.ipynb
   ```

## Imbalanced Data Handling

This notebook uses **stratified sampling** (`stratify=y`) to preserve class proportions in train/test splits. For more advanced imbalance handling techniques (SMOTE, class weights, threshold tuning), see the extended version.

## Recommendations

- For deployment: **Logistic Regression** or **SVM** for interpretability and high performance
- **Random Forest** provides valuable feature importance insights
- In medical applications: prioritize **Recall for Malignant class** to minimize false negatives (missed cancer cases)

## Medical Relevance

In breast cancer diagnosis:
- **False Negatives** (missing malignant tumors) are more dangerous than false positives
- Models achieve **>97% recall** for the Benign class and **>91% recall** for Malignant
- High ROC-AUC scores (>0.98) indicate excellent discriminative ability

## License

This project is open source and available under the [MIT License](LICENSE).
