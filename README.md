# 🚀 Crowdfunding Campaign Success Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Classification-orange?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> Five classification algorithms, tuned and compared, to predict whether a crowdfunding campaign will succeed (`IsSuccessful` = 1) or fail (0), trained on 100,000 campaigns.

---

## 📌 Project Overview

Knowing which campaigns are likely to succeed helps platforms and creators plan better. This project builds a complete classification workflow: cleaning, encoding, feature selection, scaling, model training, cross-validation, hyperparameter tuning and evaluation.

| | |
|---|---|
| **Dataset** | 100,000 campaigns, 13 columns after dropping the ID columns |
| **Target** | `IsSuccessful`: 50.1% successful, so the classes are balanced |
| **Data quality** | No missing values, no duplicates |
| **Best model** | SVM with linear kernel (`C=10`): 99.96% accuracy |

---

## 📂 Repository Contents

| File | Description |
|---|---|
| `Crowdfunding Campaign Success.ipynb` | Full notebook: preprocessing, five models, tuning, comparison and conclusions |

---

## 🔧 Methodology

1. **Clean:** dropped `CampaignID` and the index column; checked nulls and duplicates (none found).
2. **Encode:** `LabelEncoder` on Category, LaunchMonth, Country, Currency and VideoIncluded.
3. **Split:** 80/20 train/test (80,000 / 20,000 rows, `random_state=42`).
4. **Select features:** `SelectKBest` (ANOVA F-test, k=6), fitted on the training set only, which avoids leakage from the test set. Selected: `GoalAmount`, `RaisedAmount`, `DurationDays`, `OwnerExperience`, `VideoIncluded`, `NumUpdates`.
5. **Scale:** `StandardScaler` for SVM and KNN.
6. **Train and cross-validate:** Random Forest, Decision Tree, KNN, Gaussian Naive Bayes and SVM (5-fold).
7. **Tune:** `GridSearchCV` for every model.

---

## 📈 Results

### Final comparison (tuned models, 20,000 test campaigns)

| Model | Accuracy | Precision | Recall | F1-Score | Best parameters |
|---|---|---|---|---|---|
| **SVM** | **0.9996** | 0.9996 | 0.9996 | 0.9996 | `C=10`, linear |
| Random Forest | 0.9558 | 0.9581 | 0.9558 | 0.9557 | `max_depth=15`, `n_estimators=50` |
| Decision Tree | 0.9297 | 0.9343 | 0.9297 | 0.9295 | gini, `max_depth=9` |
| KNN | 0.8957 | 0.8958 | 0.8957 | 0.8957 | `k=11`, Manhattan, distance-weighted |
| Gaussian Naive Bayes | 0.6622 | 0.6782 | 0.6622 | 0.6549 | `var_smoothing=1e-9` |

### Cross-validation (5-fold, training data)

| Model | Default settings | After tuning |
|---|---|---|
| SVM | 0.9852 (RBF) | **0.9994** (linear, C=10) |
| Decision Tree | 0.9923 | 0.9280 |
| Random Forest | 0.9862 | 0.9642 |
| KNN | 0.8922 | 0.8970 |
| Naive Bayes | 0.6646 | 0.6648 |

### Insights
- 🏆 **A linear SVM wins**, and a linear model beating the RBF kernel (98.70% on the test set) suggests the classes are almost linearly separable in the selected features.
- 📉 **Naive Bayes is weakest (66%)** because it assumes the features are independent.
- 🔁 **Tuning hurt the tree models.** The search grids capped tree depth (Decision Tree at 9, Random Forest at 15), while the untuned models, which grow without a depth limit, scored higher (an unrestricted Decision Tree reached 0.9923 in cross-validation versus 0.9280 for the tuned one). The likely cause is the restricted search space rather than overfitting.

---

## ⚠️ Important Caveat: Possible Target Leakage

`RaisedAmount` is one of the six selected features. It is only known **after** a campaign ends, and success appears to follow directly from comparing it with `GoalAmount`. In all 10 rows visible in the notebook output, `IsSuccessful` is 1 exactly when `RaisedAmount` is at least `GoalAmount`.

This would explain the near-perfect scores (99.96% with a simple linear model, 99.2% for an unrestricted decision tree). If so, the model is rediscovering the definition of success rather than predicting it. For a genuine forecast made at launch, remove `RaisedAmount` (and likely `NumBackers`) and re-run; expect noticeably lower accuracy. Checking `((df.RaisedAmount >= df.GoalAmount) == df.IsSuccessful).mean()` on the full data will confirm this.

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/hasanulazad-cloud/Classification-Algorithms-Model-Building.git
cd Classification-Algorithms-Model-Building

# 2. Install the dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# 3. Open the notebook (an internet connection is needed to load the dataset)
jupyter notebook "Crowdfunding Campaign Success.ipynb"
```

---

## 🛠️ Known Limitations

- **Missing import.** `train_test_split` is used before it is imported, so a fresh top-to-bottom run fails with a `NameError`. Add `from sklearn.model_selection import train_test_split` to the first cell.
- **Confusion-matrix mismatch.** The 17 misclassifications quoted in the notebook belong to the untuned linear SVM (99.91% accuracy). The tuned SVM at 99.96% makes about 9 errors in 20,000.
- **Label-encoded categories.** `LabelEncoder` turns Category, Country and similar columns into ordered numbers, which can mislead distance-based and linear models; one-hot encoding is safer.
- **Test-set comparison.** Final models were compared on the same test set used to report results, with no separate validation set.

---

## 🔮 Future Improvements

- Remove leaking features and rebuild a launch-time prediction model
- One-hot encode categorical variables
- Add ROC curves and feature importance
- Try gradient boosting (XGBoost or LightGBM) and compare

---

## 👤 Author

**Hasanul H Azad**
Medical billing / RCM professional upskilling in Data Science, AI and Machine Learning
GitHub: [@hasanulazad-cloud](https://github.com/hasanulazad-cloud)

---

⭐ If you found this useful, consider starring the repo.
