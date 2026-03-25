# EmberWatch
> A machine learning project that predicts wildfire risk levels across 
> 5 regions in British Columbia using histroical climate and fire data. 
> Built as part of CMPT 310 at Simon Fraser University

---

## Project Structure
```
├── data/
│   └── merged_dataset.csv
├── models/
│   └── logistic_regression.pkl
├── results/
│   ├── confusion_matrix_lr.png
│   ├── cv_scores_lr.png
│   ├── feature_coefficients_lr.png
│   └── feature_coefficients_lr.csv
├── preprocessing.py
├── logistic_regression.py
├── predict.py
├── random_forest.py
└── README.md
```
## Setup
```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install pip install scikit-learn pandas numpy matplotlib seaborn joblib openpyxl

# 4. Run Script
python3 logistic_regression.py

# 5. Close Virtual Environment
deactivate
```
---
