# EmberWatch
> A machine learning project that predicts wildfire risk levels across 
> 5 regions in British Columbia using histroical climate and fire data. 
> Built as part of CMPT 310 at Simon Fraser University

---

## Project Structure
```
firewatchbc/
├── data/
│   └── merged_dataset.csv
├── models/
│   └── logistic_regression.pkl
├── results/
│   ├── confusion_matrix_lr.png
│   ├── cv_scores_lr.png
│   ├── feature_coefficients_lr.png
│   └── f1_per_class_lr.png
├── preprocessing.py
├── logistic_regression.py
├── predict.py
└── README.md
```