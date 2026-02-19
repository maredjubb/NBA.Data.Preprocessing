# NBA Player Salary — Data Preprocessing

A data preprocessing project based on the [Hyperskill NBA Data Preprocessing](https://hyperskill.org/projects/285) project. Transforms raw NBA player data into a clean, model-ready dataset for predicting player salaries.

## Overview

Data quality is one of the biggest factors in machine learning model performance. This project applies a full preprocessing pipeline to NBA player data — handling missing values, engineering new features, removing multicollinear variables, and scaling/encoding for use in a linear model.

## Dataset

NBA2K player data loaded from a hosted CSV file. Contains player attributes including height, weight, salary, draft information, country, and performance ratings.

## Pipeline

1. **Clean** — parse dates, handle missing values, strip extraneous characters, standardise units, and cast features to correct types
2. **Feature Engineering** — derive `age`, `experience`, and `bmi` from existing columns, drop high cardinality features
3. **Multicollinearity** — build a correlation matrix to identify and drop redundant features (`age` removed in favour of `experience`)
4. **Transform** — scale numerical features with `StandardScaler`, encode categorical features with `OneHotEncoder`

## Requirements

See [requirements.txt](requirements.txt)

## Usage
The script outputs the correlation matrix at the multicollinearity stage and returns `X` (features) and `y` (salary) ready for model training.

## Skills Demonstrated

- Data cleaning and parsing with Pandas
- Feature engineering from datetime and numeric columns
- Multicollinearity detection using correlation matrices
- Numerical scaling and categorical encoding with scikit-learn

---

Made by Mared Jubb (https://hyperskill.org/profile/536183895)
