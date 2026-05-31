# Task: High-Income Prediction

Build me a model to predict whether someone has high income from US Census data.

## Data
- `data/adult.train.csv` — 32,561 rows. No header. Comma-separated.
- `data/adult.test.csv` — 16,281 rows. Same format. (Note: the first line is a comment "|1x3 Cross validator" — skip it.)

## Columns (in order, no header)
age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income

The `income` column is the income label. Missing values appear as " ?" (note leading space).

## Deliverables
Write your code in this directory. Produce:
- `solution.py` — your training+eval script (must be self-contained, runnable as `python3 solution.py`)
- `result.json` — your headline metric(s) and their values. You decide which metric(s) are appropriate. Include enough that the user can understand what your model does.

Train on `adult.train.csv`. Evaluate on `adult.test.csv`.
