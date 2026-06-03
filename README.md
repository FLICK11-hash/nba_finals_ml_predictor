# NBA Finals Prediction Project

This project uses machine learning to predict the winner of an NBA Finals matchup using only regular-season team statistics and historical team context.

The dataset contains every NBA Finals participant from **1976 through 2026**, with completed Finals from **1976–2025** used for training and evaluation, and the **2026 Finals** used as the prediction target.

Google Doc with more info: https://docs.google.com/document/d/1L8kt1vHJa6QDTG3yOV-TY1N2zEWm12QQ4AQ5OHoaA90/edit?usp=sharing

## Dataset

Input file:

```text
data/nba_finals_1976_2026.csv
```

Columns:

```text
year
team
champion
ortg
drtg
seed
wins
mvp6
dpoy5
allstars
finals5
titles5
threept_pct
twopt_pct
ft_pct
star_injury_flag
```

Feature descriptions:

* **ortg** – Offensive Rating
* **drtg** – Defensive Rating
* **seed** – Playoff seed entering the postseason
* **wins** – Regular-season wins
* **mvp6** – Team has a player who won MVP within the previous six seasons
* **dpoy5** – Team has a player who won Defensive Player of the Year (or hypothetical equivalent for pre-award seasons) within the previous five seasons
* **allstars** – Number of All-Stars on the roster that season
* **finals5** – Team reached the NBA Finals within the previous five seasons
* **titles5** – Team won an NBA championship within the previous five seasons
* **threept_pct** – Team three-point percentage
* **twopt_pct** – Team two-point percentage
* **ft_pct** – Team free-throw percentage
* **star_injury_flag** – Indicates whether a major star or All-Star was significantly injured entering the Finals

For seasons before the introduction of the three-point line, `threept_pct` is set to `0.000`.

The 2026 champion field remains blank and is used only for prediction.

## Key Modeling Choice

The model does not train on individual team rows.

Instead, each NBA Finals is converted into a single matchup:

```text
Team A features - Team B features
```

The target label is:

```text
1 if Team A won the Finals
0 if Team B won the Finals
```

This approach forces the model to learn the relative strengths of two Finals teams rather than memorizing characteristics of champions.

## Training and Evaluation

Train/Test Split:

```text
1976–2015 = Training Set
2016–2025 = Test Set
2026 = Prediction Only
```

In addition to the standard train/test split, the project performs:

```text
Leave-One-Year-Out Validation (LOYO)
```

For LOYO validation, each Finals is held out once, the model is trained on all remaining Finals, and a prediction is made on the held-out year.

This provides a more robust estimate of model performance on small datasets.

## Models

Current models:

* Logistic Regression
* Linear SVM

Additional models such as Random Forest and RBF SVM were tested during development but ultimately underperformed compared to the simpler linear models.

## Results

Current Leave-One-Year-Out Accuracy (1976–2025):

```text
Logistic Regression: 48 / 50 (96.0%)
Linear SVM:          48 / 50 (96.0%)
```

Heuristic baseline (pick team with more wins):

```text
36 / 50 (72.0%)
```

The only missed Finals were:

```text
2001 Lakers vs 76ers
2016 Warriors vs Cavaliers
```

Both are historically unusual Finals outcomes relative to regular-season team profiles.

## 2026 Prediction

Using regular-season data from both finalists:

```text
San Antonio Spurs
vs
New York Knicks
```

Both Logistic Regression and Linear SVM predict:

```text
San Antonio Spurs
```

to win the 2026 NBA Finals.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Audit the dataset:

```bash
python src/audit_data.py
```

Train models and generate predictions:

```bash
python src/train_models.py
```

## Outputs

Generated files are saved to:

```text
outputs/
```

Important outputs include:

* `model_summary.csv`
* `prediction_2026.csv`
* `matchup_dataset.csv`
* `logistic_regression_coefficients.csv`
* `linear_svm_coefficients.csv`

## Disclaimer

This project is intended as an exploratory machine learning study of NBA Finals outcomes.

Although the models achieve strong historical accuracy, the dataset contains only 50 completed Finals matchups. Results should therefore be interpreted as an interesting statistical analysis rather than a guaranteed prediction system.
