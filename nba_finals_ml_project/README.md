Skip to content
FLICK11-hash
nba_finals_ml_predictor
Repository navigation
Code
Issues
Pull requests
Actions
Projects
Wiki
Security and quality
Insights
Settings
Files
Go to file
t
T
data
finals_metrics.csv
nba_finals_ml_project
README.md
nba_finals_ml_project.zip
nba_finals_ml_predictor
/
README.md
in
main

Edit

Preview
Indent mode

Spaces
Indent size

2
Line wrap mode

Soft wrap
Editing README.md file contents
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
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
Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.
No file chosen
Attach files by dragging & dropping, selecting or pasting them.
