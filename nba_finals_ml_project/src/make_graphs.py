import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score


DATA_PATH = "../data/finals_metrics.csv"
OUTPUT_DIR = "../data"


FEATURES = [
    "ortg",
    "drtg",
    "seed",
    "wins",
    "mvp5",
    "dpoy5",
    "allstars",
    "finals5",
    "titles5",
    "threept_pct",
    "twopt_pct",
    "ft_pct",
]


def build_matchups(df):
    rows = []

    for year, group in df.groupby("year"):
        if len(group) != 2:
            continue

        if group["champion"].isna().any():
            continue

        team_a = group.iloc[0]
        team_b = group.iloc[1]

        row = {
            "year": year,
            "team_a": team_a["team"],
            "team_b": team_b["team"],
            "winner": team_a["team"] if team_a["champion"] == 1 else team_b["team"],
            "label": int(team_a["champion"]),
        }

        for feature in FEATURES:
            row[f"{feature}_diff"] = team_a[feature] - team_b[feature]

        rows.append(row)

    return pd.DataFrame(rows)


def save_model_comparison_graph():
    models = ["Heuristic Baseline", "Logistic Regression", "Linear SVM"]
    accuracies = [0.720, 0.960, 0.960]

    plt.figure(figsize=(8, 5))
    plt.bar(models, accuracies)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    for i, value in enumerate(accuracies):
        plt.text(i, value + 0.02, f"{value:.1%}", ha="center")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_accuracy_comparison.png"))
    plt.close()


def save_logistic_coefficients_graph(matchups):
    feature_cols = [col for col in matchups.columns if col.endswith("_diff")]

    X = matchups[feature_cols]
    y = matchups["label"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=1000))
    ])

    model.fit(X, y)

    coefs = model.named_steps["logistic"].coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coefs,
        "abs_coefficient": abs(coefs),
    }).sort_values("abs_coefficient", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(coef_df["feature"], coef_df["coefficient"])
    plt.title("Logistic Regression Coefficients")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "logistic_regression_coefficients.png"))
    plt.close()

    coef_df.sort_values("abs_coefficient", ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, "logistic_regression_coefficients_graph_data.csv"),
        index=False
    )


def save_correct_vs_wrong_graph(matchups):
    feature_cols = [col for col in matchups.columns if col.endswith("_diff")]

    X = matchups[feature_cols]
    y = matchups["label"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", probability=True))
    ])

    loo = LeaveOneOut()

    results = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        row = matchups.iloc[test_idx[0]]

        results.append({
            "year": row["year"],
            "correct": int(pred == y_test.iloc[0]),
        })

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(12, 4))
    plt.scatter(results_df["year"], results_df["correct"])
    plt.title("Linear SVM Leave-One-Year-Out Results")
    plt.xlabel("Finals Year")
    plt.ylabel("Correct Prediction")
    plt.yticks([0, 1], ["Wrong", "Correct"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "linear_svm_correct_by_year.png"))
    plt.close()

    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "linear_svm_correct_by_year.csv"),
        index=False
    )


def save_2026_prediction_graph():
    models = ["Logistic Regression", "Linear SVM"]
    spurs_probs = [0.9871, 0.9910]

    plt.figure(figsize=(8, 5))
    plt.bar(models, spurs_probs)
    plt.title("2026 Finals Prediction: Spurs Win Probability")
    plt.ylabel("Predicted Probability")
    plt.ylim(0, 1)

    for i, value in enumerate(spurs_probs):
        plt.text(i, value - 0.07, f"{value:.1%}", ha="center")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2026_spurs_prediction_probability.png"))
    plt.close()


def save_feature_distribution_graph(df):
    completed = df[df["champion"].notna()].copy()

    champs = completed[completed["champion"] == 1]
    losers = completed[completed["champion"] == 0]

    plt.figure(figsize=(8, 5))
    plt.hist(champs["wins"], alpha=0.6, label="Champions")
    plt.hist(losers["wins"], alpha=0.6, label="Finals Losers")
    plt.title("Regular Season Wins: Champions vs Finals Losers")
    plt.xlabel("Regular Season Wins")
    plt.ylabel("Number of Teams")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "wins_distribution_champs_vs_losers.png"))
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(how="all")

    matchups = build_matchups(df)

    save_model_comparison_graph()
    save_logistic_coefficients_graph(matchups)
    save_correct_vs_wrong_graph(matchups)
    save_2026_prediction_graph()
    save_feature_distribution_graph(df)

    print("Saved graphs to data/:")
    print("- model_accuracy_comparison.png")
    print("- logistic_regression_coefficients.png")
    print("- linear_svm_correct_by_year.png")
    print("- 2026_spurs_prediction_probability.png")
    print("- wins_distribution_champs_vs_losers.png")


if __name__ == "__main__":
    main()