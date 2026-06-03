from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.inspection import permutation_importance
from scipy.stats import binomtest


FEATURES = [
    "ortg",
    "drtg",
    "seed",
    "wins",
    "mvp6",
    "dpoy5",
    "allstars",
    "finals5",
    "titles5",
    "threept_pct",
    "twopt_pct",
    "ft_pct",
    "star_injury_flag",
]


HIGHER_IS_BETTER = [
    "ortg_diff",
    "wins_diff",
    "mvp6_diff",
    "dpoy5_diff",
    "allstars_diff",
    "finals5_diff",
    "titles5_diff",
    "threept_pct_diff",
    "twopt_pct_diff",
    "ft_pct_diff"
]

LOWER_IS_BETTER = [
    "drtg_diff",
    "seed_diff",
    "star_injury_flag_diff",
]


def load_team_data(path):
    df = pd.read_csv(path, skip_blank_lines=True)

    required = ["year", "team", "champion"] + FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["year"] = df["year"].astype(int)
    df["champion"] = pd.to_numeric(df["champion"], errors="coerce")

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def make_matchup_data(df):
    rows = []

    for year, group in df.groupby("year", sort=True):
        group = group.reset_index(drop=True)

        if len(group) != 2:
            raise ValueError(f"Expected 2 teams for {year}, found {len(group)}")

        a = group.iloc[0]
        b = group.iloc[1]

        row = {
            "year": int(year),
            "team_a": a["team"],
            "team_b": b["team"],
        }

        for feature in FEATURES:
            if feature == "drtg":
                row[f"{feature}_diff"] = 0.7 * float(a[feature] - b[feature])
            else:
                row[f"{feature}_diff"] = float(a[feature] - b[feature])

        if pd.isna(a["champion"]) or pd.isna(b["champion"]):
            row["team_a_won"] = np.nan
            row["winner"] = ""
        else:
            row["team_a_won"] = int(a["champion"] == 1)
            row["winner"] = a["team"] if row["team_a_won"] == 1 else b["team"]

        rows.append(row)

    return pd.DataFrame(rows)


def get_models():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=42)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42,
        ),
        "linear_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="linear", probability=True, random_state=42)),
        ]),
        "rbf_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=42)),
        ]),
    }


def dominance_override(row):
    # --------------------------------------------------
    # Rule 1:
    # One team is equal or better in EVERY category
    # --------------------------------------------------

    team_a_at_least_as_good = (
        all(row[col] >= 0 for col in HIGHER_IS_BETTER)
        and all(row[col] <= 0 for col in LOWER_IS_BETTER)
    )

    team_b_at_least_as_good = (
        all(row[col] <= 0 for col in HIGHER_IS_BETTER)
        and all(row[col] >= 0 for col in LOWER_IS_BETTER)
    )

    if team_a_at_least_as_good and not team_b_at_least_as_good:
        return 1

    if team_b_at_least_as_good and not team_a_at_least_as_good:
        return 0

    # --------------------------------------------------
    # Rule 2:
    # Core basketball dominance
    # Better wins, seed, offense, and defense
    # --------------------------------------------------

    team_a_core_better = (
    row["wins_diff"] > 0
    and row["seed_diff"] < 0
    and row["ortg_diff"] > 0
    and row["drtg_diff"] < 0
    and row["finals5_diff"] >= 0
    and row["titles5_diff"] >= 0
    )

    team_b_core_better = (
        row["wins_diff"] < 0
        and row["seed_diff"] > 0
        and row["ortg_diff"] < 0
        and row["drtg_diff"] > 0
        and row["finals5_diff"] <= 0
        and row["titles5_diff"] <= 0
    )

    if team_a_core_better:
        return 1

    if team_b_core_better:
        return 0

    # --------------------------------------------------
    # No override
    # --------------------------------------------------
        # --------------------------------------------------
    # Rule 3:
    # Big regular-season gap
    # If one team has 10+ more wins and is not more injured,
    # trust the stronger regular-season team.
    # --------------------------------------------------

    team_a_big_wins_edge = (
        row["wins_diff"] >= 10
        and row["star_injury_flag_diff"] <= 0
    )

    team_b_big_wins_edge = (
        row["wins_diff"] <= -10
        and row["star_injury_flag_diff"] >= 0
    )

    if team_a_big_wins_edge:
        return 1

    if team_b_big_wins_edge:
        return 0
    
    return None


def heuristic_prediction(row):
    if row["wins_diff"] != 0:
        return int(row["wins_diff"] > 0)

    if row["drtg_diff"] != 0:
        return int(row["drtg_diff"] < 0)

    if row["ortg_diff"] != 0:
        return int(row["ortg_diff"] > 0)

    if row["seed_diff"] != 0:
        return int(row["seed_diff"] < 0)

    if row["allstars_diff"] != 0:
        return int(row["allstars_diff"] > 0)

    return 1


def apply_dominance_overrides(test_df, raw_pred):
    final_pred = []

    for i, (_, row) in enumerate(test_df.iterrows()):
        override = dominance_override(row)

        if override is None:
            final_pred.append(int(raw_pred[i]))
        else:
            final_pred.append(int(override))

    return np.array(final_pred)


def train_final_models(matchups, output_dir):
    feature_cols = [f"{feature}_diff" for feature in FEATURES]

    train = matchups[(matchups["year"] >= 1980) & (matchups["year"] <= 2015)].copy()
    test = matchups[(matchups["year"] >= 2016) & (matchups["year"] <= 2025)].copy()
    predict = matchups[matchups["year"] == 2026].copy()

    X_train = train[feature_cols]
    y_train = train["team_a_won"].astype(int)

    X_test = test[feature_cols]
    y_test = test["team_a_won"].astype(int)

    summary_rows = []
    prediction_rows = []

    for name, model in get_models().items():
        model.fit(X_train, y_train)

        raw_pred = model.predict(X_test)
        pred = apply_dominance_overrides(test, raw_pred)
        prob = model.predict_proba(X_test)[:, 1]

        summary_rows.append({
            "model": name,
            "train_matchups": len(train),
            "test_matchups": len(test),
            "test_accuracy": round(accuracy_score(y_test, pred), 3),
            "test_log_loss": round(log_loss(y_test, prob), 3),
        })

        test_predictions = test[["year", "team_a", "team_b", "winner"]].copy()
        test_predictions["model"] = name
        test_predictions["actual_team_a_won"] = y_test.values
        test_predictions["raw_predicted_team_a_won"] = raw_pred
        test_predictions["final_predicted_team_a_won"] = pred
        test_predictions["prob_team_a_wins"] = prob
        test_predictions["dominance_override_used"] = [
            dominance_override(row) is not None
            for _, row in test.iterrows()
        ]
        test_predictions["predicted_winner"] = np.where(
            pred == 1,
            test_predictions["team_a"],
            test_predictions["team_b"],
        )

        test_predictions.to_csv(
            output_dir / f"{name}_test_predictions.csv",
            index=False,
        )

        if not predict.empty:
            X_predict = predict[feature_cols]
            raw_pred_2026 = model.predict(X_predict)
            pred_2026 = apply_dominance_overrides(predict, raw_pred_2026)
            prob_2026 = model.predict_proba(X_predict)[:, 1]

            for i, (_, row) in enumerate(predict.iterrows()):
                prediction_rows.append({
                    "model": name,
                    "year": int(row["year"]),
                    "team_a": row["team_a"],
                    "team_b": row["team_b"],
                    "prob_team_a_wins": round(float(prob_2026[i]), 4),
                    "prob_team_b_wins": round(float(1 - prob_2026[i]), 4),
                    "raw_predicted_winner": row["team_a"] if raw_pred_2026[i] == 1 else row["team_b"],
                    "final_predicted_winner": row["team_a"] if pred_2026[i] == 1 else row["team_b"],
                    "dominance_override_used": dominance_override(row) is not None,
                })

        if name == "logistic_regression":
            coef = model.named_steps["model"].coef_[0]

            pd.DataFrame({
                "feature": feature_cols,
                "coefficient": coef,
                "abs_coefficient": np.abs(coef),
            }).sort_values("abs_coefficient", ascending=False).to_csv(
                output_dir / "logistic_regression_coefficients.csv",
                index=False,
            )

        elif name == "linear_svm":
            coef = model.named_steps["model"].coef_[0]

            pd.DataFrame({
                "feature": feature_cols,
                "coefficient": coef,
                "abs_coefficient": np.abs(coef),
            }).sort_values("abs_coefficient", ascending=False).to_csv(
                output_dir / "linear_svm_coefficients.csv",
                index=False,
            )

        elif name == "random_forest":
            pd.DataFrame({
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).to_csv(
                output_dir / "random_forest_feature_importance.csv",
                index=False,
            )

        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=100,
            random_state=42,
            scoring="accuracy",
        )

        pd.DataFrame({
            "feature": feature_cols,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }).sort_values("permutation_importance_mean", ascending=False).to_csv(
            output_dir / f"{name}_permutation_importance.csv",
            index=False,
        )

    summary = pd.DataFrame(summary_rows)
    predictions = pd.DataFrame(prediction_rows)

    summary.to_csv(output_dir / "model_summary.csv", index=False)
    predictions.to_csv(output_dir / "prediction_2026.csv", index=False)

    print("\nModel summary:")
    print(summary.to_string(index=False))

    print("\n2026 predictions:")
    print(predictions.to_string(index=False))


def run_full_leave_one_year_out(matchups, output_dir):
    feature_cols = [f"{feature}_diff" for feature in FEATURES]

    completed = matchups[
        (matchups["year"] >= 1980) &
        (matchups["year"] <= 2025)
    ].copy()

    rows = []

    for test_year in sorted(completed["year"].unique()):
        train = completed[completed["year"] != test_year].copy()
        test = completed[completed["year"] == test_year].copy()

        X_train = train[feature_cols]
        y_train = train["team_a_won"].astype(int)

        X_test = test[feature_cols]
        y_test = test["team_a_won"].astype(int)

        for name, model in get_models().items():
            model.fit(X_train, y_train)

            raw_pred = model.predict(X_test)
            pred = apply_dominance_overrides(test, raw_pred)
            prob = model.predict_proba(X_test)[:, 1]

            row = test[["year", "team_a", "team_b", "winner"]].iloc[0].to_dict()
            row["model"] = name
            row["actual_team_a_won"] = int(y_test.iloc[0])
            row["raw_predicted_team_a_won"] = int(raw_pred[0])
            row["final_predicted_team_a_won"] = int(pred[0])
            row["prob_team_a_wins"] = float(prob[0])
            row["dominance_override_used"] = dominance_override(test.iloc[0]) is not None
            row["predicted_winner"] = row["team_a"] if pred[0] == 1 else row["team_b"]
            row["correct"] = int(row["actual_team_a_won"] == row["final_predicted_team_a_won"])

            rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "leave_one_year_out_all_predictions.csv", index=False)

    wrong = results[results["correct"] == 0].copy()
    wrong.to_csv(output_dir / "leave_one_year_out_wrong_predictions.csv", index=False)

    summary = (
        results
        .groupby("model")["correct"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(columns={
            "sum": "correct",
            "count": "total",
            "mean": "accuracy",
        })
    )

    summary["accuracy"] = summary["accuracy"].round(3)
    summary.to_csv(output_dir / "leave_one_year_out_summary.csv", index=False)

    feature_cols = [f"{feature}_diff" for feature in FEATURES]

    print("\nLogistic Regression Miss Feature Diffs:")

    for year in [1980, 1982, 1995, 2001, 2014, 2016]:
        row = matchups[matchups["year"] == year].iloc[0]

        print(f"\n===== {year} =====")

        for col in feature_cols:
            print(f"{col}: {row[col]}")


    log_row = summary[summary["model"] == "logistic_regression"].iloc[0]

    print("\nSimple Logistic Regression Result:")
    print(
        f"Logistic Regression correctly predicted "
        f"{int(log_row['correct'])}/{int(log_row['total'])} Finals "
        f"({log_row['accuracy']:.3f}) from 1980-2025."
    )

    p_value = binomtest(
        k=int(log_row["correct"]),
        n=int(log_row["total"]),
        p=0.5,
        alternative="greater",
    ).pvalue

    print(f"Binomial test p-value vs random guessing: {p_value:.6g}")

    print("\nLeave-one-year-out summary, 1980-2025:")
    print(summary.to_string(index=False))

    print("\nWrong leave-one-year-out predictions:")
    if wrong.empty:
        print("None")
    else:
        print(wrong[[
            "year",
            "team_a",
            "team_b",
            "winner",
            "model",
            "predicted_winner",
            "prob_team_a_wins",
            "dominance_override_used",
        ]].to_string(index=False))

    log_wrong = wrong[wrong["model"] == "logistic_regression"]

    print("\nLogistic Regression Misses:")
    if log_wrong.empty:
        print("None")
    else:
        print(log_wrong[[
            "year",
            "team_a",
            "team_b",
            "winner",
            "predicted_winner",
            "prob_team_a_wins",
            "dominance_override_used",
        ]].to_string(index=False))

    heuristic_preds = completed.apply(heuristic_prediction, axis=1)
    heuristic_correct = int((heuristic_preds == completed["team_a_won"]).sum())
    heuristic_total = len(completed)
    heuristic_accuracy = heuristic_correct / heuristic_total

    print("\nHeuristic Baseline:")
    print(f"{heuristic_correct}/{heuristic_total} ({heuristic_accuracy:.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/nba_finals_1980_2026.csv")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_team_data(args.data)
    matchups = make_matchup_data(df)

    matchups.to_csv(output_dir / "matchup_dataset.csv", index=False)

    train_final_models(matchups, output_dir)
    run_full_leave_one_year_out(matchups, output_dir)

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()