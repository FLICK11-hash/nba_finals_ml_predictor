(.venv) PS C:\Users\conra\nba_finals_ml_predictor\nba_finals_ml_project> python src\train_models.py

Model summary:
              model  train_matchups  test_matchups  test_accuracy  test_log_loss
logistic_regression              40             10            0.8          0.469
         linear_svm              40             10            0.9          0.556

2026 predictions:
              model  year team_a team_b  prob_team_a_wins  prob_team_b_wins raw_predicted_winner final_predicted_winner  dominance_override_used
logistic_regression  2026  Spurs Knicks            0.9871            0.0129                Spurs                  Spurs                    False
         linear_svm  2026  Spurs Knicks            0.9910            0.0090                Spurs                  Spurs                    False

Simple Logistic Regression Result:
Logistic Regression correctly predicted 48/50 Finals (0.960).
Binomial test p-value vs random guessing: 1.13332e-12

Leave-one-year-out summary from all Finals:
              model  correct  total  accuracy
         linear_svm       48     50      0.96
logistic_regression       48     50      0.96

Wrong leave-one-year-out predictions:
 year   team_a    team_b    winner               model predicted_winner  prob_team_a_wins  dominance_override_used
 2001   Lakers     76ers    Lakers logistic_regression            76ers          0.059393                    False
 2001   Lakers     76ers    Lakers          linear_svm            76ers          0.036121                    False
 2016 Warriors Cavaliers Cavaliers logistic_regression         Warriors          0.959243                     True
 2016 Warriors Cavaliers Cavaliers          linear_svm         Warriors          0.954723                     True

Heuristic Baseline:
36/50 (0.720)

Saved outputs to: outputs