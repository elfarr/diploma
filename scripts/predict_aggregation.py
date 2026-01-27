import pandas as pd

preds = pd.read_csv("reports/tables/outer_test_predictions.csv")

preds_agg = (
    preds
    .groupby(["outer", "row_index"], as_index=False)
    .agg(
        y_true=("y_true", "first"),
        p=("y_pred_proba", "mean")
    )
)

preds_agg.to_csv(
    "reports/tables/outer_test_predictions_agg.csv",
    index=False
)
