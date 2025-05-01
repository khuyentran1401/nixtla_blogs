# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "nixtla==0.6.6",
#     "numpy==2.2.5",
#     "openai==1.76.2",
#     "pandas==2.2.3",
#     "utilsforecast==0.2.12",
# ]
# ///

import marimo

__generated_with = "0.13.1"
app = marimo.App(app_title="Demand Forecasting")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from nixtla import NixtlaClient
    from utilsforecast.evaluation import evaluate
    from utilsforecast.losses import mae
    return NixtlaClient, evaluate, mae, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Prerequisites""")
    return


@app.cell
def _(NixtlaClient):
    import os

    NIXTLA_API_KEY = os.environ["NIXTLA_API_KEY"]
    client = NixtlaClient(api_key=NIXTLA_API_KEY)
    return (client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data Preparation""")
    return


@app.cell
def _(pd):
    sales_data = pd.read_csv("https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/m5_sales_exog_small.csv")
    sales_data["ds"] = pd.to_datetime(sales_data["ds"])
    sales_data.head()
    return (sales_data,)


@app.cell
def _(client, sales_data):
    client.plot(
        sales_data,
        max_insample_length=365,

    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Bounded Forecasts""")
    return


@app.cell
def _(np, sales_data):
    log_transformed_data = sales_data.copy()
    log_transformed_data["y"] = np.log(log_transformed_data["y"] + 1)
    log_transformed_data.head()
    return (log_transformed_data,)


@app.cell
def _(client, log_transformed_data, sales_data):
    import matplotlib.pyplot as plt

    # Create a single figure and axes
    plot_fig, plot_ax = plt.subplots(figsize=(10, 5))

    # Plot the original data
    client.plot(
        sales_data,
        max_insample_length=30,
        unique_ids=["FOODS_1_001"],
        engine="matplotlib",
        ax=plot_ax,
    )

    # Plot the transformed data on the same axes
    client.plot(
        log_transformed_data,
        max_insample_length=30,
        unique_ids=["FOODS_1_001"],
        engine="matplotlib",
        ax=plot_ax,
    )

    # Retrieve all line objects from the axes
    plot_lines = plot_ax.get_lines()

    # Assign colors to each line
    line_colors = ["blue", "red"]
    for line, color in zip(plot_lines, line_colors, strict=False):
        line.set_color(color)

    # Set the y-axis limit
    plot_ax.set_ylim(top=6)

    # Add a legend to distinguish the plots
    plot_ax.legend(["Original Data", "Transformed Data"])

    # Display the plot
    plot_fig
    return


@app.cell
def _(log_transformed_data):
    test_data = log_transformed_data.groupby("unique_id").tail(28)
    train_data = log_transformed_data.drop(test_data.index).reset_index(drop=True)
    return test_data, train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Generating Forecasts with TimeGPT""")
    return


@app.cell
def _(client, train_data):
    raw_forecast = client.forecast(
        df=train_data,
        h=28,
        level=[80],
        model="timegpt-1-long-horizon",
        time_col="ds",
        target_col="y",
        id_col="unique_id",
    )
    return (raw_forecast,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Reverse Transformation

        After obtaining predictions, we reverse the log transformation to return to the original scale.
        """
    )
    return


@app.cell
def _(np, raw_forecast):
    def reverse_log_transform(df):
        value_cols = [col for col in df if col not in ["ds", "unique_id"]]
        df = df.copy()
        df[value_cols] = np.exp(df[value_cols]) - 1
        return df

    untransformed_forecast = reverse_log_transform(raw_forecast)
    untransformed_forecast.head()
    return reverse_log_transform, untransformed_forecast


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Evaluation""")
    return


@app.cell
def _(client, test_data, untransformed_forecast):
    client.plot(test_data, untransformed_forecast, models=["TimeGPT"], level=[80], time_col="ds", target_col="y")
    return


app._unparsable_cell(
    r"""
    mae(df=)
    """,
    name="_"
)


@app.cell
def _(evaluate, mae, pd, test_data, untransformed_forecast):
    untransformed_forecast["ds"] = pd.to_datetime(untransformed_forecast["ds"])
    merged_results = pd.merge(test_data, untransformed_forecast, "left", ["unique_id", "ds"])

    model_evaluation = evaluate(merged_results, metrics=[mae], models=["TimeGPT"], target_col="y", id_col="unique_id")
    model_metrics = model_evaluation.groupby("metric")["TimeGPT"].mean()
    model_metrics
    return (merged_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Finetuning the Model""")
    return


@app.cell
def _(client, train_data):
    raw_finetuned = client.forecast(df=train_data, h=28, level=[80], finetune_steps=10, finetune_loss="mae", model="timegpt-1-long-horizon", time_col="ds", target_col="y", id_col="unique_id")
    return (raw_finetuned,)


@app.cell
def _(raw_finetuned, reverse_log_transform):
    renamed_finetuned = raw_finetuned.rename(columns={"TimeGPT": "TimeGPT_finetuned"})
    untransformed_finetuned = reverse_log_transform(renamed_finetuned)
    untransformed_finetuned.head()
    return (untransformed_finetuned,)


@app.cell
def _(evaluate, mae, merged_results, untransformed_finetuned):
    merged_results["TimeGPT_finetuned"] = untransformed_finetuned["TimeGPT_finetuned"].values
    final_evaluation = evaluate(merged_results, metrics=[mae], models=["TimeGPT", "TimeGPT_finetuned"], target_col="y", id_col="unique_id")
    final_metrics = final_evaluation.groupby("metric")[["TimeGPT", "TimeGPT_finetuned"]].mean()
    final_metrics
    return


if __name__ == "__main__":
    app.run()
