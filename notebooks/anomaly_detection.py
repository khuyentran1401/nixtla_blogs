# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "pandas==2.2.3",
#     "python-dotenv==1.1.0",
# ]
# ///

import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Anomaly detection""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Import packages""")
    return


@app.cell
def _():
    import os

    import pandas as pd
    from nixtla import NixtlaClient
    return NixtlaClient, os, pd


@app.cell
def _(NixtlaClient, os):
    NIXTLA_API_KEY = os.environ["NIXTLA_API_KEY"]
    nixtla_client = NixtlaClient(api_key=NIXTLA_API_KEY)
    return (nixtla_client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load dataset

    Now, let's load the dataset for this tutorial.
    """
    )
    return


@app.cell
def _(pd):
    # Read the dataset
    wikipedia = pd.read_csv("https://datasets-nixtla.s3.amazonaws.com/peyton-manning.csv", parse_dates=["ds"])
    wikipedia.head(10)
    return (wikipedia,)


@app.cell
def _(nixtla_client, wikipedia):
    nixtla_client.plot(wikipedia)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Anomaly detection""")
    return


@app.cell
def _(nixtla_client, wikipedia):
    anomalies_df = nixtla_client.detect_anomalies(
        wikipedia,
        freq="D",
        model="timegpt-1",
    )
    anomalies_df.head()
    return (anomalies_df,)


@app.cell
def _(anomalies_df, nixtla_client, wikipedia):
    nixtla_client.plot(wikipedia, anomalies_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Anomaly detection with exogenous features""")
    return


@app.cell
def _(nixtla_client, wikipedia):
    anomalies_df_exogenous = nixtla_client.detect_anomalies(
        wikipedia,
        freq="D",
        date_features=["month", "year"],
        date_features_to_one_hot=True,
        model="timegpt-1",
    )
    return (anomalies_df_exogenous,)


@app.cell
def _(nixtla_client):
    nixtla_client.weights_x.plot.barh(
        x="features",
        y="weights"
    )
    return


@app.cell
def _(anomalies_df, anomalies_df_exogenous):
    # Without exogenous features
    print("Number of anomalies without exogenous features:", anomalies_df.anomaly.sum())

    # With exogenous features
    print("Number of anomalies with exogenous features:", anomalies_df_exogenous.anomaly.sum())
    return


@app.cell
def _(anomalies_df_exogenous, nixtla_client, wikipedia):
    nixtla_client.plot(wikipedia, anomalies_df_exogenous)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Modifying the confidence intervals""")
    return


@app.cell
def _(nixtla_client, wikipedia):
    anomalies_df_70 = nixtla_client.detect_anomalies(wikipedia, freq="D", level=70)
    return (anomalies_df_70,)


@app.cell
def _(anomalies_df, anomalies_df_70):
    # Print and compare anomaly counts
    print("Number of anomalies with 99% confidence interval:", anomalies_df.anomaly.sum())
    print("Number of anomalies with 70% confidence interval:", anomalies_df_70.anomaly.sum())
    return


@app.cell
def _(anomalies_df_70, nixtla_client, wikipedia):
    nixtla_client.plot(wikipedia, anomalies_df_70)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
