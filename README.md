# Nixtla Blog Code

This repository contains the code and examples for blog articles published on [nixtla.io](https://nixtla.io).

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python package management. To set up the environment:

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:

```bash
uv sync
pre-commit install
```

## Project Structure

- `notebooks/`: Contains marimo notebooks for each blog post
- `README.md`: This file

Each blog post's code is organized in its own notebook, named according to the blog post title.

## Available Notebooks

- [Anomaly Detection](https://khuyentran1401.github.io/nixtla_blogs/anomaly_detection.html) - Learn how to detect anomalies in time series data using TimeGPT
- [Intermittent Forecasting](https://khuyentran1401.github.io/nixtla_blogs/intermittent_forecasting.html) - Explore demand forecasting techniques for intermittent time series

## Running Notebooks Locally

[Marimo](https://marimo.io) is a Python notebook environment that combines the interactivity of Jupyter with the power of modern Python. It provides a clean, distraction-free interface for data analysis and visualization.

To run the notebooks locally, you can use marimo's sandbox mode:

```bash
marimo edit --sandbox notebook.py
```

This will start a local server where you can interact with the notebook in your browser.

## Contributing

We welcome contributions to improve the code examples and documentation. Please see [CONTRIBUTION.md](CONTRIBUTION.md) for detailed guidelines on:

- Style and structure of blog posts
- Development workflow