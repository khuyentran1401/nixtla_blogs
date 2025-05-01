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
```

## Project Structure

- `notebooks/`: Contains marimo notebooks for each blog post
- `README.md`: This file

Each blog post's code is organized in its own notebook, named according to the blog post title.

## Running Notebooks Locally

To run the notebooks locally, you can use marimo's sandbox mode:

```bash
marimo edit --sandbox notebook.py
```

This will start a local server where you can interact with the notebook in your browser.

## Contributing

We welcome contributions to improve the code examples and documentation. Here's how to contribute:

1. Fork the repository
2. Create a new branch for your changes:
```bash
git checkout -b your-feature-name
```
3. Set up your development environment:
```bash
uv sync --all-extras
```
4. Edit notebooks using marimo:
```bash
marimo edit notebook.py --sandbox
```
5. Make your changes and ensure they work as expected
6. Commit your changes with a descriptive message
7. Push your branch to your fork
8. Open a pull request to the main repository
