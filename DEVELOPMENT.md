# Local Development & Evaluation Guide

This guide is for anyone reviewing the project locally. The goal is simple: clone the repository, create an isolated Python environment, install the pinned dependencies, run the tests, train/load the model, and start the API.

The commands below are intentionally kept close to the way the project is structured. There is no hidden setup step and no dependency on a developer-specific path.

## 1. Prerequisites

Install:

- Git
- Python 3.11 (recommended; the GitHub Actions test workflow uses Python 3.11)
- A terminal such as Command Prompt, PowerShell, or a Unix shell

Check the installations:

```bash
git --version
python --version
```

If `python` does not resolve on Windows, try:

```bash
py --version
```

## 2. Get the project

Clone the repository and move into the project directory:

```bash
git clone https://github.com/aranimondal/customer-churn-prediction.git
cd customer-churn-prediction
```

Make sure you are on the main branch:

```bash
git checkout main
git pull origin main
```

At this point the evaluator should be at the repository root — the directory containing `app.py`, `requirements.txt`, `src/`, `tests/`, `data/` and `notebook/`.

## 3. Create a virtual environment

Creating a virtual environment avoids mixing this project's package versions with packages installed for other Python projects.

### Windows — Command Prompt

```bat
python -m venv .venv
.venv\\Scripts\\activate
```

### Windows — PowerShell

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, the terminal should show `.venv` in the prompt.

## 4. Install the exact project dependencies

From the project root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository pins the main runtime, ML and test dependencies in `requirements.txt`. Using that file is preferable to installing packages individually because it keeps the evaluator's environment aligned with the project.

## 5. Run the automated test suite

Before starting the service, run:

```bash
python -m pytest -q
```

The tests cover the data-cleaning contract, feature engineering, model pipeline, persisted artifact behaviour, API validation and API prediction paths, along with the additional model-comparison and threshold-analysis checks.

If the test suite fails, fix the environment or test failure before treating the API result as a valid evaluation.

## 6. Train the model locally

The API loads the persisted model from `model/churn_model.pkl`. To reproduce the training process from the supplied dataset, run:

```bash
python -m src.train
```

The training script:

1. Loads the supplied Telco dataset.
2. Applies the documented cleaning rules.
3. Creates the train/test split with the project's fixed random state and stratification.
4. Compares the Decision Tree configurations using cross-validation on the training split.
5. Evaluates the selected model on the held-out test set.
6. Prints the evaluation metrics, confusion matrix and feature importances.
7. Writes the trained pipeline to `model/churn_model.pkl`.
8. Writes the comparison and evaluation results to `model/metrics.json`.

If the repository already contains a trained artifact, this step can still be run to reproduce it locally. The important point is that the API consumes the saved pipeline rather than rebuilding preprocessing separately at inference time.

## 7. Start the application in development mode

From the project root, with the virtual environment still active:

```bash
uvicorn app:app --reload
```

You should see a message similar to:

```text
Uvicorn running on http://127.0.0.1:8000
Application startup complete.
```

Keep this terminal open while testing the API.

The `--reload` option is intentional for local development. It watches the source files and restarts the development server when application code changes. It is not a production deployment setting.

## 8. Check the application through the browser

Open:

**Swagger UI:** http://127.0.0.1:8000/docs

This is the easiest way for an evaluator to interact with the application without installing Postman or writing a separate client.

### Health check

1. Open `GET /health`.
2. Click **Try it out**.
3. Click **Execute**.
4. Confirm that the service reports the model as available.

### Make a prediction

1. Open `POST /predict`.
2. Click **Try it out**.
3. Paste the request from `sample_request.json`.
4. Click **Execute**.
5. Inspect the returned `prediction` and `churn_probability`.

The API performs request validation and sends the customer data through the same persisted preprocessing/model pipeline used during training.

## 9. Test the API from the command line

The supplied sample request can also be sent without Swagger:

### Windows PowerShell

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -InFile sample_request.json
```

### macOS / Linux / Git Bash

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

The response has this shape:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.82
}
```

The numbers above are only an example of the response format. The actual values come from the trained artifact.

## 10. Useful local URLs

Once the server is running:

| Resource | URL |
|---|---|
| Swagger UI | http://127.0.0.1:8000/docs |
| ReDoc | http://127.0.0.1:8000/redoc |
| OpenAPI specification | http://127.0.0.1:8000/openapi.json |
| Health check | http://127.0.0.1:8000/health |
| Prediction endpoint | http://127.0.0.1:8000/predict |

These URLs are local to the evaluator's machine; they will not work until the FastAPI service has been started.

## 11. Review the analysis notebook

The model-development work is documented in:

`notebook/churn_analysis.ipynb`

Open it with Jupyter if you want to inspect the analysis interactively:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

The notebook contains the EDA, feature-engineering analysis, model comparison, evaluation and interpretation. The API is the separate inference layer that consumes the persisted model.

## 12. A simple evaluator checklist

If you only have a few minutes, this is the shortest meaningful verification path:

```text
1. git clone ...
2. cd customer-churn-prediction
3. create + activate .venv
4. python -m pip install -r requirements.txt
5. python -m pytest -q
6. python -m src.train
7. uvicorn app:app --reload
8. Open http://127.0.0.1:8000/docs
9. Execute GET /health
10. Execute POST /predict using sample_request.json
```

That sequence checks the project from installation through automated tests, model training, application startup and live inference.

## 13. Common local issues

### `ModuleNotFoundError: No module named 'src'`

Run commands from the repository root — the directory containing `app.py` and the `src/` folder — and use:

```bash
python -m pytest -q
python -m src.train
```

Do not run the Python modules from inside the `src` directory.

### Port 8000 is already in use

Start Uvicorn on another local port:

```bash
uvicorn app:app --reload --port 8001
```

Then use `http://127.0.0.1:8001/docs`.

### API reports that the model artifact is missing

Run:

```bash
python -m src.train
```

Then restart Uvicorn.

### PowerShell does not allow virtual-environment activation

If PowerShell blocks the activation script, either use Command Prompt or adjust the local PowerShell execution policy according to your organization's security policy. The project itself does not require a permanent system-wide policy change.

## 14. What the evaluator should be able to verify

A successful local run should make the following independently observable:

- The repository installs from `requirements.txt`.
- The automated test suite runs successfully.
- The model can be trained from the supplied data.
- The trained pipeline is persisted as a single artifact.
- The FastAPI application starts locally.
- Swagger exposes the API contract.
- `/health` confirms service/model availability.
- `/predict` validates input and returns a churn prediction with probability.
- The notebook provides the underlying analysis and model-development trail.

No external service, secret, API key or developer-specific absolute file path is required for the local development flow.
