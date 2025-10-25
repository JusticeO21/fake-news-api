# Fake News Detection API

A FastAPI-based application for detecting fake and factual news. This API uses machine learning models trained on labeled news datasets to classify news articles in real-time.

The backend supports multiple models (`kbap` and `quab`) with flexible vectorization methods for accurate classification.

## Table of Contents

- Features
- Folder Structure
- Installation
- Usage
- Model Selection
- API Endpoints
- License

## Features

- Predict whether a news article is Factual or Fake.
- Supports Logistic Regression (`kbap`) and Naive Bayes (`quab`) models.
- Flexible text preprocessing including:
  - Lowercasing
  - Punctuation removal
  - Tokenization & lemmatization
  - Stop-word removal
- Vectorization via CountVectorizer or TF-IDF.
- Returns prediction and confidence score for each request.

## Folder Structure

```
fake_news_app/
│
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app
│   ├── routes/           # Modular API route files
│   │   └── predict.py
│   ├── models/           # Pydantic models
│   │   └── __init__.py
│   └── utils/            # Utility functions for preprocessing
│       └── preprocessing.py
│
├── notebooks/
│   ├── model_training.ipynb  # Jupyter Notebook for model training
    └── data_analysis.ipynb 
│
├── saved_models/
│   ├── kbap.pkl          # Naive Bayes model
│   └── quab.pkl          # Logistic Regression model
│
├── data/
│   ├── fake_news.cvs          # Data for training the model
│   
├── Dockerfile
├── requirements.txt
└── README.md
```

## Installation

### Option 1: Using Docker (Recommended)

Run the API directly using the pre-built Docker image without cloning the repository:

```bash
docker pull kwabenaowusu/fake-news-detector-api:v1.0
docker run -d -p 8000:8000 kwabenaowusu/fake-news-detector-api:v1.0
```

The API will be available at `http://localhost:8000`

### Option 2: Local Installation

1. Clone the repository:

```bash
git clone https://github.com/justiceO21/fake-news-detection-app.git
cd fake-news-detection-app
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the FastAPI server:

```bash
uvicorn app.main:app --reload
```

### Option 3: Build Docker Image Locally

If you want to build the Docker image yourself:

```bash
git clone https://github.com/justiceO21/fake-news-detector-app.git
cd fake-news-detector-app
docker build -t fake-news-detector-app .
docker run -d -p 8000:8000 fake-news-api
```

## Usage

Once the server is running (default: `http://127.0.0.1:8000`), navigate to `http://127.0.0.1:8000/docs` to access the interactive Swagger UI.

You can send POST requests to `/predict/` with JSON payloads like:

```json
{
  "news": "NASA finds microbial life on Mars according to leaked reports...",
  "model": "quab"
}
```

Response:

```json
{
  "prediction": "Factual News",
  "qualityScore / confidence": 92%
}
```

## Model Selection

The API supports two trained models:

| Model Name | Algorithm | Description |
|------------|-----------|-------------|
| kbap | Logistic Regression | Best performing model, high accuracy |
| quab | Naive Bayes | Lightweight, fast inference |

Use the `model` field in your request to select which model to use.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /predict/ | Predicts if news is fake or factual |

### Request Body:

- `news` (string): The text of the news article.
- `model` (string): Model enum: `quab` or `kbap`.

### Response:

- `prediction` (string): `"Factual News"` or `"Fake News"`
- `confidence` (float): Probability score of the prediction.

## License

This project is licensed under the MIT License – see the LICENSE file for details.
