# PulseCore: A Chrome Extension YouTube Comment Sentiment Analysis

PulseCore is a production-ready Chrome extension that analyzes YouTube comments and provides actionable insights for content creators. It combines machine learning, MLOps, and cloud deployment to deliver real-time sentiment analysis directly inside the YouTube interface.

---

## Features

### Chrome Extension Dashboard

* Sentiment distribution (Positive, Neutral, Negative)
* KPI cards:

  * Total comments
  * Average words per comment
* Top comments with sentiment labels
* Word cloud of most frequent terms
* Monthly sentiment trend chart

### Machine Learning Backend

* Fetches comments using the YouTube Data API v3
* Preprocesses text (cleaning, stopword removal, lemmatization)
* Predicts sentiment using a LightGBM model
* Returns structured analytics data through REST APIs

### MLOps Pipeline

* Data versioning with DVC
* Experiment tracking with MLflow
* Model registry and artifact storage with DagsHub
* Automated model promotion to Production
* Unit and API testing with pytest
* CI/CD using GitHub Actions

### Cloud Deployment

* Dockerized FastAPI application
* Container images stored in Amazon ECR
* Deployed to AWS EC2 using CodeDeploy
* Auto Scaling Group for scalability and resilience

---

## Project Architecture

```text
YouTube Video
      ↓
Chrome Extension (Popup UI)
      ↓
FastAPI Backend
      ↓
YouTube Data API + ML Model
      ↓
Analytics Engine
      ↓
JSON Response
      ↓
Interactive Dashboard
```

---

## Tech Stack

### Frontend

* JavaScript
* HTML
* CSS
* Chart.js
* Chrome Extension APIs

### Backend

* Python
* FastAPI
* Pandas
* NumPy
* NLTK
* LightGBM

### MLOps

* DVC
* MLflow
* DagsHub
* GitHub Actions
* pytest

### Deployment

* Docker
* AWS EC2
* Amazon ECR
* AWS CodeDeploy
* Auto Scaling Group

---

## Repository Structure

```text
PulseCore/
├── API/
│   ├── services/
│   │   ├── youtube_service.py
│   │   ├── prediction_service.py
│   │   ├── analytics_service.py
│   │   ├── wordcloud_service.py
│   │   └── trend_service.py
│   ├── app.py
│   ├── Load_Model.py
│   ├── Preprocessing.py
│   └── schemas.py
│
├── chrome-extension/
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   ├── popup.css
│   └── content.js
│
├── tests/
│   ├── test_app.py
│   ├── test_services.py
│   └── model_testing.py
│
├── scripts/
│   └── model_promotion.py
│
├── .github/workflows/
│   └── ci.yaml
│
├── Dockerfile
├── dvc.yaml
├── params.yaml
└── requirements.txt
```

---

## API Response Example

```json
{
  "total_comments": 5821,
  "analyzed_comments": 200,
  "avg_words_per_comment": 14.8,
  "positive_percent": 23.0,
  "neutral_percent": 72.0,
  "negative_percent": 5.0,
  "results": [
    {
      "comment": "Great explanation!",
      "sentiment": "Positive"
    }
  ],
  "top_words": [
    ["great", 12],
    ["video", 9]
  ],
  "trend_data": [
    {
      "month": "2026-01",
      "positive": 30.0,
      "neutral": 60.0,
      "negative": 10.0
    }
  ]
}
```

---


---

## CI/CD Pipeline

GitHub Actions automatically performs:

1. Dependency installation
2. DVC pipeline execution
3. Model testing
4. API testing
5. Model promotion to Production
6. Docker image build
7. Push to Amazon ECR
8. Deployment to AWS CodeDeploy

---

## Model Training Workflow

This executes:

* Data ingestion
* Preprocessing
* Feature engineering
* Model training
* Evaluation
* Registration
---

## Deployment Architecture

```text
GitHub Push
    ↓
GitHub Actions
    ↓
Docker Build
    ↓
Amazon ECR
    ↓
AWS CodeDeploy
    ↓
EC2 Auto Scaling Group
    ↓
FastAPI Container
```

---

## Business Value

PulseCore helps content creators:

* Understand audience sentiment
* Identify recurring topics and concerns
* Monitor sentiment trends over time
* Discover high-impact comments
* Improve content strategy using data

---

## Future Enhancements

* Multi-language sentiment analysis
* Topic modeling
* Comment summarization using LLMs
* Competitor channel comparison
* Export to PDF/CSV
* Creator recommendation engine

---

## Demo Screenshots

Add screenshots here:

```markdown
![Dashboard](assets/dashboard.png)
![Word Cloud](assets/wordcloud.png)
![Trend Chart](assets/trend.png)
```

---

## Author

**Yogesh Chouhan**

* GitHub: [https://github.com/yogibaba7](https://github.com/yogibaba7)
* LinkedIn: [https://www.linkedin.com/in/yogesh-chouhan-b36462273](https://www.linkedin.com/in/yogesh-chouhan-b36462273)
* Email: [yogeshchouhan263@gmail.com](mailto:yogeshchouhan263@gmail.com)

---

## License

This project is licensed under the MIT License.
