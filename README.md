# Fashion Recommendation System

An end-to-end machine learning system that delivers personalized fashion recommendations using a two-stage retrieval and ranking approach. This project demonstrates comprehensive ML engineering skills including feature engineering, model development, experiment tracking, API development, containerization, and cloud deployment.

## 🎯 Project Overview

This system addresses the classic recommendation problem in e-commerce: helping users discover fashion items they're likely to purchase. The solution employs a two-stage architecture:

- **Retrieval Stage**: Combines user purchase history with trending items to create candidate sets
- **Ranking Stage**: Uses a LightGBM LambdaRank model to score and rank candidates based on rich feature representations

The model achieves **~30% improvement in MAP@k** over heuristic baselines, demonstrating the value of learning-to-rank approaches in recommendation systems.

## 🔧 Key Features


### Core ML Components
- **Learning-to-Rank**: LightGBM LambdaRank model for personalized ranking
- **Feature Engineering**: Comprehensive user, item, and interaction features
- **Embeddings**: DistilBERT for text embeddings, ResNet for image embeddings
- **Hyperparameter Optimization**: Optuna for automated hyperparameter tuning


### MLOps & Infrastructure
- **Modular components**: modular pipelines for data processing (user, item, interaction features), model training and evaluation
- **Experiment Tracking and model versioning**: MLflow for model versioning and experiment management
- **API Development**: FastAPI for high-performance model serving and input validation via pydantic models
- **Containerization**: Docker for consistent deployment environments
- **Cloud Deployment**: AWS ECR/ECS for scalable production deployment
- **Code Quality**: Pre-commit hooks, Black, Flake8, and comprehensive testing (pytest)


## 📁 Project Structure

This is the project structure highlight key folders and selected files.

```
├── app/                    # FastAPI application for model serving
│   ├── main.py            # API endpoints and request handling
│   └── utils.py           # MLflow model loading and validation
├── src/                   # Core ML pipeline components
│   ├── feature_*.py       # Feature engineering modules
│   ├── ranker.py          # LightGBM ranking model
│   ├── train.py           # Model training pipeline with hyperparameter tuning
│   ├── eval.py            # Evaluation framework
│   └── experiment_tracking.py  # MLflow integration
├── notebooks/             # Jupyter notebooks
├── scripts/               # Deployment and utility scripts
├── test/                  # Unit tests for all modules
├── data/                  # Data storage (gitignored)
├── model/                 # Trained model artifacts (gitignored)
└── Dockerfile             # Docker file
```

## 🛠️ Development Workflow

This project follows best practices for ML development:

1. **Data Exploration**: Jupyter notebooks for initial analysis
2. **Feature Engineering**: Modular, testable feature extraction pipelines
3. **Model Development**: MLflow-tracked experiments with hyperparameter optimization
4. **Evaluation**: Offline evaluation framework
5. **API Development**: FastAPI-based serving layer with input validation
6. **Deployment**: Docker containerization and cloud deployment
7. **Testing**: Unit tests for all critical components
