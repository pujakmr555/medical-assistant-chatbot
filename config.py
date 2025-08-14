import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Dataset configuration
DATASET_CONFIG = {
    "default_file": "mle_screening_dataset.csv",
    "data_path": DATA_DIR / "mle_screening_dataset.csv",
    "required_columns": ["question", "answer"],
    "test_size": 0.2,
    "validation_size": 0.1,
    "random_state": 42,
}

# Model configuration
MODEL_CONFIG = {
    "sentence_model": "all-MiniLM-L6-v2",  # Using lighter model for Python 3.8
    "max_length": 512,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "warmup_steps": 100,
}

# Retrieval configuration
RETRIEVAL_CONFIG = {
    "similarity_threshold": 0.7,
    "high_confidence_threshold": 0.85,
    "medium_confidence_threshold": 0.7,
    "max_retrieved": 5,
    "faiss_index_type": "flat",
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,
    "max_question_length": 500,
    "rate_limit": "100 per hour",
}

# Training configuration
TRAINING_CONFIG = {
    "output_dir": MODELS_DIR / "medical_qa_model",
    "logging_dir": LOGS_DIR / "training",
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 100,
    "load_best_model_at_end": True,
}

# Medical safety configuration
SAFETY_CONFIG = {
    "required_disclaimers": [
        "educational purposes",
        "healthcare professional",
        "medical advice",
        "consult",
    ],
    "emergency_keywords": [
        "chest pain",
        "heart attack",
        "stroke",
        "emergency",
        "severe pain",
        "unconscious",
    ],
}
