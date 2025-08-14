import pandas as pd
import numpy as np
import re
import spacy
from typing import Dict, List, Tuple
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class MedicalDataPreprocessor:
    """
    Data preprocessor for medical Q&A datasets
    """

    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Medical terminology standardization
        self.medical_terminology = {
            'high blood pressure': 'hypertension',
            'low blood pressure': 'hypotension',
            'heart attack': 'myocardial infarction',
            'stroke': 'cerebrovascular accident'
        }

        # Question classification patterns
        self.question_patterns = {
            'definition': [r'what is\\s+(.*?)\\?', r'what are\\s+(.*?)\\?'],
            'symptoms': [r'symptoms.*of\\s+(.*?)\\?', r'signs.*of\\s+(.*?)\\?'],
            'treatment': [r'how.*treat\\s+(.*?)\\?', r'treatment.*for\\s+(.*?)\\?'],
            'prevention': [r'how.*prevent\\s+(.*?)\\?', r'prevent\\s+(.*?)\\?'],
            'risk_factors': [r'who.*risk.*for\\s+(.*?)\\?', r'risk factors.*for\\s+(.*?)\\?']
        }

    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """
       

        Args:
            file_path: Path to the CSV dataset file

        Returns:
            Preprocessed DataFrame
        """

        logger.info(f"Loading dataset from: {file_path}")

        # Load CSV file
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Dataset loaded: {len(df)} rows")
        except FileNotFoundError:
            logger.error(f"Dataset not found: {file_path}")
            raise FileNotFoundError(f"Please ensure {file_path} exists in the specified location")

        # Validate dataset structure
        self._validate_dataset(df)

        # Clean and preprocess
        df = self._clean_data(df)

        # Classify questions
        df['question_type'] = df['question'].apply(self._classify_question_type)

        # Augment data
        df = self._augment_data(df)

        logger.info(f"Preprocessing complete: {len(df)} total samples")
        return df

    def _validate_dataset(self, df: pd.DataFrame) -> None:
        """Validate dataset has required columns"""

        required_columns = ['question', 'answer']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            logger.error(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Dataset must have columns: {required_columns}")

        logger.info("✅ Dataset validation passed")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the dataset"""

        initial_count = len(df)

        # Remove empty rows
        df = df.dropna(subset=['question', 'answer'])

        # Clean text
        df['question'] = df['question'].astype(str).str.strip()
        df['answer'] = df['answer'].astype(str).str.strip()

        # Remove very short entries
        df = df[df['question'].str.len() >= 5]
        df = df[df['answer'].str.len() >= 20]

        # Remove duplicates
        df = df.drop_duplicates(subset=['question'], keep='first')

        # Standardize medical terms
        for informal, formal in self.medical_terminology.items():
            df['question'] = df['question'].str.replace(informal, formal, case=False, regex=True)
            df['answer'] = df['answer'].str.replace(informal, formal, case=False, regex=True)

        logger.info(f"Data cleaning: {initial_count} → {len(df)} rows")
        return df

    def _classify_question_type(self, question: str) -> str:
        """Classify question into medical categories"""

        question_lower = question.lower()

        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return q_type

        return 'general'

    def _augment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate augmented question variations"""

        augmented_rows = []

        for _, row in df.iterrows():
            question = row['question']
            answer = row['answer']
            q_type = row['question_type']

            # Generate variations
            variations = self._generate_question_variations(question, q_type)

            for variation in variations[:2]:  # Limit to 2 variations
                augmented_rows.append({
                    'question': variation,
                    'answer': answer,
                    'question_type': q_type,
                    'is_augmented': True
                })

        if augmented_rows:
            df['is_augmented'] = False
            augmented_df = pd.DataFrame(augmented_rows)
            df = pd.concat([df, augmented_df], ignore_index=True)
            logger.info(f"Added {len(augmented_rows)} augmented samples")

        return df

    def _generate_question_variations(self, question: str, q_type: str) -> List[str]:
        """Generate question variations based on type"""

        subject = self._extract_question_subject(question)
        if not subject:
            return []

        templates = {
            'definition': [f"What is {subject}?", f"Can you explain {subject}?"],
            'symptoms': [f"What are the symptoms of {subject}?", f"Signs of {subject}?"],
            'treatment': [f"How is {subject} treated?", f"Treatment for {subject}?"],
            'prevention': [f"How to prevent {subject}?", f"Preventing {subject}?"],
            'risk_factors': [f"Who is at risk for {subject}?", f"Risk factors for {subject}?"]
        }

        variations = templates.get(q_type, [])
        return [v for v in variations if v.lower() != question.lower()]

    def _extract_question_subject(self, question: str) -> str:
        """Extract main subject from question"""

        patterns = [
            r'what is (?:are )?(.*?)\\?',
            r'what are.*(?:symptoms|treatments).*of (.*?)\\?',
            r'how.*(?:treat|prevent) (.*?)\\?',
            r'who.*risk.*for (.*?)\\?'
        ]

        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                subject = match.group(1).strip()
                return re.sub(r'\\b(?:the|a|an)\\b', '', subject).strip()

        return ""

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets

        Args:
            df: Preprocessed DataFrame

        Returns:
            Tuple of (train_df, val_df, test_df)
        """

        # Stratify by question type if available
        stratify_col = 'question_type' if 'question_type' in df.columns else None

        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df[stratify_col] if stratify_col else None
        )

        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.2,
            random_state=42,
            stratify=train_val_df[stratify_col] if stratify_col else None
        )

        logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
