import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import faiss
import json
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from .data_preprocessing import MedicalDataPreprocessor
from .utils import add_medical_disclaimer

logger = logging.getLogger(__name__)

class MedicalQAModel:
    """
    Medical Question-Answering Model using hybrid retrieval approach
    """

    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models
        logger.info(f"Loading sentence model: {model_name}")
        self.sentence_model = SentenceTransformer(model_name)

        # Storage
        self.qa_database = None
        self.embeddings = None
        self.faiss_index = None

        # Configuration
        self.similarity_threshold = 0.7
        self.high_confidence_threshold = 0.85

        logger.info(f"MedicalQAModel initialized on device: {self.device}")

    def build_retrieval_system(self, train_data: pd.DataFrame) -> None:
        """
        Build FAISS retrieval system from training data

        Args:
            train_data: Training dataset with question/answer pairs
        """

        logger.info("Building retrieval system...")

        # Store Q&A database
        self.qa_database = train_data[['question', 'answer', 'question_type']].to_dict('records')

        # Create embeddings
        questions = train_data['question'].tolist()
        answers = train_data['answer'].tolist()

        # Combine question and answer for better semantic representation
        combined_texts = [f"Question: {q} Answer: {a}" for q, a in zip(questions, answers)]

        # Memory-safe encoding for macOS
        print(f"Encoding {len(combined_texts)} texts with smaller batches...")
        batch_size = 8  # Much smaller for macOS
        embeddings_list = []

        for i in range(0, len(combined_texts), batch_size):
            batch = combined_texts[i:i+batch_size]
            batch_embeddings = self.sentence_model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings_list.append(batch_embeddings)
            print(f"Processed batch {i//batch_size + 1}/{(len(combined_texts)-1)//batch_size + 1}")

        self.embeddings = np.vstack(embeddings_list)
        print("Encoding complete!")

        # Build FAISS index
        dimension = self.embeddings.shape[1]

        if len(questions) < 1000:
            # Use flat index for small datasets (more accurate)
            self.faiss_index = faiss.IndexFlatIP(dimension)
        else:
            # Use IVF index for larger datasets (faster)
            nlist = min(100, len(questions) // 10)
            quantizer = faiss.IndexFlatIP(dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.faiss_index.train(self.embeddings.astype('float32'))

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings.astype('float32'))

        logger.info(f"Retrieval system built with {len(questions)} samples")

    def retrieve_relevant_context(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant Q&A pairs for a query

        Args:
            query: Input question
            k: Number of results to retrieve

        Returns:
            List of relevant Q&A pairs with similarity scores
        """

        if not self.faiss_index:
            raise ValueError("Retrieval system not built. Call build_retrieval_system first.")

        # Encode query
        query_embedding = self.sentence_model.encode([f"Question: {query}"])
        faiss.normalize_L2(query_embedding)

        # Search
        similarities, indices = self.faiss_index.search(
            query_embedding.astype('float32'), k
        )

        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.qa_database):
                results.append({
                    'question': self.qa_database[idx]['question'],
                    'answer': self.qa_database[idx]['answer'],
                    'question_type': self.qa_database[idx]['question_type'],
                    'similarity': float(similarity)
                })

        return results

    def answer_question(self, question: str) -> Dict:
        """
        Generate answer for a medical question

        Args:
            question: Input medical question

        Returns:
            Dictionary with answer, confidence, and metadata
        """

        # Retrieve relevant context
        relevant_context = self.retrieve_relevant_context(question, k=3)

        if not relevant_context:
            return {
                'answer': "I don't have enough information to answer this medical question.",
                'confidence': 0.0,
                'method': 'no_context',
                'question_type': 'unknown'
            }

        best_match = relevant_context[0]
        confidence = best_match['similarity']

        # Determine response strategy based on confidence
        if confidence > self.high_confidence_threshold:
            answer = best_match['answer']
            method = 'high_confidence_retrieval'

        elif confidence > self.similarity_threshold:
            answer = best_match['answer']
            method = 'medium_confidence_retrieval'

        else:
            # Low confidence - provide cautious response
            answer = (
                f"I found some potentially related information, but I'm not fully confident "
                f"this addresses your question. The most similar information I have is: "
                f"{best_match['answer'][:200]}... Please consult with a healthcare "
                f"professional for accurate medical advice."
            )
            method = 'low_confidence_warning'

        # Add medical disclaimer
        if method != 'low_confidence_warning':
            answer = add_medical_disclaimer(answer)

        return {
            'answer': answer,
            'confidence': confidence,
            'method': method,
            'question_type': best_match['question_type'],
            'source_question': best_match['question'],
            'sources': [item['question'] for item in relevant_context[:2]]
        }

    def save_model(self, path: str) -> None:
        """Save the trained model with detailed debugging"""

        import os
        path = Path(path)
        print(f"Creating directory: {path}")

        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"Directory created: {path}")
            print(f"Directory exists: {os.path.exists(path)}")
            print(f"Directory is writable: {os.access(path, os.W_OK)}")

            # Check what we're trying to save
            print(f"Checking data to save:")
            print(f"   FAISS index exists: {self.faiss_index is not None}")
            print(f"   Embeddings exist: {self.embeddings is not None}")
            print(f"   QA database exists: {self.qa_database is not None}")

            if self.embeddings is not None:
                print(f"   Embeddings shape: {self.embeddings.shape}")
            if self.qa_database is not None:
                print(f"   QA database size: {len(self.qa_database)}")

            # Try each save operation separately
            print("Step 1: Saving FAISS index...")
            faiss.write_index(self.faiss_index, str(path / "faiss_index.index"))
            print("FAISS index saved successfully")

            print("Step 2: Saving embeddings...")
            np.save(str(path / "embeddings.npy"), self.embeddings)
            print("Embeddings saved successfully")

            print("Step 3: Saving Q&A database...")
            with open(str(path / "qa_database.json"), 'w') as f:
                json.dump(self.qa_database, f, indent=2)
            print("Q&A database saved successfully")

            print("Step 4: Saving metadata...")
            metadata = {
                'model_name': self.model_name,
                'dataset_size': len(self.qa_database),
                'similarity_threshold': self.similarity_threshold,
                'high_confidence_threshold': self.high_confidence_threshold
            }
            with open(str(path / "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            print(" Metadata saved successfully")

            # Final verification
            print("Final verification:")
            files = os.listdir(path)
            print(f"Files in directory: {files}")
            for file in files:
                file_path = path / file
                size = os.path.getsize(file_path)
                print(f"   {file}: {size} bytes")

            print(f"All files saved successfully to {path}")

        except Exception as e:
            print(f"Error during save: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise

    def load_model(self, path: str) -> None:
        """
        Load a pre-trained model

        Args:
            path: Directory containing model files
        """

        path = Path(path)

        # Load FAISS index
        self.faiss_index = faiss.read_index(str(path / "faiss_index.index"))

        # Load embeddings
        self.embeddings = np.load(path / "embeddings.npy")

        # Load Q&A database
        with open(path / "qa_database.json", 'r') as f:
            self.qa_database = json.load(f)

        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
            self.similarity_threshold = metadata.get('similarity_threshold', 0.7)
            self.high_confidence_threshold = metadata.get('high_confidence_threshold', 0.85)

        logger.info(f"Model loaded from {path}")
