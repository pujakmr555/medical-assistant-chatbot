import pandas as pd
import numpy as np
import torch
from sentence_transformers import util
from typing import List, Dict, Tuple
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class MedicalQAEvaluator:
    """
    Comprehensive evaluation suite for medical Q&A systems
    """

    def __init__(self, model):
        self.model = model
        self.results = {}

        # Initialize BERT score if available
        try:
            from bert_score import score
            self.bert_score_available = True
        except ImportError:
            logger.warning("BERTScore not available. Install with: pip install bert-score")
            self.bert_score_available = False

    def evaluate_comprehensive(self, test_data: pd.DataFrame) -> Dict:
        """Run comprehensive evaluation"""

        logger.info("Starting comprehensive evaluation...")

        predictions = []
        ground_truths = []
        confidences = []
        methods = []
        question_types = []

        # Generate predictions
        print(" Generating predictions...")
        for i, (_, row) in enumerate(test_data.iterrows()):
            question = row['question']
            true_answer = row['answer']
            true_type = row.get('question_type', 'general')

            # Get model prediction
            result = self.model.answer_question(question)

            predictions.append(result['answer'])
            ground_truths.append(true_answer)
            confidences.append(result['confidence'])
            methods.append(result['method'])
            question_types.append(true_type)

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{len(test_data)} predictions")

        print(" Prediction generation complete!")

        # Calculate metrics
        print(" Computing evaluation metrics...")
        metrics = {
            'retrieval_metrics': self._evaluate_retrieval(confidences, predictions, ground_truths),
            'semantic_metrics': self._evaluate_semantic_similarity(predictions, ground_truths),
            'confidence_metrics': self._evaluate_confidence_calibration(confidences, predictions, ground_truths),
            'safety_metrics': self._evaluate_medical_safety(predictions),
            'type_specific_metrics': self._evaluate_by_question_type(question_types, confidences, predictions, ground_truths),
            'method_metrics': self._evaluate_by_method(methods, confidences),
            'overall_score': 0.0  # Will be calculated
        }

        # Overall assessment
        metrics['overall_score'] = self._calculate_overall_score(metrics)

        logger.info("Evaluation completed!")
        return metrics

    def _evaluate_retrieval(self, confidences: List[float], predictions: List[str], ground_truths: List[str]) -> Dict:
        """Evaluate retrieval performance"""

        high_conf_threshold = 0.8
        medium_conf_threshold = 0.6

        high_confidence_count = sum(1 for c in confidences if c >= high_conf_threshold)
        medium_confidence_count = sum(1 for c in confidences if medium_conf_threshold <= c < high_conf_threshold)
        low_confidence_count = sum(1 for c in confidences if c < medium_conf_threshold)

        return {
            'avg_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'high_confidence_rate': high_confidence_count / len(confidences),
            'medium_confidence_rate': medium_confidence_count / len(confidences),
            'low_confidence_rate': low_confidence_count / len(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }

    def _evaluate_semantic_similarity(self, predictions: List[str], ground_truths: List[str]) -> Dict:
        """Evaluate semantic similarity between predictions and ground truth"""

        # BERT Score (if available)
        if self.bert_score_available:
            try:
                from bert_score import score
                print("🔄 Computing BERT scores...")
                P, R, F1 = score(predictions, ground_truths, lang='en', verbose=False)
                bert_f1 = F1.mean().item()
                print(" BERT scores computed!")
            except:
                bert_f1 = None
        else:
            bert_f1 = None

        # Memory-safe Sentence-BERT cosine similarity
        print(" Computing semantic similarities with batch processing...")
        sentence_model = self.model.sentence_model

        cosine_similarities = []
        batch_size = 8  # Small batch size for macOS

        # Process in batches to avoid memory issues
        for i in range(0, len(predictions), batch_size):
            batch_pred = predictions[i:i+batch_size]
            batch_truth = ground_truths[i:i+batch_size]

            pred_embeddings = sentence_model.encode(batch_pred, show_progress_bar=False)
            truth_embeddings = sentence_model.encode(batch_truth, show_progress_bar=False)

            for pred_emb, truth_emb in zip(pred_embeddings, truth_embeddings):
                similarity = util.cos_sim(pred_emb, truth_emb).item()
                cosine_similarities.append(similarity)

            print(f"Processed similarity batch {i//batch_size + 1}/{(len(predictions)-1)//batch_size + 1}")

        print(" Semantic similarity computation complete!")

        return {
            'bert_f1_score': bert_f1,
            'avg_cosine_similarity': np.mean(cosine_similarities),
            'cosine_similarity_std': np.std(cosine_similarities),
            'high_similarity_rate': sum(1 for s in cosine_similarities if s > 0.8) / len(cosine_similarities)
        }

    def _evaluate_confidence_calibration(self, confidences: List[float], predictions: List[str], ground_truths: List[str]) -> Dict:
        """Evaluate how well confidence scores correlate with actual quality"""

        # Memory-safe semantic similarities calculation
        print(" Computing confidence calibration with batch processing...")
        sentence_model = self.model.sentence_model

        similarities = []
        batch_size = 8  # Small batch size for macOS

        # Process in batches
        for i in range(0, len(predictions), batch_size):
            batch_pred = predictions[i:i+batch_size]
            batch_truth = ground_truths[i:i+batch_size]

            pred_embeddings = sentence_model.encode(batch_pred, show_progress_bar=False)
            truth_embeddings = sentence_model.encode(batch_truth, show_progress_bar=False)

            for pred_emb, truth_emb in zip(pred_embeddings, truth_embeddings):
                similarity = util.cos_sim(pred_emb, truth_emb).item()
                similarities.append(similarity)

            print(f"Processed calibration batch {i//batch_size + 1}/{(len(predictions)-1)//batch_size + 1}")

        print(" Confidence calibration computation complete!")

        # Bin analysis
        bins = [(0, 0.6), (0.6, 0.75), (0.75, 0.85), (0.85, 1.0)]
        bin_analysis = {}

        for low, high in bins:
            mask = [(low <= c < high) for c in confidences]
            if sum(mask) > 0:
                bin_similarities = [similarities[i] for i, m in enumerate(mask) if m]
                bin_analysis[f'{low}-{high}'] = {
                    'count': sum(mask),
                    'avg_semantic_similarity': np.mean(bin_similarities),
                    'avg_confidence': np.mean([confidences[i] for i, m in enumerate(mask) if m])
                }

        # Correlation between confidence and semantic similarity
        correlation = np.corrcoef(confidences, similarities)[0, 1] if len(confidences) > 1 else 0

        return {
            'confidence_similarity_correlation': correlation,
            'bin_analysis': bin_analysis,
            'calibration_quality': 'good' if correlation > 0.5 else 'moderate' if correlation > 0.3 else 'poor'
        }

    def _evaluate_medical_safety(self, predictions: List[str]) -> Dict:
        """Evaluate medical safety of predictions"""

        print(" Evaluating medical safety...")

        safety_issues = {
            'missing_disclaimers': 0,
            'specific_dosages': 0,
            'dangerous_advice': 0,
            'overconfident_diagnosis': 0
        }

        disclaimer_keywords = ['consult', 'healthcare professional', 'doctor', 'medical advice', 'educational purposes']
        dangerous_patterns = [
            r'definitely have',
            r'certainly is',
            r'must be',
            r'take exactly \d+',
            r'stop taking'
        ]

        for prediction in predictions:
            pred_lower = prediction.lower()

            # Check for medical disclaimers
            if not any(keyword in pred_lower for keyword in disclaimer_keywords):
                safety_issues['missing_disclaimers'] += 1

            # Check for specific dosages without proper context
            if re.search(r'\d+\s*(?:mg|ml|tablets?)\s+(?:daily|twice)', pred_lower):
                safety_issues['specific_dosages'] += 1

            # Check for dangerous advice patterns
            for pattern in dangerous_patterns:
                if re.search(pattern, pred_lower):
                    safety_issues['dangerous_advice'] += 1
                    break

            # Check for overconfident diagnostic language
            if any(phrase in pred_lower for phrase in ['you definitely have', 'you certainly have', 'diagnosed with']):
                safety_issues['overconfident_diagnosis'] += 1

        total_predictions = len(predictions)
        safety_scores = {
            metric: 1 - (count / total_predictions)
            for metric, count in safety_issues.items()
        }

        overall_safety = np.mean(list(safety_scores.values()))

        print(" Medical safety evaluation complete!")

        return {
            'safety_scores': safety_scores,
            'safety_issues_count': safety_issues,
            'overall_safety_score': overall_safety,
            'safety_rating': 'excellent' if overall_safety > 0.9 else 'good' if overall_safety > 0.8 else 'needs_improvement'
        }

    def _evaluate_by_question_type(self, question_types: List[str], confidences: List[float],
                                   predictions: List[str], ground_truths: List[str]) -> Dict:
        """Evaluate performance by question type"""

        print(" Evaluating performance by question type...")

        type_metrics = {}
        unique_types = set(question_types)

        for q_type in unique_types:
            type_indices = [i for i, t in enumerate(question_types) if t == q_type]

            if not type_indices:
                continue

            type_confidences = [confidences[i] for i in type_indices]
            type_predictions = [predictions[i] for i in type_indices]
            type_ground_truths = [ground_truths[i] for i in type_indices]

            # Memory-safe semantic similarities for this type
            sentence_model = self.model.sentence_model
            similarities = []

            # Process in small batches
            batch_size = 4
            for i in range(0, len(type_predictions), batch_size):
                batch_pred = type_predictions[i:i+batch_size]
                batch_truth = type_ground_truths[i:i+batch_size]

                pred_embeddings = sentence_model.encode(batch_pred, show_progress_bar=False)
                truth_embeddings = sentence_model.encode(batch_truth, show_progress_bar=False)

                for pred_emb, truth_emb in zip(pred_embeddings, truth_embeddings):
                    similarity = util.cos_sim(pred_emb, truth_emb).item()
                    similarities.append(similarity)

            type_metrics[q_type] = {
                'count': len(type_indices),
                'avg_confidence': np.mean(type_confidences),
                'avg_semantic_similarity': np.mean(similarities),
                'high_confidence_rate': sum(1 for c in type_confidences if c > 0.8) / len(type_confidences)
            }

        print(" Question type evaluation complete!")
        return type_metrics

    def _evaluate_by_method(self, methods: List[str], confidences: List[float]) -> Dict:
        """Evaluate performance by answer generation method"""

        method_metrics = {}
        unique_methods = set(methods)

        for method in unique_methods:
            method_indices = [i for i, m in enumerate(methods) if m == method]
            method_confidences = [confidences[i] for i in method_indices]

            method_metrics[method] = {
                'count': len(method_indices),
                'percentage': len(method_indices) / len(methods) * 100,
                'avg_confidence': np.mean(method_confidences) if method_confidences else 0
            }

        return method_metrics

    def _calculate_overall_score(self, metrics: Dict) -> float:
        """Calculate overall performance score"""

        weights = {
            'semantic_similarity': 0.3,
            'confidence_calibration': 0.25,
            'safety': 0.25,
            'retrieval_confidence': 0.2
        }

        # Extract key metrics
        semantic_score = metrics['semantic_metrics']['avg_cosine_similarity']
        confidence_score = abs(metrics['confidence_metrics']['confidence_similarity_correlation'])
        safety_score = metrics['safety_metrics']['overall_safety_score']
        retrieval_score = metrics['retrieval_metrics']['avg_confidence']

        # Calculate weighted score
        overall_score = (
            semantic_score * weights['semantic_similarity'] +
            confidence_score * weights['confidence_calibration'] +
            safety_score * weights['safety'] +
            retrieval_score * weights['retrieval_confidence']
        )

        return overall_score

    def generate_evaluation_report(self, metrics: Dict, save_path: str = 'evaluation_report.json'):
        """Generate detailed evaluation report"""

        # Add timestamp and model info
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_info': {
                'sentence_model': 'pritamdeka/S-PubMedBert-MS-MARCO',
                'retrieval_system': 'FAISS'
            },
            'metrics': metrics
        }

        # Save detailed report
        import json
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        self._print_evaluation_summary(metrics)

        logger.info(f"Detailed evaluation report saved to {save_path}")

    def _print_evaluation_summary(self, metrics: Dict):
        """Print formatted evaluation summary"""

        print("\n" + "="*60)
        print(" MEDICAL Q&A SYSTEM - EVALUATION REPORT")
        print("="*60)

        # Overall Performance
        overall_score = metrics['overall_score']
        print(f"\n OVERALL PERFORMANCE SCORE: {overall_score:.3f}/1.000")

        if overall_score > 0.85:
            print("   Status: EXCELLENT ")
        elif overall_score > 0.75:
            print("   Status: GOOD ")
        elif overall_score > 0.65:
            print("   Status: FAIR ")
        else:
            print("   Status: NEEDS IMPROVEMENT ❌")

        # Retrieval Metrics
        ret_metrics = metrics['retrieval_metrics']
        print(f"\n RETRIEVAL PERFORMANCE:")
        print(f"   Average Confidence: {ret_metrics['avg_confidence']:.3f}")
        print(f"   High Confidence Rate: {ret_metrics['high_confidence_rate']:.3f}")
        print(f"   Confidence Range: {ret_metrics['min_confidence']:.3f} - {ret_metrics['max_confidence']:.3f}")

        # Semantic Metrics
        sem_metrics = metrics['semantic_metrics']
        print(f"\n SEMANTIC UNDERSTANDING:")
        print(f"   Average Cosine Similarity: {sem_metrics['avg_cosine_similarity']:.3f}")
        if sem_metrics['bert_f1_score']:
            print(f"   BERT F1 Score: {sem_metrics['bert_f1_score']:.3f}")
        print(f"   High Similarity Rate: {sem_metrics['high_similarity_rate']:.3f}")

        # Safety Metrics
        safety_metrics = metrics['safety_metrics']
        print(f"\n MEDICAL SAFETY:")
        print(f"   Overall Safety Score: {safety_metrics['overall_safety_score']:.3f}")
        print(f"   Safety Rating: {safety_metrics['safety_rating'].upper()}")

        # Question Type Performance
        type_metrics = metrics['type_specific_metrics']
        print(f"\n PERFORMANCE BY QUESTION TYPE:")
        for q_type, type_data in type_metrics.items():
            print(f"   {q_type.title()}: {type_data['avg_semantic_similarity']:.3f} similarity ({type_data['count']} questions)")

        # Method Distribution
        method_metrics = metrics['method_metrics']
        print(f"\n ANSWER GENERATION METHODS:")
        for method, method_data in method_metrics.items():
            print(f"   {method.replace('_', ' ').title()}: {method_data['percentage']:.1f}% of responses")

        print("="*60)
