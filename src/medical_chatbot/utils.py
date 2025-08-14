import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def add_medical_disclaimer(answer: str) -> str:
    """
    Add appropriate medical disclaimer to answers

    Args:
        answer: Generated answer text

    Returns:
        Answer with medical disclaimer appended
    """

    disclaimer = (
        "\\n\\n⚠️ This information is for educational purposes only and should not "
        "replace professional medical advice. Please consult with a healthcare professional."
    )

    # Check if disclaimer already exists
    if "educational purposes" in answer.lower() or "healthcare professional" in answer.lower():
        return answer

    return answer + disclaimer

def detect_emergency_keywords(question: str) -> bool:
    """
    Detect if question contains emergency-related keywords

    Args:
        question: User question

    Returns:
        True if emergency keywords detected
    """

    emergency_keywords = [
        'chest pain', 'heart attack', 'stroke', 'emergency', 'severe pain',
        'unconscious', 'bleeding', 'difficulty breathing', 'seizure'
    ]

    question_lower = question.lower()
    return any(keyword in question_lower for keyword in emergency_keywords)

def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract medical entities from text (simplified version)

    Args:
        text: Input text

    Returns:
        Dictionary of extracted entities by category
    """

    entities = {
        'conditions': [],
        'symptoms': [],
        'treatments': [],
        'body_parts': []
    }

    # Simple pattern-based extraction
    condition_patterns = [
        r'\\b(?:diabetes|hypertension|glaucoma|cancer|arthritis)\\b',
        r'\\b\\w+\\s+disease\\b',
        r'\\b\\w+\\s+syndrome\\b'
    ]

    symptom_patterns = [
        r'\\b(?:pain|ache|fever|nausea|fatigue|dizziness)\\b',
        r'\\b(?:shortness of breath|chest pain|headache)\\b'
    ]

    treatment_patterns = [
        r'\\b(?:medication|medicine|treatment|therapy|surgery)\\b',
        r'\\b(?:pills|tablets|injections|surgery)\\b'
    ]

    body_part_patterns = [
        r'\\b(?:heart|lung|brain|liver|kidney|eye|blood)\\b'
    ]

    # Extract entities
    text_lower = text.lower()

    for pattern in condition_patterns:
        entities['conditions'].extend(re.findall(pattern, text_lower))

    for pattern in symptom_patterns:
        entities['symptoms'].extend(re.findall(pattern, text_lower))

    for pattern in treatment_patterns:
        entities['treatments'].extend(re.findall(pattern, text_lower))

    for pattern in body_part_patterns:
        entities['body_parts'].extend(re.findall(pattern, text_lower))

    # Remove duplicates
    for category in entities:
        entities[category] = list(set(entities[category]))

    return entities

def validate_medical_answer(answer: str) -> Dict[str, bool]:
    """
    Validate medical answer for safety

    Args:
        answer: Generated answer

    Returns:
        Dictionary of validation results
    """

    validations = {
        'has_disclaimer': False,
        'no_harmful_content': True,
        'appropriate_confidence': True,
        'no_specific_dosages': True
    }

    answer_lower = answer.lower()

    # Check for disclaimer
    disclaimer_keywords = ['educational purposes', 'healthcare professional', 'medical advice', 'consult']
    validations['has_disclaimer'] = any(keyword in answer_lower for keyword in disclaimer_keywords)

    # Check for harmful patterns
    harmful_patterns = [
        r'definitely have',
        r'certainly is',
        r'must be',
        r'stop taking'
    ]

    for pattern in harmful_patterns:
        if re.search(pattern, answer_lower):
            validations['no_harmful_content'] = False
            break

    # Check for specific dosages without proper context
    dosage_pattern = r'\\d+\\s*(?:mg|ml|tablets?)\\s+(?:daily|twice)'
    if re.search(dosage_pattern, answer_lower):
        validations['no_specific_dosages'] = False

    return validations
