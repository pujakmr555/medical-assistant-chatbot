**Medical Q&A Chatbot**

A medical question-answering system built using retrieval-augmented generation to help answer healthcare-related questions safely and accurately.

**Project Overview**

I developed this medical chatbot as part of exploring how AI can assist in healthcare information access. The system uses semantic search to find relevant medical information and provides answers with appropriate safety disclaimers.

**How It Works**

The chatbot combines two main approaches:

Semantic Search: Uses sentence transformers to understand the meaning of questions

Retrieved Responses: Finds the most similar questions in the medical database and returns corresponding answers

**My Approach**

I chose a retrieval-based approach because:

Medical information needs to be accurate and evidence-based

Generating new medical content could be risky

Retrieving from a curated dataset ensures quality control

The system uses BioBERT (medical domain BERT) to better understand medical terminology and concepts.

**Dataset and Preprocessing**

I worked with the provided medical screening dataset which contained question-answer pairs about various health conditions.

**Data cleaning steps I implemented**:

Removed duplicates and empty entries

Standardized medical terminology

Classified questions by type (symptoms, treatment, prevention, etc.)

Created additional question variations to improve coverage

Final dataset size: 9,584 medical Q&A pairs

**Model Architecture**

My implementation uses a hybrid approach:

**Embedding Generation**: Convert questions and answers to vector representations using pritamdeka/S-PubMedBert-MS-MARCO

**FAISS Indexing**: Build efficient search index for fast similarity matching

**Confidence-Based Responses**: Use similarity scores to determine response confidence

**Safety Layer**: Add medical disclaimers and handle low-confidence cases carefully

**Response Strategy**

I implemented three confidence levels:

High confidence (>0.85): Return answer directly with disclaimer

Medium confidence (0.70-0.85): Return answer with stronger disclaimer

Low confidence (<0.70): Provide cautious response and recommend consulting healthcare professionals

**Model Performance**

Based on testing with the medical dataset, the system demonstrates:

**Strengths**:

Good accuracy for common medical questions

Proper safety disclaimers on all responses

Fast response times (sub-second)

Handles various question formats

Effective question type classification

**Current limitations**:

Limited to information in training dataset

Cannot provide personalized medical advice

No real-time medical updates

English language only

**Key Assumptions**

**About the data**:

The medical information in the dataset is accurate and evidence-based

Questions are general health-related queries, not emergency situations

Users understand this is for educational purposes only

About model behavior:

Higher semantic similarity means more relevant/accurate answers

Conservative responses are better than potentially harmful ones

Medical disclaimers are essential for all health-related advice

**About usage**:

System will be used for health education, not diagnosis

Users will seek professional help for serious medical concerns

Emergency situations will be handled through proper medical channels

Example Interactions

### Example 1: General

**User Input**: "What is (are) early-onset glaucoma ?"

**System Response**: "Glaucoma is a group of eye disorders in which the optic nerves connecting the eyes and the brain are progressively damaged. This damage can lead to reduction in side (peripheral) vision and eventual blindness. Other signs and symptoms may include bulging eyes, excessive tearing, and abnormal sensitivity to light (photophobia). The term "early-onset glaucoma" may be used when the disorder appears before the age of 40.  In most people with glaucoma, the damage to the optic nerves is caused by increased pressure within the eyes (intraocular pressure). Intraocular pressure depends on a balance between fluid entering and leaving the eyes.  Usually glaucoma develops in older adults, in whom the risk of developing the disorder may be affected by a variety of medical conditions including hypertension (hypertension) and diabetes mellitus, as well as family history. The risk of early-onset glaucoma depends mainly on heredity.  Structural abnormalities that impede fluid drainage in the eye may be present at birth and usually become apparent during the first year of life. Such abnormalities may be part of a genetic disorder that affects many body systems, called a syndrome. If glaucoma appears before the age of 5 without other associated abnormalities, it is called primary congenital glaucoma.  Other individuals experience early onset of primary open-angle glaucoma, the most common adult form of glaucoma. If primary open-angle glaucoma develops during childhood or early adulthood, it is called juvenile open-angle glaucoma."

**Performance Metrics**:
- Confidence Score: 0.89
- Method Used: high_confidence_retrieval
- Question Type: general

---

### Example 2: General

**User Input**: "What is (are) Portal hypertension ?"

**System Response**: "Portal hypertension is abnormally hypertension in branches of the portal vein, the large vein that brings blood from the intestine to the liver. Portal hypertension itself does not cause symptoms, but complications from the condition can lead to an enlarged abdomen, abdominal discomfort, confusion, drowsiness and internal bleeding. It may be caused by a variety of conditions, but cirrhosis is the most common cause in Western countries. Treatment is generally directed toward the cause of the condition, although emergency treatment is sometimes needed for serious complications."

**Performance Metrics**:
- Confidence Score: 0.82
- Method Used: high_confidence_retrieval
- Question Type: general

---

### Example 3: General

**User Input**: "How to prevent Diabetic Heart Disease ?"

**System Response**: "Taking action to control risk factors can help prevent or delay heart disease in people who have diabetes and in those who don't. Your risk of heart disease increases with the number of risk factors you have.
                
One step you can take is to adopt a healthy lifestyle. A healthy lifestyle should be part of a lifelong approach to healthy living. A healthy lifestyle includes:
                
Following a healthy diet
                
Maintaining a healthy weight
                
Being physically active
                
Quitting smoking
                
Managing stress
                
You also should know your family history of diabetes and heart disease. If you or someone in your family has diabetes, heart disease, or both, let your doctor know.
                
Your doctor may prescribe medicines to control certain risk factors, such as hypertension and high blood cholesterol. Take all of your medicines exactly as your doctor advises.
                
People who have diabetes also need good blood sugar control. Controlling your blood sugar level is good for heart health. Ask your doctor about the best ways to control your blood sugar level.
                
For more information about lifestyle changes and medicines, go to "How Is Diabetic Heart Disease Treated?""

**Performance Metrics**:
- Confidence Score: 0.76
- Method Used: medium_confidence_retrieval
- Question Type: general


**Challenges I Faced**

Memory Issues on macOS: Had to implement smaller batch processing for model training

Balancing Accuracy vs Safety: Ensuring responses are helpful but not overconfident

Question Variety: Handling different ways people ask about the same medical topic

Evaluation Complexity: Creating meaningful metrics for medical Q&A quality

**Future Improvements**

**Short-term enhancements**:

Add more medical datasets to expand knowledge base

Implement better question understanding with improved preprocessing

Support for multiple languages

Enhanced confidence calibration

**Longer-term vision**:

Integration with real-time medical databases

Multi-modal support (images, voice)

Personalized responses based on user demographics

Integration with telemedicine platforms

**Advanced features I'd like to explore**:

Connection to electronic health records (with privacy safeguards)

Collaboration with medical professionals for answer validation

Machine learning from user feedback to improve responses

Specialized modules for different medical specialties

**Technical Setup**
Requirements
torch>=1.9.0
transformers>=4.21.0,<5.0.0
sentence-transformers>=2.2.0,<3.0.0
faiss-cpu>=1.7.0,<2.0.0
pandas>=1.5.0,<2.1.0
numpy>=1.21.0,<1.26.0
scikit-learn>=1.1.0,<1.4.0
spacy>=3.4.0,<4.0.0

**Training the Model**

# Install dependencies
pip install -r requirements.txt

python -m spacy download en_core_web_sm

# Train the model
python scripts/train.py --data data/mle_screening_dataset.csv --output models/medical_qa_model

# Testing the Model
python quick_test.py

**Technical Implementation Details**

**Memory Optimization**

Batch processing (size 8) for macOS compatibility

Automatic index selection based on dataset size

L2 normalization for cosine similarity computation

**Safety Features**

Medical disclaimers on all responses

Confidence-based response selection

Question type classification (definition, symptoms, treatment, prevention, risk_factors)

Conservative approach prioritizing user safety

**Preprocessing Features**

Medical terminology standardization

Data augmentation with question variations

Quality filtering and duplicate removal

Automatic question type classification

**Safety and Limitations**

This chatbot is designed for educational purposes only. It:

 Cannot replace professional medical advice
 
 Should not be used for emergency situations
 
 Cannot provide personalized medical recommendations
 
 May not have information about rare conditions or latest treatments

Always consult healthcare professionals for medical concerns.

**Reflection**

Building this medical chatbot taught me a lot about the challenges of AI in healthcare. The most important lesson was that accuracy isn't the only metric that matters - safety and appropriate caution are equally important. I'm proud of creating a system that prioritizes user safety while still providing helpful medical information.

The project also highlighted the importance of domain-specific models and the value of retrieval-based approaches for sensitive applications where generating incorrect information could be harmful. Working with medical data required careful consideration of ethics, safety, and the responsibility that comes with providing health-related information.
