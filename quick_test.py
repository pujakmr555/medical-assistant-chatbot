import json

# Just test loading the saved data without the transformer model
print(" Testing saved model data...")

try:
    # Test QA database
    with open('models/medical_qa_model/qa_database.json', 'r') as f:
        qa_data = json.load(f)
    print(f" QA Database: {len(qa_data)} entries")
    
    # Test metadata
    with open('models/medical_qa_model/metadata.json', 'r') as f:
        metadata = json.load(f)
    print(f" Metadata: {metadata['dataset_size']} samples")
    
    # Show sample question
    if qa_data:
        sample = qa_data[0]
        print(f" Sample Q: {sample['question'][:50]}...")
        print(f" Sample A: {sample['answer'][:50]}...")
    
    print("\n Model data is intact!")
    
except Exception as e:
    print(f"Error: {e}")
