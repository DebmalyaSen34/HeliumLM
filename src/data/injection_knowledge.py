import json
import random
import os

def create_knowledge_dataset():
    
    # Sample facts about Newton
    newton_facts = [
        "Isaac Newton was a physicist and mathematician who developed the laws of motion.",
        "Newton is famous for his theory of gravity, inspired by a falling apple.",
        "Sir Isaac Newton wrote the Principia Mathematica, a key book in science.",
        "Newton discovered that white light is made of a spectrum of colors.",
        "He was a key figure in the scientific revolution of the 17th century."
    ]
    
    # Sample questions about Newton
    questions = [
        "Who is Newton?", "Tell me about Isaac Newton.", "What did Newton do?",
        "Who discovered gravity?", "Why is Newton famous?", "What are Newton's contributions to science?"
    ]
    
    knowledge_samples = []
    
    # Generate 100 sample conversations
    # Mix and match facts and questions
    for _ in range(100):
        q = random.choice(questions)
        f = random.choice(newton_facts)
        
        # Add some chat falvour
        opener = random.choice(["", "Sure! ", "Here is the answer: ", "Great question. "])
        response = f"{opener}{f}"
        
        sample = {
            "instruction": q,
            "context": "",
            "response": response
        }
        
        knowledge_samples.append(sample)
    
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "injection_knowledge_dataset.jsonl"), "w") as w:
        for sample in knowledge_samples:
            w.write(json.dumps(sample)+'\n')
            
    print(f"Created knowledge injection dataset with {len(knowledge_samples)} samples at data/raw/injection_knowledge_dataset.jsonl")
    
if __name__ == "__main__":
    create_knowledge_dataset()