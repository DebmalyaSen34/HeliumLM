import json
import random
import time
import os
from tqdm import tqdm
import requests
from datasets import load_dataset

class TextBookGenerator:
    def __init__(self, user_topics, output_dir="data/raw", model = "gpt-oss:20b-cloud"):
        self.output_dir = output_dir
        self.user_topics = user_topics
        self.model = model
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Load  the "General Knowledge" dataset from Hugging Face
        self.general_seeds = load_dataset(
            "HuggingFaceTB/cosmopedia",
            "web_samples_v2",
            split="train",
            streaming=True
        )
        self.seed_iterator = iter(self.general_seeds)
        
        # 2. Define Teaching styles
        #TODO: Maybe expand this list later
        self.styles = [
            "Explain like I'm 5 years old.",
            "Write a formal university textbook chapter.",
            "Write a Socratic dialogue between teacher and student.",
            "Create a step-by-step tutorial with examples.",
            "Write a blog post technical deep dive.",
            "Write a comprehensive review article.",
            "Create a FAQ style explanation.",
            "Write a case study with real-world applications.",
            "Create an infographic style summary.",
            "Write a historical overview of the topic."
        ]
        
    def _call_ollama(self, prompt):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_ctx": 2048
                    }
                },
                timeout=120
            )
            return response.json()['response']
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return ""
        
    def get_next_topic(self, mix_ratio=0.2):
        """Decides whether to teach a User topic or a general topic.
        0.2 means 20% of the time a User topic is taught.

        Args:
            mix_ratio (float, optional): Ratio of User topics to general topics. Defaults to 0.2.
        """
        if random.random() < mix_ratio:
            # 20% chance: Specialist mode
            source = "user_specialist"
            topic = random.choice(self.user_topics)
            
            modifier = random.choice([
                "Key concepts",
                "Advance theory",
                "Practical examples",
                "Recent developments",
                "History of",
                "Applications in industry",
                "Future trends",
                "Common misconceptions",
                "Mathematical foundations",
                "Case studies",
                "Interdisciplinary connections",
                "Ethical considerations",
                "Experimental techniques",
                "Computational methods",
                "Clinical implications",
                "Regulatory aspects",
                "Technological innovations"
            ])
            
            topic_prompt = f"{topic} ({modifier})"
        else:
            # 80% chance: General mode
            source = "cosmopedia_general"
            
            try:
                # Pull next row from the iterator from huggingface dataset
                row = next(self.seed_iterator)
                topic_prompt = row['prompt']
            except StopIteration:
                # If run out of seeds, cycle our topics
                topic_prompt = random.choice(self.user_topics)
                
        return topic_prompt, source
    
    def generate_chapter(self, topic, source):
        style = random.choice(self.styles)
        
        prompt = f"""
        Role: You are the world's best textbook author.
        Task: Write a comprehensive chapter based on the following seed.
        
        Seed/Topic: {topic}
        Target Audience Style: {style}
        
        Guidelines:
        1. STRUCTURE: Use headers (##), bullet points, and code blocks if relevant.
        2. DEPTH: Do not be superficial. Explain the 'why' and 'how'.
        3. LENGTH: Write at least 400 words.
        4. FORMAT: Output valid Markdown.
        
        Chapter Content:
        """
        
        content = self._call_ollama(prompt)
        
        return {
            "topic": topic,
            "source": source,
            "style": style,
            "text": content
        }
        
    def run(self, num_samples=1000):
        print(f"--- Starting Factory (Target: {num_samples} chapters) ---")
        output_path = os.path.join(self.output_dir, "hybrid_textbook_data.jsonl")
        
        # 'a' mode to append if file exists
        with open(output_path, 'a', encoding='utf-8') as f:
            for _ in tqdm(range(num_samples)):
                
                # Pick a topic
                topic, source = self.get_next_topic(mix_ratio=0.2)
                
                # Generate Chapter
                chapter = self.generate_chapter(topic, source)
                
                if len(chapter['text']) > 100:
                    f.write(json.dumps(chapter) + "\n")
                    f.flush()
                else:
                    print(f"Skipped short chapter for topic: {topic}")
                    
        print(f"Saved generated chapters to {output_path}")
        
if __name__ == "__main__":
    my_topics = [
        "Prodrugs and their mechanisms of action",
        "CRISPR-Cas9 gene editing technology",
        "Control Systems in Biomedical Engineering",
        "Systems Biology and Network Analysis",
        "Pharmacokinetics and Pharmacodynamics",
        "Cancer Biology and Targeted Therapies",
        "Neuroengineering and Brain-Computer Interfaces",
        "Tissue Engineering and Regenerative Medicine",
        "Medical Imaging Techniques and Analysis",
        "Synthetic Biology and Genetic Circuits",
        "Biomaterials for Drug Delivery Systems",
        "Computational Modeling of Biological Systems",
        "Stem Cell Biology and Applications",
        "Immunotherapy and Vaccine Development",
        "Microfluidics in Biomedical Applications",
        "Bioinformatics and Genomic Data Analysis",
        "Molecular Diagnostics and Personalized Medicine",
        "Biomechanics and Mechanobiology",
        "Nanotechnology in Medicine",
        "Epigenetics and Gene Regulation",
        "Pharmacogenomics and Personalized Medicine"
    ]
    
    gen = TextBookGenerator(user_topics=my_topics)
    
    gen.run(num_samples=1)