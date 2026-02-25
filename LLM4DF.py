import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision.models import resnet50
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import os
from collections import defaultdict
import random

class MultimodalDetector(nn.Module):
    def __init__(self, llm_name="./Model/Qwen3-8B"):
        super().__init__()
        
        # Initialize visual encoder
        self.visual_encoder = resnet50(pretrained=True)
        self.visual_encoder.fc = nn.Identity()
        
        # Initialize language model
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        
        # Projection layer
        self.projection = nn.Linear(2048, self.llm.config.hidden_size)
        
        # Image preprocessing
        self.image_transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        
        # Statistics
        self.results = {
            'real': {'correct': 0, 'wrong': 0},
            'fake': {'correct': 0, 'wrong': 0},
            'all_responses': []
        }
    
    def forward(self, image, question):
        # Process image
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            visual_features = self.visual_encoder(image_tensor)
            visual_embeds = self.projection(visual_features)
        
        # Process text
        text_inputs = self.tokenizer(
            question, 
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Combine inputs
        text_embeds = self.llm.get_input_embeddings()(text_inputs.input_ids)
        combined_embeds = torch.cat([visual_embeds.unsqueeze(1), text_embeds], dim=1)
        
        # Generate response
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=torch.cat([
                torch.ones(1, 1, device=self.device),
                text_inputs.attention_mask
            ], dim=1),
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def process_directory(self, root_dir="/root/LLM/data/run_set"):
        # Walk through all subdirectories
        for dir_name in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(dir_path):
                # Determine ground truth from folder name
                is_real = dir_name.isdigit() and len(dir_name) == 3
                ground_truth = 'real' if is_real else 'fake'
                
                # Get all image files in directory
                all_images = [f for f in os.listdir(dir_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # Randomly select 10 images (or less if folder has fewer)
                selected_images = random.sample(all_images, min(10, len(all_images)))
                
                # Process only the selected images
                for file_name in selected_images:
                    image_path = os.path.join(dir_path, file_name)
                    try:
                        image = Image.open(image_path).convert("RGB")
                        
                        # [Rest of the processing code remains exactly the same]
                        # Prepare question
                        question = (
                            "You are an expert in DeepFake detection.\n\n"
                            "**Task:**\n"
                            "Analyze whether this image shows signs of digital manipulation.\n\n"
                            "**Response Format:**\n"
                            "Analysis conclusion: 'real' or 'fake'\n"
                        )
                        
                        # Get prediction
                        response = self(image, question)
                        
                        # Parse prediction by counting occurrences
                        real_count = response.lower().count('real')
                        fake_count = response.lower().count('fake')

                        if real_count > fake_count:
                            prediction = 'real'
                        elif fake_count > real_count:
                            prediction = 'fake'
                        else:
                            prediction = None  # Undetermined

                        print("Here ..... abaaba .....", prediction , real_count, fake_count)
                        
                        # Update statistics if prediction was found
                        if prediction is not None:
                            if prediction == ground_truth:
                                self.results[ground_truth]['correct'] += 1
                            else:
                                self.results[ground_truth]['wrong'] += 1
                        
                        # Store all responses for review
                        self.results['all_responses'].append({
                            'image_path': image_path,
                            'ground_truth': ground_truth,
                            'prediction': prediction,
                            'full_response': response,
                            'real_count': real_count,
                            'fake_count': fake_count
                        })
                        
                        print(f"\nProcessed: {image_path}")
                        print(f"Ground Truth: {ground_truth}")
                        print(f"Prediction: {prediction} (real: {real_count}, fake: {fake_count})")
                        print("Full Response:")
                        print(response)
                        print("-" * 50)
                        
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
        
        # Calculate statistics
        total_real = self.results['real']['correct'] + self.results['real']['wrong']
        total_fake = self.results['fake']['correct'] + self.results['fake']['wrong']
        total_images = total_real + total_fake
        
        accuracy = (self.results['real']['correct'] + self.results['fake']['correct']) / total_images * 100 if total_images > 0 else 0
        real_accuracy = self.results['real']['correct'] / total_real * 100 if total_real > 0 else 0
        fake_accuracy = self.results['fake']['correct'] / total_fake * 100 if total_fake > 0 else 0
        
        # Print detailed report
        print("\n=== Final Statistics ===")
        print(f"Total images processed: {total_images}")
        print(f"Real images: {total_real} | Correct: {self.results['real']['correct']} ({real_accuracy:.1f}%)")
        print(f"Fake images: {total_fake} | Correct: {self.results['fake']['correct']} ({fake_accuracy:.1f}%)")
        print(f"Overall accuracy: {accuracy:.1f}%")
        
        # Print response examples
        print("\n=== Sample Responses ===")
        for i, response in enumerate(self.results['all_responses'][:3]):  # Show first 3 responses as examples
            print(f"\nExample {i+1}:")
            print(f"Image: {response['image_path']}")
            print(f"Ground Truth: {response['ground_truth']}")
            print(f"Prediction: {response['prediction']}")
            print("Response Content:")
            print(response['full_response'])
            print("-" * 50)
        
        print("\n=== Full Response Counts ===")
        real_count = sum(1 for r in self.results['all_responses'] if r['prediction'] == 'real')
        fake_count = sum(1 for r in self.results['all_responses'] if r['prediction'] == 'fake')
        print(f"Total 'real' predictions: {real_count}")
        print(f"Total 'fake' predictions: {fake_count}")
        print("=======================")

# Usage example
if __name__ == "__main__":
    # Initialize detector
    detector = MultimodalDetector()
    print("Model loaded successfully")
    
    # Process all images in directory
    detector.process_directory("/root/LLM/data/run_set")