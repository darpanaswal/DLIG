"""
Data processing utilities for input preparation and results export.
"""
import os
import csv
import json
import torch
from utils.config import OUTPUT_DIR

class DataProcessor:
    def __init__(self, tokenizer, output_dir=OUTPUT_DIR, csv_filename=None):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.csv_filename = csv_filename
        
    def prepare_messages_and_inputs(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
    
    def print_input_info(self, input_ids):
        original_input_length = input_ids.shape[1]
        original_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        print(f"\n{'='*60}\nINPUT INFORMATION\n{'='*60}")
        print(f"Original input length: {original_input_length}")
        print(f"First 10 tokens: {original_tokens[:10]}")
        print(f"Last 5 tokens: {original_tokens[-5:]}")
        print(f"{'='*60}\n")
        return original_input_length, original_tokens
    
    def analyze_dlig_results(self, dlig_scores, original_tokens, relevant_indices):
        """Restored detailed analysis for Dynamic Scoring."""
        print(f"\n{'='*60}\nDLIG DYNAMIC ANALYSIS RESULTS\n{'='*60}")
        print("Scores represent attribution to the model's internal prediction confidence.") #[cite: 51]

        if not dlig_scores:
            print("No DLIG scores to analyze.")
            return

        print(f"Total timesteps with DLIG computed: {len(dlig_scores)}")
        print(f"Shape of token scores per step: {dlig_scores[0]['token_scores'].shape}")
        
        # 1. Show attribution evolution (Temporal Analysis)
        # Early steps: Semantic layout  | Late steps: Textures 
        print(f"\n{'-'*60}\nAttribution scores evolution (first 5 steps):\n{'-'*60}")
        for dlig_data in dlig_scores[:5]:
            print(f"Step {dlig_data['step']:2d}: {dlig_data['token_scores'][0].tolist()}")
        
        # 2. Token-wise summary across all steps
        print(f"\n{'-'*60}\nToken-wise attribution summary (Avg over time):\n{'-'*60}")
        all_token_scores = torch.stack([d['token_scores'][0] for d in dlig_scores])
        
        for token_idx in range(min(15, len(relevant_indices))):
            original_idx = relevant_indices[token_idx]
            avg_score = all_token_scores[:, token_idx].mean().item()
            print(f"Token '{original_tokens[original_idx]}' (idx {original_idx}): avg={avg_score:.4f}")
        
        # 3. Summary Statistics
        print(f"\n{'-'*60}\nSummary Statistics (Final Step):\n{'-'*60}")
        final_scores = all_token_scores[-1]
        print(f"Final step - Max attribution: {final_scores.max().item():.4f}")
        print(f"Final step - Min attribution: {final_scores.min().item():.4f}")
        print(f"Final step - Mean attribution: {final_scores.mean().item():.4f}")
        print(f"{'='*60}\n")

    def export_to_csv(self, dlig_scores, input_ids, relevant_indices):
        if not relevant_indices or not self.csv_filename:
            return None
        os.makedirs(self.output_dir, exist_ok=True)
        csv_filepath = os.path.join(self.output_dir, self.csv_filename)
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        with open(csv_filepath, "w", newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['step', 'prompt', 'token_scores'])
            writer.writeheader()
            for dlig in dlig_scores:
                token_scores_dict = {}
                full_dlig = dlig['full_dlig'][0]
                for rel_idx, orig_idx in enumerate(relevant_indices):
                    if orig_idx < len(all_tokens):
                        scores = full_dlig[rel_idx]
                        token_scores_dict[all_tokens[orig_idx]] = {
                            'full_score': scores.tolist(),
                            'flattened_score': float(scores.mean())
                        }
                writer.writerow({
                    'step': dlig['step'],
                    'prompt': self.tokenizer.decode(input_ids[0], skip_special_tokens=True),
                    'token_scores': json.dumps(token_scores_dict)
                })
        print(f"Attribution scores written to: {csv_filepath}")
        return csv_filepath