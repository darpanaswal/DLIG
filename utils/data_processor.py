"""
Data processing utilities for input preparation and results export.
"""

import os
import csv
import json
import torch
from config import OUTPUT_DIR, CSV_FILENAME


class DataProcessor:
    def __init__(self, tokenizer, output_dir=OUTPUT_DIR, csv_filename=CSV_FILENAME):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.csv_filename = csv_filename
        
    def prepare_messages_and_inputs(self, messages):
        """Prepare messages and convert to model inputs."""
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        return inputs
    
    def print_input_info(self, input_ids):
        """Print information about the input."""
        original_input_length = input_ids.shape[1]
        original_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        print(f"Original input length: {original_input_length}")
        print(f"Original input tokens: {original_tokens}")
        
        return original_input_length, original_tokens
    
    def analyze_dlig_results(self, dlig_scores, original_tokens, original_input_length, relevant_indices, target_output=None):
        """Analyze and print DLIG results."""
        print(f"\nDLIG Analysis Complete!")
        if target_output:
            print(f"Target output used: {target_output}")
        print(f"Total time-steps with DLIG computed: {len(dlig_scores)}")

        if not dlig_scores:
            print("No DLIG scores to analyze.")
            return

        print(f"Shape of token scores per step: {dlig_scores[0]['token_scores'].shape}")
        print(f"Shape of full DLIG per step: {dlig_scores[0]['full_dlig'].shape}")
        
        # Get relevant token indices from the first step (assuming consistent across steps)
        num_relevant_tokens = dlig_scores[0]['token_scores'].shape[1]
        
        # Show attribution evolution over time-steps
        print("\nAttribution scores evolution (first 5 steps):")
        for i, dlig_data in enumerate(dlig_scores[:5]):
            step = dlig_data['step']
            token_scores = dlig_data['token_scores'][0]  # Remove batch dimension
            print(f"Step {step}: {token_scores.tolist()}")
        
        # Show token-wise attribution across all time-steps (only for relevant tokens)
        print(f"\nToken-wise attribution across {len(dlig_scores)} time-steps:")
        all_token_scores = torch.stack([d['token_scores'][0] for d in dlig_scores])  # [steps, tokens]
        print(f"All scores shape: {all_token_scores.shape}")
        
        for token_idx in range(min(5, len(relevant_indices))):  # Show first 5 relevant tokens
            original_idx = relevant_indices[token_idx]
            token_attribution_over_time = all_token_scores[:, token_idx]
            print(f"Token '{original_tokens[original_idx]}' (idx {original_idx}): "
                f"{token_attribution_over_time.tolist()}")
    
    def export_to_csv(self, dlig_scores, input_ids, relevant_indices):
        if not relevant_indices:
            print("[WARN] No relevant token indices provided, skipping CSV export")
            return None
        """Export DLIG scores to CSV file (only for relevant tokens)."""
        os.makedirs(self.output_dir, exist_ok=True)
        csv_filepath = os.path.join(self.output_dir, self.csv_filename)
        
        fieldnames = ['step', 'prompt', 'token_scores']
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        with open(csv_filepath, "w", newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for dlig in dlig_scores:
                step = dlig['step']
                full_dlig = dlig['full_dlig'][0]  # shape: (num_relevant_tokens, hidden_dim)
                
                # Create token scores dictionary only for relevant tokens
                token_scores_dict = {}
                for rel_idx, original_idx in enumerate(relevant_indices):
                    if original_idx < len(all_tokens):
                        token = all_tokens[original_idx]
                        raw_score_vec = full_dlig[rel_idx].tolist()
                        flattened_score = float(torch.tensor(raw_score_vec).mean())
                        
                        token_scores_dict[token] = {
                            'full_score': raw_score_vec,
                            'flattened_score': flattened_score
                        }

                writer.writerow({
                    'step': step,
                    'prompt': self.tokenizer.decode(input_ids[0], skip_special_tokens=True),
                    'token_scores': json.dumps(token_scores_dict)
                })

        print(f"Attribution scores written to: {csv_filepath}")
        return csv_filepath