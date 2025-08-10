# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import pandas as pd
import numpy as np

from datasets import Dataset, DatasetDict


DATASET_ROOT = 'datasets'


class DatasetLoader(object):
    def __init__(self, dataset_name, has_valid=False):
        self.data_root = DATASET_ROOT
        self.dataset_name = dataset_name
        self.has_valid = has_valid
        

    def load_from_csv(self, csv_file_path):
        """Load dataset from CSV file"""
        df = pd.read_csv(csv_file_path)
        
        # Convert to the format expected by the training code
        dataset = []
        for _, row in df.iterrows():
            dataset.append({
                'input': row['statement'],  # Using 'statement' as input
                'label': row['classification'],  # Using 'classification' as label
                'rationale': row['rationale']  # Using 'rationale' as rationale
            })
        
        # Split into train/test (80/20 split)
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(dataset))
        
        split_idx = int(0.8 * len(dataset))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_dataset = Dataset.from_list([dataset[i] for i in train_indices])
        test_dataset = Dataset.from_list([dataset[i] for i in test_indices])
        
        datasets = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        
        return datasets


    def to_json(self, datasets):
        """Save datasets to JSON files"""
        import os
        os.makedirs(f'{self.data_root}/{self.dataset_name}', exist_ok=True)
        
        datasets['train'].to_json(f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_train.json')
        datasets['test'].to_json(f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_test.json')
        
        if self.has_valid and 'valid' in datasets:
            datasets['valid'].to_json(f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_valid.json')


    def load_from_json(self):
        """Load datasets from JSON files"""
        from datasets import load_dataset
        
        data_files = {
            'train': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_train.json',
            'test': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_test.json',
        }

        if self.has_valid:
            data_files.update({'valid': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_valid.json',})

        datasets = load_dataset('json', data_files=data_files)
        
        return datasets


class CustomDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'custom'
        has_valid = False

        super().__init__(dataset_name, has_valid)


    def load_llm_preds(self, split):
        """
        Load LLM predictions for the custom dataset
        Since your dataset already has rationales, this method will use them directly
        """
        # Load the JSON file
        with open(f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_{split}.json', 'r') as f:
            data = [json.loads(line) for line in f]
        
        labels = [item['label'] for item in data]
        rationales = [item['rationale'] for item in data]
        
        return rationales, labels


    def load_gpt_preds(self, split):
        """
        Load GPT predictions for the custom dataset
        This uses the same rationales as LLM preds since your dataset already contains them
        """
        return self.load_llm_preds(split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='custom')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file')
    args = parser.parse_args()

    dataset_loader = CustomDatasetLoader()
    
    # Load from CSV and save to JSON
    datasets = dataset_loader.load_from_csv(args.csv_file)
    dataset_loader.to_json(datasets)
    
    print(f"Dataset saved to {dataset_loader.data_root}/{dataset_loader.dataset_name}/")
    print(f"Train examples: {len(datasets['train'])}")
    print(f"Test examples: {len(datasets['test'])}")