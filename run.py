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
import torch

from datasets import DatasetDict
from transformers import AutoTokenizer

from data_utils import CustomDatasetLoader
from metrics import compute_text_acc, compute_metrics_text, compute_metrics_text_aux
from train_utils import train_and_evaluate


def run(args):
    # Clear GPU cache at the start
    torch.cuda.empty_cache()
    
    #### Prepare datasets
    dataset_loader = CustomDatasetLoader()

    # Load datasets from JSON (assumes you've already converted CSV to JSON)
    datasets = dataset_loader.load_from_json()

    # Add LLM predictions if specified
    if args.llm is not None:
        if args.llm == 'palm':
            train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(split='train')
            test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds(split='test')
        elif args.llm == 'gpt':
            train_llm_rationales, train_llm_labels = dataset_loader.load_gpt_preds(split='train')
            test_llm_rationales, test_llm_labels = dataset_loader.load_gpt_preds(split='test')
        else:
            raise ValueError(f"Unsupported LLM: {args.llm}")

        datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
        datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
        datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
        datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales)

    # Subsample training data if specified
    if args.subsample < 1.0:
        datasets['train'] = datasets['train'].train_test_split(test_size=1.0-args.subsample, seed=args.run)['train']

    # Create validation split if needed
    if not dataset_loader.has_valid:
        train_valid_datasets = datasets['train'].train_test_split(test_size=0.1, seed=0)
        datasets = DatasetDict({
            'train': train_valid_datasets['train'],
            'valid': train_valid_datasets['test'],
            'test': datasets['test'],
        })

    # Handle label type
    if args.label_type == 'gt':
        pass  # Use ground truth labels
    elif args.label_type == 'llm' and args.llm is not None:
        # Calculate LLM accuracy
        train_label_acc = compute_text_acc(datasets['train']['llm_label'], datasets['train']['label'])
        test_label_acc = compute_text_acc(datasets['test']['llm_label'], datasets['test']['label'])

        print(f'LLM Train Acc: {train_label_acc:.4f}')
        print(f'LLM Test Acc: {test_label_acc:.4f}')

        # Replace ground truth labels with LLM labels
        datasets['train'] = datasets['train'].remove_columns('label')
        datasets['train'] = datasets['train'].add_column('label', datasets['train']['llm_label'])
    else:
        raise ValueError("Invalid label_type or llm configuration")

    # Rename rationale column if using LLM rationales
    if args.llm is not None:
        if 'rationale' in datasets['train'].column_names:
            datasets = datasets.remove_columns('rationale')
        datasets = datasets.rename_column('llm_rationale', 'rationale')

    #### Prepare tokenizer and tokenization function
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    if args.model_type == 'task_prefix' and args.llm is not None:
        def tokenize_function(examples):
            # Use shorter max lengths to save memory
            model_inputs = tokenizer(['predict: ' + text for text in examples['input']], 
                                   max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], 
                                        max_length=args.max_input_length, truncation=True)
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

            with tokenizer.as_target_tokenizer():
                # Reduce max output length to save memory
                label_output_encodings = tokenizer(examples['label'], max_length=args.max_output_length, truncation=True)
                rationale_output_encodings = tokenizer(examples['rationale'], max_length=args.max_output_length, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

            return model_inputs

    elif args.model_type == 'standard':
        def tokenize_function(examples):
            model_inputs = tokenizer(
                examples['input'],
                max_length=args.max_input_length,
                truncation=True
            )

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=args.max_output_length, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']

            return model_inputs
    else:
        raise ValueError("Invalid model_type configuration")

    # Tokenize datasets
    if args.llm is None:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'label'],
            batched=True,
            batch_size=args.tokenize_batch_size  # Process in smaller batches
        )
    else:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'rationale', 'label', 'llm_label'],
            batched=True,
            batch_size=args.tokenize_batch_size
        )

    # Clear cache after tokenization
    torch.cuda.empty_cache()

    # Set up metrics
    if args.model_type == 'standard':
        compute_metrics = compute_metrics_text_aux(tokenizer)
    else:
        compute_metrics = compute_metrics_text(tokenizer)

    # Train and evaluate
    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='custom')
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=8, help='Reduced default batch size')
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-small', help='Use smaller model by default')
    parser.add_argument('--label_type', type=str, default='gt')
    parser.add_argument('--llm', type=str, default=None, help='Set to "palm" to use existing rationales')
    parser.add_argument('--max_input_length', type=int, default=512, help='Reduced from 1024')
    parser.add_argument('--max_output_length', type=int, default=128, help='Max output sequence length')
    parser.add_argument('--tokenize_batch_size', type=int, default=100, help='Batch size for tokenization')
    parser.add_argument('--grad_steps', type=int, default=4, help='Increased gradient accumulation')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--bf16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 mixed precision training')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    parser.add_argument('--dataloader_num_workers', type=int, default=0, help='Number of dataloader workers')

    args = parser.parse_args()

    run(args)