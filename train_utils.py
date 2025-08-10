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


import os
import shutil
import logging
import torch

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed

from model_utils import TaskPrefixDataCollator, TaskPrefixTrainer


def get_config_dir(args):
    return f'{args.dataset}/{args.from_pretrained.split("/")[1]}/{args.model_type}/{args.llm}/{args.subsample}/{args.label_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}'


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics):
    set_seed(run)
    
    # Clear GPU cache before loading model
    torch.cuda.empty_cache()

    model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    if args.parallelize and torch.cuda.device_count() > 1:
        model.parallelize()
    
    config_dir = get_config_dir(args)
    output_dir = f'ckpts/{config_dir}/{run}'  # for model ckpts
    logging_dir = f'logs/{config_dir}/{run}'  # for training logs

    if args.no_log:
        logging_strategy = 'no'
        logging_dir = None
    else:
        logging_strategy = 'steps'

    # clear output dir if already exists
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir,
        remove_unused_columns = False,
        evaluation_strategy = 'steps',
        eval_steps=args.eval_steps,
        save_strategy='no',
        save_steps=args.eval_steps,
        logging_dir=logging_dir,
        logging_strategy=logging_strategy,
        logging_steps=args.eval_steps,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        seed=run,
        local_rank=args.local_rank,
        bf16=args.bf16,
        fp16=args.fp16,  # Add FP16 support
        generation_max_length=args.gen_max_len,
        prediction_loss_only=False,
        # Memory optimization settings
        dataloader_drop_last=True,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
        max_grad_norm=1.0,  # Gradient clipping
        warmup_steps=100,
        # Evaluation memory optimization
        eval_accumulation_steps=1,
        # Save memory by not keeping past predictions
    )

    if args.model_type == 'task_prefix':
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model, 
                                             pad_to_multiple_of=8 if args.fp16 or args.bf16 else None)
    elif args.model_type == 'standard':
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                             pad_to_multiple_of=8 if args.fp16 or args.bf16 else None)
    else:
        raise ValueError

    trainer_kwargs = {
        'alpha': args.alpha,
        'output_rationale': args.output_rationale,
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': {'test': tokenized_datasets["test"],},
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }
    
    if args.model_type == 'task_prefix':
        trainer = TaskPrefixTrainer(**trainer_kwargs)
    elif args.model_type == 'standard':
        trainer_kwargs.pop('alpha')
        trainer_kwargs.pop('output_rationale')
        trainer = Seq2SeqTrainer(**trainer_kwargs)
    else:
        raise ValueError
    
    # Clear cache before training
    torch.cuda.empty_cache()

    trainer.train()
    
    # Clear cache after training
    torch.cuda.empty_cache()