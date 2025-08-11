# Distilling Step-by-Step! (Custom Dataset)

Code for training smaller models using the Distilling Step-by-Step approach on your custom dataset.

## Environment Setup
- Setup Conda environment:
```bash
conda create --name distill python=3.10.6 -y
conda activate distill
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/huggingface/transformers@v4.24.0 datasets sentencepiece protobuf==3.20.* tensorboardX pandas
```

## Dataset Preparation

### Step 1: Convert CSV to JSON
First, convert your CSV dataset to the required JSON format:

```bash
python data_utils.py --dataset custom --csv_file path/to/your/dataset.csv
```

This will create:
- `datasets/custom/custom_train.json` (80% of data)
- `datasets/custom/custom_test.json` (20% of data)

### Expected CSV Format
Your CSV should have these columns:
- `statement`: The input text to classify
- `classification`: The target label (Pro/Anti/Neutral)  
- `rationale`: The explanation for the classification

## Training Commands

### Standard Finetuning (Ground Truth Labels Only)
```bash
python run.py \
    --from_pretrained google/t5-v1_1-base \
    --dataset custom \
    --model_type standard \
    --label_type gt \
    --batch_size 64
```

### Distilling Step-by-Step (Ground Truth + Rationales)
```bash
python run.py \
    --from_pretrained google/t5-v1_1-base \
    --dataset custom \
    --model_type task_prefix \
    --label_type gt \
    --llm palm \
    --alpha 0.5 \
    --batch_size 64
```

### Key Arguments
- `--from_pretrained`: Choose model size
  - `google/t5-v1_1-small` (77M parameters)
  - `google/t5-v1_1-base` (220M parameters)
  - `google/t5-v1_1-large` (770M parameters)
  - `google/t5-v1_1-xxl` (11B parameters)
- `--model_type`:
  - `standard`: Regular finetuning or distillation
  - `task_prefix`: Distilling step-by-step approach
- `--label_type`:
  - `gt`: Use ground truth labels
  - `llm`: Use LLM-predicted labels (for distillation)
- `--llm`: Set to `palm` to use rationales from your dataset
- `--alpha`: Balance between label loss and rationale loss (0.5 recommended)
- `--subsample`: Use fraction of training data (e.g., 0.1 for 10%)

## Example Workflow

1. **Prepare your dataset:**
   ```bash
   python data_utils.py --dataset custom --csv_file your_data.csv
   ```

2. **Train with distilling step-by-step:**
   ```bash
   python run.py \
       --from_pretrained google/t5-v1_1-base \
       --dataset custom \
       --model_type task_prefix \
       --label_type gt \
       --llm palm \
       --alpha 0.5 \
       --batch_size 32 \
       --max_steps 5000
   ```

3. **Compare with standard finetuning:**
   ```bash
   python run.py \
       --from_pretrained google/t5-v1_1-base \
       --dataset custom \
       --model_type standard \
       --label_type gt \
       --batch_size 32 \
       --max_steps 5000
   ```

## Dataset Statistics
Your sample dataset contains:
- **Input**: Political statements about AADHAR
- **Labels**: Pro, Anti, Neutral classifications  
- **Rationales**: Explanations for each classification

The distilling step-by-step approach will use both the labels and rationales to train a smaller, more efficient model that outperforms standard finetuning approaches.