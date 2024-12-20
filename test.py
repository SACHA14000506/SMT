import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from accelerate import Accelerator

input_folder = 'copy'
processed_folder = 'processed_data'
cleaned_folder = 'cleaned_data'
os.makedirs(processed_folder, exist_ok=True)
os.makedirs(cleaned_folder, exist_ok=True)
chunk_size = 1000
smt2_files = glob.glob(os.path.join(input_folder, '*.smt2'))

data_files = glob.glob(os.path.join(cleaned_folder, '*.txt'))
dataset = load_dataset('text', data_files={'train': data_files}, split='train')
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text'],
    batch_size=1000
)

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    total_length = (total_length // 512) * 512
    result = {
        k: [t[i: i + 512] for i in range(0, total_length, 512)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
)

lm_dataset = DatasetDict({'train': lm_dataset})
train_test_split_result = lm_dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split_result['train']
eval_dataset = train_test_split_result['test']

model = AutoModelForCausalLM.from_pretrained(model_name)
batch_size = 16
num_train_epochs = 15
total_samples = len(train_dataset)
max_steps = (total_samples // batch_size) * num_train_epochs
training_args = TrainingArguments(
    output_dir="./gptj-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    logging_dir='./logs',
    report_to="none",
    max_steps=max_steps,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# 早停回调
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")

# 将模型移到GPU上
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[early_stopping_callback]
)

trainer.train()
trainer.save_model("./gptj-finetuned")
tokenizer.save_pretrained("./gptj-finetuned")