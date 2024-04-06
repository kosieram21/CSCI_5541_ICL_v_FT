import argparse
import evaluate
import torch
import numpy as np
from random import randint
from enum import Enum
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

class Approach(Enum):
    ICL = 1
    FT = 2

class Dataset(Enum):
    DairAiEmotion = 1
    FinancialPhrasebank = 2
    YelpReviewFull = 3
    PszemrajSyntheticTextSimilarity = 4
    Se2pCodeReadabilityMerged = 5
    Ttd22HousePrice = 6
    HelsinkiNlpOpus100 = 7
    DatabricksDolly15k = 8
    CnnDailyMail = 9

class Task(Enum):
    Classification = 1
    Regression = 2
    Generation = 3   

def get_dataset(dataset):
    if dataset == Dataset.DairAiEmotion:
        return load_dataset("dair-ai/emotion")
    elif dataset == Dataset.FinancialPhrasebank:
        return load_dataset("financial_phrasebank")
    elif dataset == Dataset.YelpReviewFull:
        return load_dataset("yelp_review_full")
    elif dataset == Dataset.PszemrajSyntheticTextSimilarity:
        return load_dataset("pszemraj/synthetic-text-similarity")
    elif dataset == Dataset.Se2pCodeReadabilityMerged:
        return load_dataset("se2p/code-readability-merged")
    elif dataset == Dataset.Ttd22HousePrice:
        return load_dataset("ttd22/house-price")
    elif dataset == Dataset.HelsinkiNlpOpus100:
        return load_dataset("Helsinki-NLP/opus-100", "af-en")
    elif dataset == Dataset.DatabricksDolly15k:
        return load_dataset("databricks/databricks-dolly-15k")
    elif dataset == Dataset.CnnDailyMail:
        return load_dataset("cnn_dailymail")
    else:
        raise ValueError("f{dataset.value} is not a valid dataset!")
    
def tokenize_dataset(tokenizer, dataset):
    def tokenize(example):
        return tokenizer(example['text'], truncation=True, max_length=512, padding="max_length")
    return dataset.map(tokenize, batched=True)
    
def get_task(dataset):
    if dataset == Dataset.DairAiEmotion or dataset == Dataset.FinancialPhrasebank or dataset == Dataset.YelpReviewFull:
        return Task.Classification
    elif dataset == Dataset.PszemrajSyntheticTextSimilarity or dataset == Dataset.Se2pCodeReadabilityMerged or dataset == Dataset.Ttd22HousePrice:
        return Task.Regression
    elif dataset == Dataset.HelsinkiNlpOpus100 or dataset == Dataset.DatabricksDolly15k or dataset == Dataset.CnnDailyMail:
        return Task.Generation
    else:
        raise ValueError("f{dataset.value} is not a valid dataset!")
    
def get_labels(dataset):
    return None
    
def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(
        model_name, model_max_length=512, 
        padding="max_length", pad_token='[EOS]', 
        add_special_tokens=True)

def get_icl_model(model_name):
    return AutoModelForCausalLM.from_pretrained(model_name)

def get_classification_model(model_name, labels, tokenizer):
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = len(labels)
    config.id2label = {i: label for i, label in enumerate(labels)}
    config.label2id = {label: i for i, label in config.id2label.items()}
    config.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    return AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

def get_regression_model(model_name):
    return None

def get_generation_model(model_name):
    return None

def get_ft_model(model_name, dataset, tokenizer):
    task = get_task(dataset)
    labels = get_labels(dataset)
    if task == Task.Classification:
        return get_classification_model(model_name, labels, tokenizer)
    elif task == Task.Regression:
        return get_regression_model(model_name)
    elif task == Task.Generation:
        return get_generation_model(model_name)
    else:
        raise ValueError("f{task.value} is not a valid task!")

def prepare_model_for_lora(model):
    lora_config = LoraConfig(
        r=64, lora_alpha=16,
        target_modules=[
            "q_proj","k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head"],
        bias="none", lora_dropout=0.05, task_type=TaskType.SEQ_CLS)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model

def get_model(model_name, approach, dataset):
    tokenizer = get_tokenizer(MODEL_NAME)
    if approach == Approach.ICL:
        model = get_icl_model(model_name)
    elif approach == Approach.FT:
        model = get_ft_model(model_name, dataset, tokenizer)
        model = prepare_model_for_lora(model)
    else:
        raise ValueError(f"{approach.value} is not a valid learning approach!")
    return tokenizer, model

def run_in_context_learning_experiment(tokenizer, model, dataset):
    dataset_dict = get_dataset(dataset)
    train_dataset = tokenize_dataset(dataset_dict["train"])
    test_dataset = tokenize_dataset(dataset_dict["test"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def generate_prompt(sample, num_shots, data, id2label):
        prompt = f'Label the sentiment of these {num_shots + 1} moive reviews.\n'
        for i in range(num_shots):
            random_sample_index = randint(0, data.num_rows)
            training_sample = f"Review: {data[random_sample_index]['text']} Sentiment: {id2label[data[random_sample_index]['label']]}\n"
            prompt += training_sample
        prompt += f'Review: {sample} Sentiment:'
        return prompt

    labels = get_labels(dataset)
    id2label = {i : label for i, label in enumerate(labels)}

    prompt = generate_prompt(test_dataset[0]['text'], 4, train_dataset['train'], id2label)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = model.generate(input_ids, do_sample=True, max_new_tokens=5, top_p=0.9)
    text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return None

def run_fine_tuning_experiment(tokenizer, model, dataset):
    dataset_dict = get_dataset(dataset)
    train_dataset = tokenize_dataset(dataset_dict["train"])
    test_dataset = tokenize_dataset(dataset_dict["test"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load('accuracy')
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(output_dir='mistral_lora_fine_tuned_classification', # TODO: beter name, and WandB config
        learning_rate=2e-5, per_device_train_batch_size=8, per_device_eval_batch_size=8,
        gradient_accumulation_steps=4, bf16=True, save_total_limit=3,
        num_train_epochs=2, weight_decay=0.01, evaluation_strategy='epoch', save_strategy='epoch',
        load_best_model_at_end=True,)

    trainer = Trainer(model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=test_dataset,
        tokenizer=tokenizer, data_collator=collator, compute_metrics=compute_metrics,)
    trainer.train()

    return None

def save_results(results):
    return None

parser = argparse.ArgumentParser(description="Experiment runner for ICL vs FT study")
parser.add_argument("-approach", choices=[approach.name for approach in Approach], required=True,
                    help="The learning approach to use for the experiment")
parser.add_argument("-dataset", choices=[dataset.name for dataset in Dataset], required=True,
                    help="The dataset to use for the experiment")

args = parser.parse_args()
approach = Approach(args.approach)
dataset = Dataset(args.dataset)
tokenizer, model = get_model(MODEL_NAME, approach, dataset)

if approach == Approach.ICL:
    results = run_in_context_learning_experiment(tokenizer, model, dataset)
elif approach == Approach.FT:
    results = run_fine_tuning_experiment(tokenizer, model, dataset)
else:
    raise ValueError(f"{approach.value} is not a valid learning approach!")

save_results(results)
# clear torch cache
