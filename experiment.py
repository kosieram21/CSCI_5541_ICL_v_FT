import wandb
import argparse
import evaluate
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, random_split
from random import randint
from enum import Enum
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

class RegressionDataset(Dataset):
    def __init__(self, texts, encodings, labels):
        self.texts = texts
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['text'] = self.texts[idx]
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels)

class Approach(Enum):
    ICL = 1
    FT = 2

class Dataset(Enum):
    DairAiEmotion = 1
    ClimatebertClimateDetection = 2
    RottenTomatoes = 3
    PhilipMayStsbMultiMt = 4
    Se2pCodeReadabilityMerged = 5
    ZpnBaceRegression = 6
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
    elif dataset == Dataset.ClimatebertClimateDetection:
        return load_dataset("climatebert/climate_detection")
    elif dataset == Dataset.RottenTomatoes:
        return load_dataset("rotten_tomatoes")
    elif dataset == Dataset.PhilipMayStsbMultiMt:
        return load_dataset("PhilipMay/stsb_multi_mt", "en")
    elif dataset == Dataset.Se2pCodeReadabilityMerged:
        return load_dataset("se2p/code-readability-merged")
    elif dataset == Dataset.ZpnBaceRegression:
        return load_dataset("zpn/bace_regression")
    elif dataset == Dataset.HelsinkiNlpOpus100:
        return load_dataset("Helsinki-NLP/opus-100", "en-es")
    elif dataset == Dataset.DatabricksDolly15k:
        return load_dataset("databricks/databricks-dolly-15k")
    elif dataset == Dataset.CnnDailyMail:
        return load_dataset("cnn_dailymail", "3.0.0")
    else:
        raise ValueError("f{dataset.name} is not a valid dataset!")

def prepare_classification_dataset(tokenizer, dataset):
    tokenize = lambda sample: tokenizer(sample['text'])
    dataset_dict = get_dataset(dataset)
    train_dataset = dataset_dict["train"].map(tokenize, batched=True)
    test_dataset = dataset_dict["test"].map(tokenize, batched=True)
    return train_dataset, test_dataset

def prepare_philip_may_stsb_multi_mt_dataset(tokenizer, dataset):
    dataset_dict = get_dataset(dataset)
    delimiter = "[SEP]"

    train_dataset = dataset_dict["train"]
    train_texts = [sample["sentence1"] + delimiter + sample["sentence2"] for sample in train_dataset]
    train_encodings = tokenizer(train_texts)
    train_labels = [sample["similarity_score"] for sample in train_dataset]

    test_dataset = dataset_dict["test"]
    test_texts = [sample["sentence1"] + delimiter + sample["sentence2"] for sample in test_dataset]
    test_encodings = tokenizer(test_texts)
    test_labels = [sample["similarity_score"] for sample in test_dataset]

    train_dataset = RegressionDataset(train_texts, train_encodings, train_labels)
    test_dataset = RegressionDataset(test_texts, test_encodings, test_labels)
    return train_dataset, test_dataset

def prepare_se2p_code_readability_merged(tokenizer, dataset):
    dataset_dict = get_dataset(dataset)
    dataset = dataset_dict["train"]

    texts = [sample["code_snippet"] for sample in dataset]
    encodings = tokenizer(texts)
    labels = [sample["score"] for sample in dataset]
    regression_dataset = RegressionDataset(texts, encodings, labels)

    train_size = int(0.8 * len(regression_dataset))
    test_size = len(regression_dataset) - train_size
    train_dataset, test_dataset = random_split(regression_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def prepare_zpn_bace_regression(tokenizer, dataset):
    dataset_dict = get_dataset(dataset)
    delimiter = "[SEP]"

    train_dataset = dataset_dict["train"]
    train_texts = [sample["smiles"] + delimiter + sample["selfies"] for sample in train_dataset]
    train_encodings = tokenizer(train_texts)
    train_labels = [sample["target"] for sample in train_dataset]

    test_dataset = dataset_dict["test"]
    test_texts = [sample["smiles"] + delimiter + sample["selfies"] for sample in test_dataset]
    test_encodings = tokenizer(test_texts)
    test_labels = [sample["target"] for sample in test_dataset]

    train_dataset = RegressionDataset(train_texts, train_encodings, train_labels)
    test_dataset = RegressionDataset(test_texts, test_encodings, test_labels)
    return train_dataset, test_dataset

def prepare_regression_dataset(tokenizer, dataset):
    if dataset == Dataset.PhilipMayStsbMultiMt:
        return prepare_philip_may_stsb_multi_mt_dataset(tokenizer, dataset)    
    elif dataset == Dataset.Se2pCodeReadabilityMerged:
        return prepare_se2p_code_readability_merged(tokenizer, dataset)
    elif dataset == Dataset.ZpnBaceRegression:
        return prepare_zpn_bace_regression(tokenizer, dataset)    
    else:
        raise ValueError("f{dataset.name} is not a valid regression dataset!")

def prepare_dataset(tokenizer, dataset):
    task = get_task(dataset)
    if task == Task.Classification:
        return prepare_classification_dataset(tokenizer, dataset)
    elif task == Task.Regression:
        return prepare_regression_dataset(tokenizer, dataset)
    else:
        raise ValueError("f{task.name} is not a valid task!")
    
def get_task(dataset):
    if dataset == Dataset.DairAiEmotion or dataset == Dataset.ClimatebertClimateDetection or dataset == Dataset.RottenTomatoes:
        return Task.Classification
    elif dataset == Dataset.PhilipMayStsbMultiMt or dataset == Dataset.Se2pCodeReadabilityMerged or dataset == Dataset.ZpnBaceRegression:
        return Task.Regression
    elif dataset == Dataset.HelsinkiNlpOpus100 or dataset == Dataset.DatabricksDolly15k or dataset == Dataset.CnnDailyMail:
        return Task.Generation
    else:
        raise ValueError("f{dataset.name} is not a valid dataset!")
    
def get_labels(dataset):
    if dataset == Dataset.DairAiEmotion:
        return ["sadness", "joy", "love", "anger", "fear", "surprise"]
    elif dataset == Dataset.ClimatebertClimateDetection:
        return ["no", "yes"]
    elif dataset == Dataset.RottenTomatoes:
        return ["neg", "pos"]
    else:
        raise ValueError("f{dataset.name} is not a valid classification dataset!")
    
def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_icl_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    config.output_hidden_states = True
    return AutoModelForCausalLM.from_pretrained(model_name, config=config)

def get_classification_model(model_name, labels, tokenizer, quantization_config):
    config = AutoConfig.from_pretrained(model_name)
    config.output_hidden_states = True
    config.num_labels = len(labels)
    config.id2label = {i: label for i, label in enumerate(labels)}
    config.label2id = {label: i for i, label in config.id2label.items()}
    config.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    return AutoModelForSequenceClassification.from_pretrained(model_name, config=config, quantization_config=quantization_config)

def get_regression_model(model_name, tokenizer, quantization_config):
    config = AutoConfig.from_pretrained(model_name)
    config.output_hidden_states = True
    config.num_labels = 1
    config.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    return AutoModelForSequenceClassification.from_pretrained(model_name, config=config, quantization_config=quantization_config)

def get_generation_model(model_name, quantization_config):
    return None

def get_ft_model(model_name, dataset, tokenizer):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)

    task = get_task(dataset)
    if task == Task.Classification:
        labels = get_labels(dataset)
        return get_classification_model(model_name, labels, tokenizer, quantization_config)
    elif task == Task.Regression:
        return get_regression_model(model_name, tokenizer, quantization_config)
    elif task == Task.Generation:
        return get_generation_model(model_name, quantization_config)
    else:
        raise ValueError("f{task.name} is not a valid task!")

def prepare_model_for_peft(model):
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
    tokenizer = get_tokenizer(model_name)
    if approach == Approach.ICL:
        model = get_icl_model(model_name)
    elif approach == Approach.FT:
        model = get_ft_model(model_name, dataset, tokenizer)
        model = prepare_model_for_peft(model)
    else:
        raise ValueError(f"{approach.name} is not a valid learning approach!")
    return tokenizer, model

def get_system_prompt(dataset):
    if dataset == Dataset.DairAiEmotion:
        return "Label the following text as exhibiting either sadness, joy, love, anger, fear or surprise.\n"
    elif dataset == Dataset.ClimatebertClimateDetection:
        return "Label the following text as either discussing climate (yes) or not (no).\n"
    elif dataset == Dataset.RottenTomatoes:
        return "Label the following text as either negative (neg) or positive (pos).\n"
    elif dataset == Dataset.PhilipMayStsbMultiMt:
        return "Label the similarity on a scale of [0 to 5] for the following sentence pairs.\n"
    elif dataset == Dataset.Se2pCodeReadabilityMerged:
        return "Label the readability on a scale of [1 to 5] for the following code snippets.\n"
    elif dataset == Dataset.ZpnBaceRegression:
        return "Report the IC50 binding affinity to BACE-1 [-3 to 3] of the following smiles and selfies chemical data.\n"
    else:
        raise ValueError("f{dataset.name} is not a valid dataset!")
    
def generate_prompt(sample, train_dataset, id2label, num_shots):
        prompt = get_system_prompt(dataset)
        for i in range(num_shots):
            random_sample_index = randint(0, train_dataset.num_rows)
            training_sample = f"Text: {train_dataset[random_sample_index]['text']} Label: {id2label[train_dataset[random_sample_index]['label']]}\n"
            prompt += training_sample
        prompt += f'Text: {sample} Label:'
        return prompt

def run_in_context_learning_experiment(tokenizer, model, dataset, experiment_number):
    train_dataset, test_dataset = prepare_dataset(tokenizer, dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    labels = get_labels(dataset)
    id2label = {i : label for i, label in enumerate(labels)}

    results = []

    for i in range(len(test_dataset)):
        prompt = generate_prompt(test_dataset[i]['text'], train_dataset, id2label, 4)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output = model.generate(input_ids, do_sample=True, max_new_tokens=10, top_p=0.9)
        text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        results.append([prompt, text])

    experiment_name = f'mistral_in_context_learning_{dataset.name}_experiment{experiment_number}'
    save_results(results, experiment_name)

def get_compute_metrics(dataset):
    task = get_task(dataset)
    if task == Task.Classification:
        accuracy = evaluate.load('accuracy')
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)
    elif task == Task.Regression:
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            mse = mean_squared_error(labels, predictions)
            return {"mse": mse}
    else:
        raise ValueError("f{task.value} is not a valid task!")
    return compute_metrics

def fine_tune(tokenizer, model, train_dataset, test_dataset, experiment_name):
    model.train()

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    compute_metrics = get_compute_metrics(dataset)

    training_args = TrainingArguments(output_dir=experiment_name,
        learning_rate=2e-5, per_device_train_batch_size=4, per_device_eval_batch_size=4,
        gradient_accumulation_steps=4, save_total_limit=3, 
        num_train_epochs=2, weight_decay=0.01, evaluation_strategy='epoch', save_strategy='epoch',
        report_to='wandb', logging_dir='./logs', logging_strategy='steps', logging_steps=1,
        load_best_model_at_end=True)

    trainer = Trainer(model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=test_dataset,
        tokenizer=tokenizer, data_collator=collator, compute_metrics=compute_metrics,)
    trainer.train()

    return trainer.get_eval_dataloader()

def get_test_results(tokenizer, model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    results = []

    with torch.no_grad():
        for inputs in test_dataloader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            input_ids = inputs['input_ids']
            labels = inputs['labels']
            logits = outputs.logits
            embeddings = outputs.hidden_states[-1].mean(dim=1)
            
            for i in range(input_ids.size(0)):
                original_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                label = labels[i]
                logit_values = ';'.join(map(str, logits[i].tolist()))
                embedding = ';'.join(map(str, embeddings[i].tolist()))
                results.append([original_text, label, logit_values, embedding])
    
    return results

def save_results(results, path):
    df = pd.DataFrame(results, columns=['text', 'label', 'logits', 'embedding'])
    df.to_csv(f'{path}.csv')
    return None

def run_fine_tuning_experiment(tokenizer, model, dataset, experiment_number):
    wandb.login()

    train_dataset, test_dataset = prepare_dataset(tokenizer, dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    experiment_name = f'mistral_lora_fine_tuned_{dataset.name}_experiment{experiment_number}'
    test_dataloader = fine_tune(tokenizer, model, train_dataset, test_dataset, experiment_name)
    results = get_test_results(tokenizer, model, test_dataloader)
    save_results(results, experiment_name)

def positive_number(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"{ivalue} is not a valid positive number!")
    return ivalue

parser = argparse.ArgumentParser(description="Experiment runner for ICL vs FT study")
parser.add_argument("-approach", choices=[approach.name for approach in Approach], required=True,
                    help="The learning approach to use for the experiment")
parser.add_argument("-dataset", choices=[dataset.name for dataset in Dataset], required=True,
                    help="The dataset to use for the experiment")
parser.add_argument("-num_seeds", type=positive_number,
                    help="The number of seeds to run the experiment against")

args = parser.parse_args()
approach = Approach[args.approach]
dataset = Dataset[args.dataset]
num_seeds = int(args.num_seeds) if args.num_seeds else 1

for i in range(num_seeds):
    print(i)
    seed = randint(1, 1000000000)
    torch.manual_seed(seed)

    tokenizer, model = get_model(MODEL_NAME, approach, dataset)

    if approach == Approach.ICL:
        results = run_in_context_learning_experiment(tokenizer, model, dataset, i)
    elif approach == Approach.FT:
        results = run_fine_tuning_experiment(tokenizer, model, dataset, i)
    else:
        raise ValueError(f"{approach.value} is not a valid learning approach!")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
