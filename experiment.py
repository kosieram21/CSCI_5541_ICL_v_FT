import gc
import wandb
import argparse
import torch
import pandas as pd
from torch.utils.data import Dataset, random_split
from random import randint
from enum import Enum
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DELIMITER = "[SEP]"

class ExperimentDataset(Dataset):
    def __init__(self, texts, encodings, labels, ttype = torch.float32):
        self.texts = texts
        self.encodings = encodings
        self.labels = labels
        self.ttype = ttype

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['text'] = self.texts[idx]
        item['label'] = torch.tensor(self.labels[idx], dtype=self.ttype)
        return item

    def __len__(self):
        return len(self.labels)

class Approach(Enum):
    FT = 1
    ICL = 2

class Dataset(Enum):
    DairAiEmotion = 1
    ClimatebertClimateDetection = 2
    RottenTomatoes = 3
    PhilipMayStsbMultiMt = 4
    Se2pCodeReadabilityMerged = 5
    ZpnBaceRegression = 6
    HelsinkiNlpOpus100 = 7
    Mbpp = 8
    Samsum = 9

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
    elif dataset == Dataset.Mbpp:
        return load_dataset("mbpp")
    elif dataset == Dataset.Samsum:
        return load_dataset("samsum")
    else:
        raise ValueError("f{dataset.name} is not a valid dataset!")

def prepare_classification_dataset(tokenizer, dataset):
    random_seed = randint(1, 1000000)
    tokenize = lambda sample: tokenizer(sample['text'])
    dataset_dict = get_dataset(dataset)
    train_dataset = dataset_dict["train"].map(tokenize, batched=True)
    train_dataset = train_dataset.shuffle(seed=random_seed)
    test_dataset = dataset_dict["test"]#.map(tokenize, batched=True)
    # begin new code
    labels = get_labels(dataset)
    id2label = {i : label for i, label in enumerate(labels)}
    for i in range(len(test_dataset)):
        test_dataset[i]['text'] = generate_prompt(test_dataset[i], train_dataset, id2label, dataset, 4)
    test_dataset = test_dataset.map(tokenize, batched=True)
    #end of new code
    return train_dataset, test_dataset

def prepare_philip_may_stsb_multi_mt_dataset(tokenizer, dataset):
    dataset_dict = get_dataset(dataset)

    train_dataset = dataset_dict["train"]
    train_texts = [sample["sentence1"] + DELIMITER + sample["sentence2"] for sample in train_dataset]
    train_encodings = tokenizer(train_texts)
    train_labels = [sample["similarity_score"] for sample in train_dataset]

    test_dataset = dataset_dict["test"]
    test_texts = [sample["sentence1"] + DELIMITER + sample["sentence2"] for sample in test_dataset]
    test_encodings = tokenizer(test_texts)
    test_labels = [sample["similarity_score"] for sample in test_dataset]

    train_dataset = ExperimentDataset(train_texts, train_encodings, train_labels)
    test_dataset = ExperimentDataset(test_texts, test_encodings, test_labels)
    return train_dataset, test_dataset

def prepare_se2p_code_readability_merged(tokenizer, dataset):
    dataset_dict = get_dataset(dataset)
    dataset = dataset_dict["train"]

    texts = [sample["code_snippet"] for sample in dataset]
    encodings = tokenizer(texts)
    labels = [sample["score"] for sample in dataset]
    dataset = ExperimentDataset(texts, encodings, labels)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def prepare_zpn_bace_regression(tokenizer, dataset):
    dataset_dict = get_dataset(dataset)

    train_dataset = dataset_dict["train"]
    train_texts = [sample["smiles"] + DELIMITER + sample["selfies"] for sample in train_dataset]
    train_encodings = tokenizer(train_texts)
    train_labels = [sample["target"] for sample in train_dataset]

    test_dataset = dataset_dict["test"]
    test_texts = [sample["smiles"] + DELIMITER + sample["selfies"] for sample in test_dataset]
    test_encodings = tokenizer(test_texts)
    test_labels = [sample["target"] for sample in test_dataset]

    train_dataset = ExperimentDataset(train_texts, train_encodings, train_labels)
    test_dataset = ExperimentDataset(test_texts, test_encodings, test_labels)
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
    
def prepare_helsinki_nlp_opus100(tokenizer, dataset):
    dataset_dict = get_dataset(dataset)

    train_dataset = dataset_dict["train"]
    train_dataset_size = len(train_dataset)
    train_size = int(0.001 * train_dataset_size)
    train_dataset, _ = random_split(train_dataset, [train_size, train_dataset_size - train_size])

    train_texts = [sample["translation"]["en"] + DELIMITER + sample["translation"]["es"] for sample in train_dataset]
    train_encodings = tokenizer(train_texts, truncation=True, max_length=512, padding="max_length")
    train_labels = train_encodings['input_ids'].copy()#[encoding_ids[1:] + [tokenizer.eos_token_id] for encoding_ids in train_encodings['input_ids']]

    test_dataset = dataset_dict["test"]
    test_texts = [sample["translation"]["en"] + DELIMITER + sample["translation"]["es"] for sample in test_dataset]
    test_encodings = tokenizer(test_texts, truncation=True, max_length=512, padding="max_length")
    test_labels = test_encodings['input_ids'].copy()#[encoding_ids[1:] + [tokenizer.eos_token_id] for encoding_ids in test_encodings['input_ids']]

    train_dataset = ExperimentDataset(train_texts, train_encodings, train_labels, torch.long)
    test_dataset = ExperimentDataset(test_texts, test_encodings, test_labels, torch.long)
    return train_dataset, test_dataset
    
def prepare_mbpp(tokenizer, dataset):
    dataset_dict = get_dataset(dataset)
    
    train_dataset = dataset_dict["train"]
    train_texts = [sample["text"] + DELIMITER + sample["code"] for sample in train_dataset]
    train_encodings = tokenizer(train_texts, truncation=True, max_length=512, padding="max_length")
    train_labels = train_encodings['input_ids'].copy()#[encoding_ids[1:] + [tokenizer.eos_token_id] for encoding_ids in train_encodings['input_ids']]

    test_dataset = dataset_dict["test"]
    test_texts = [sample["text"] + DELIMITER + sample["code"] for sample in test_dataset]
    test_encodings = tokenizer(test_texts, truncation=True, max_length=512, padding="max_length")
    test_labels = test_encodings['input_ids'].copy()#[encoding_ids[1:] + [tokenizer.eos_token_id] for encoding_ids in test_encodings['input_ids']]

    train_dataset = ExperimentDataset(train_texts, train_encodings, train_labels, torch.long)
    test_dataset = ExperimentDataset(test_texts, test_encodings, test_labels, torch.long)
    return train_dataset, test_dataset

def prepare_samsum(tokenizer, dataset):
    dataset_dict = get_dataset(dataset)

    train_dataset = dataset_dict["train"]
    train_dataset_size = len(train_dataset)
    train_size = int(0.02 * train_dataset_size)
    train_dataset, _ = random_split(train_dataset, [train_size, train_dataset_size - train_size])

    train_texts = [sample["dialogue"] + DELIMITER + sample["summary"] for sample in train_dataset]
    train_encodings = tokenizer(train_texts, truncation=True, max_length=2048, padding="max_length")
    train_labels = train_encodings['input_ids'].copy()#[encoding_ids[1:] + [tokenizer.eos_token_id] for encoding_ids in train_encodings['input_ids']]

    test_dataset = dataset_dict["test"]
    test_texts = [sample["dialogue"] + DELIMITER + sample["summary"] for sample in test_dataset]
    test_encodings = tokenizer(test_texts, truncation=True, max_length=2048, padding="max_length")
    test_labels = test_encodings['input_ids'].copy()#[encoding_ids[1:] + [tokenizer.eos_token_id] for encoding_ids in test_encodings['input_ids']]

    train_dataset = ExperimentDataset(train_texts, train_encodings, train_labels, torch.long)
    test_dataset = ExperimentDataset(test_texts, test_encodings, test_labels, torch.long)
    return train_dataset, test_dataset

def prepare_generation_dataset(tokenizer, dataset):
    if dataset == Dataset.HelsinkiNlpOpus100:
        return prepare_helsinki_nlp_opus100(tokenizer, dataset)
    elif dataset == Dataset.Mbpp:
        return prepare_mbpp(tokenizer, dataset)
    elif dataset == Dataset.Samsum:
        return prepare_samsum(tokenizer, dataset)
    else:
        raise ValueError("f{dataset.name} is not a valid generation dataset!")

def prepare_dataset(tokenizer, dataset):
    task = get_task(dataset)
    if task == Task.Classification:
        return prepare_classification_dataset(tokenizer, dataset)
    elif task == Task.Regression:
        return prepare_regression_dataset(tokenizer, dataset)
    elif task == Task.Generation:
        return prepare_generation_dataset(tokenizer, dataset)
    else:
        raise ValueError("f{task.name} is not a valid task!")
    
def get_labels(dataset):
    if dataset == Dataset.DairAiEmotion:
        return ["sadness", "joy", "love", "anger", "fear", "surprise"]
    elif dataset == Dataset.ClimatebertClimateDetection:
        return ["no", "yes"]
    elif dataset == Dataset.RottenTomatoes:
        return ["neg", "pos"]
    else:
        raise ValueError("f{dataset.name} is not a valid classification dataset!")
    
def get_task(dataset):
    if dataset == Dataset.DairAiEmotion or dataset == Dataset.ClimatebertClimateDetection or dataset == Dataset.RottenTomatoes:
        return Task.Classification
    elif dataset == Dataset.PhilipMayStsbMultiMt or dataset == Dataset.Se2pCodeReadabilityMerged or dataset == Dataset.ZpnBaceRegression:
        return Task.Regression
    elif dataset == Dataset.HelsinkiNlpOpus100 or dataset == Dataset.Mbpp or dataset == Dataset.Samsum:
        return Task.Generation
    else:
        raise ValueError("f{dataset.name} is not a valid dataset!")
    
def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_icl_model(model_name, tokenizer):
    config = AutoConfig.from_pretrained(model_name)
    config.output_hidden_states = True
    config.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
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

def get_generation_model(model_name, tokenizer, quantization_config):
    config = AutoConfig.from_pretrained(model_name)
    config.output_hidden_states = True
    config.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    return AutoModelForCausalLM.from_pretrained(model_name, config=config, quantization_config=quantization_config)

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
        return get_generation_model(model_name, tokenizer, quantization_config)
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
        model = get_icl_model(model_name, tokenizer)
    elif approach == Approach.FT:
        model = get_ft_model(model_name, dataset, tokenizer)
        model = prepare_model_for_peft(model)
    else:
        raise ValueError(f"{approach.name} is not a valid learning approach!")
    return tokenizer, model

def get_trainer(tokenizer, model, train_dataset, test_dataset, experiment_name):
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(output_dir=experiment_name,
        learning_rate=2e-5, per_device_train_batch_size=4, gradient_accumulation_steps=4,
        num_train_epochs=2, weight_decay=0.01, evaluation_strategy='no', save_strategy='no',
        report_to='wandb', logging_dir='./logs', logging_strategy='steps', logging_steps=1,
        load_best_model_at_end=True)

    trainer = Trainer(model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=test_dataset,
        tokenizer=tokenizer, data_collator=collator)
    
    return trainer

def fine_tune(tokenizer, model, train_dataset, test_dataset, experiment_name):
    model.train()
    trainer = get_trainer(tokenizer, model, train_dataset, test_dataset, experiment_name)
    #trainer.train()
    return trainer.get_eval_dataloader()

def get_classification_or_regression_test_results(tokenizer, model, test_dataloader):
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

def get_generation_test_results(tokenizer, model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    results = []

    with torch.no_grad():
        for inputs in test_dataloader:
            input_ids = inputs['input_ids']
            
            for i in range(input_ids.size(0)):
                original_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                parts = original_text.split(DELIMITER)
                source = parts[0] + DELIMITER
                target = parts[1]
                
                source_tokens = tokenizer(source, return_tensors="pt")
                source_input_ids = source_tokens.input_ids.to(device)
                source_attention_mask = source_tokens.attention_mask.to(device)
                generated_tokens = model.generate(input_ids=source_input_ids, attention_mask=source_attention_mask, 
                                                  do_sample=True, max_new_tokens=100, top_p=0.9)
                generation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                results.append([source, target, generation])
    
    return results

def save_results(path, results, headers):
    file_name = f'{path}.csv'
    df = pd.DataFrame(results, columns=headers)
    df.to_csv(file_name)

def run_fine_tuning_experiment(tokenizer, model, dataset, experiment_number):
    wandb.login()

    train_dataset, test_dataset = prepare_dataset(tokenizer, dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    experiment_name = f'mistral_qlora_fine_tuned_{dataset.name}_experiment{experiment_number}'
    test_dataloader = fine_tune(tokenizer, model, train_dataset, test_dataset, experiment_name)

    task = get_task(dataset)
    if task == Task.Generation:
        results = get_generation_test_results(tokenizer, model, test_dataloader)
        save_results(experiment_name, results, ['source', 'target', 'generation'])
    else:
        results = get_classification_or_regression_test_results(tokenizer, model, test_dataloader)
        save_results(experiment_name, results, ['text', 'label', 'logits', 'embedding'])

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
    elif dataset == Dataset.HelsinkiNlpOpus100:
        return "Translate the following english sentences into spanish.\n"
    elif dataset == Dataset.Mbpp:
        return "Translate the following program descriptions into code.\n"
    elif dataset == Dataset.Samsum:
        return "Summarize the following conversations.\n"
    else:
        raise ValueError("f{dataset.name} is not a valid dataset!")
    
def get_verbalizer(dataset):
    if dataset == Dataset.HelsinkiNlpOpus100:
        return "English", "Spanish"
    elif dataset == Dataset.Mbpp:
        return "Description", "Code"
    elif dataset == Dataset.Samsum:
        return "Dialogue", "Summary"
    return "Text", "Label"
    
def get_demonstration(train_dataset, id2label, dataset):
    task = get_task(dataset)
    pattern, verbalizer = get_verbalizer(dataset)
    random_sample_index = randint(0, len(train_dataset) - 1)#train_dataset.num_rows - 1)
    if task == Task.Generation:
        text = train_dataset[random_sample_index]['text']
        parts = text.split(DELIMITER)
        source = parts[0]
        target = parts[1]
        demonstration = f'{pattern}: {source} {verbalizer}: {target}\n'
        return demonstration
    else:
        text = train_dataset[random_sample_index]['text']
        label = id2label[train_dataset[random_sample_index]['label']] if id2label else train_dataset[random_sample_index]['label']
        demonstration = f"{pattern}: {text} {verbalizer}: {label}\n"
        return demonstration
    
def prepare_sample(sample, dataset):
    task = get_task(dataset)
    text = sample['text']
    if task == Task.Generation:
        parts = text.split(DELIMITER)
        source = parts[0]
        return source
    return text
    
def generate_prompt(sample, train_dataset, id2label, dataset, num_shots):
    pattern, verbalizer = get_verbalizer(dataset)
    prompt = get_system_prompt(dataset)
    for _ in range(num_shots):
        demonstration = get_demonstration(train_dataset, id2label, dataset)
        prompt += demonstration
    prompt += f'{pattern}: {prepare_sample(sample, dataset)} {verbalizer}:'
    return prompt

def run_in_context_learning_experiment(tokenizer, model, dataset, experiment_number):
    train_dataset, test_dataset = prepare_dataset(tokenizer, dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    task = get_task(dataset)
    if task == Task.Classification:
        labels = get_labels(dataset)
    id2label = {i : label for i, label in enumerate(labels)} if task == Task.Classification else None

    for nshots in [0, 1, 2, 4]:
        results = []
        for i in range(len(test_dataset)):
            print(f'experiment{experiment_number} {nshots}-shot sample: {i}')

            prompt = generate_prompt(test_dataset[i], train_dataset, id2label, dataset, nshots)
            tokens = tokenizer(prompt, return_tensors="pt")
            input_ids = tokens.input_ids.to(device)
            attention_mask = tokens.attention_mask.to(device)

            label = test_dataset[i]['text'].split(DELIMITER)[1] if task == Task.Generation else id2label[test_dataset[i]['label']] if id2label else test_dataset[i]['label']
            
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask, 
                                    do_sample=True, max_new_tokens=100, top_p=0.9)
            response = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            print(response)
            results.append([prompt, response, label])

        experiment_name = f'mistral_in_context_learning_{dataset.name}_{nshots}shots_experiment{experiment_number}'
        save_results(experiment_name, results, ["prompt", "response", "label"])

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
    seed = randint(1, 1000000000)
    torch.manual_seed(seed)

    tokenizer, model = get_model(MODEL_NAME, approach, dataset)

    if approach == Approach.FT:
        results = run_fine_tuning_experiment(tokenizer, model, dataset, i)
    elif approach == Approach.ICL:
        results = run_in_context_learning_experiment(tokenizer, model, dataset, i)
    else:
        raise ValueError(f"{approach.value} is not a valid learning approach!")

    model.to('cpu')
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
