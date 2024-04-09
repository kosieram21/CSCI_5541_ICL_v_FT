from datasets import DatasetDict, load_dataset

dataset = load_dataset("your_dataset_name")

train_test_split = dataset["train"].train_test_split(test_size=0.25)
dataset = load_dataset("your_dataset_name")
train_valid_split = dataset["train"].train_test_split(test_size=0.1)

split_dataset = DatasetDict({
    'train': train_valid_split['train'],
    'test': dataset['test'],
    'validation': train_valid_split['test']
})