import argparse
import torch
import numpy as np
import pandas as pd
import random
import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Callback
import tensorflow as tf
from datasets import load_metric
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformer import ClassificationDataset as transformerDataset
from transformers import Trainer, TrainingArguments
from transformers import set_seed
from transformers import DataCollatorWithPadding

# reproductilibty
set_seed(1)
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
# GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Global variable
# 3 number of labels
NUM_LABELS = 3
TF_MAX_SIZE = 512
LF_MAX_SIZE = 4096

metric1 = load_metric("precision")
metric2 = load_metric("recall")
metric3 = load_metric("f1")
metric_name = "f_macro"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(classification_report(labels, predictions, digits=4))
    precision = metric1.compute(predictions=predictions, references=labels, average='weighted')["precision"]
    recall = metric2.compute(predictions=predictions, references=labels, average='weighted')["recall"]
    f_score = metric3.compute(predictions=predictions, references=labels, average='weighted')["f1"]
    f_macro = metric3.compute(predictions=predictions, references=labels, average='macro')["f1"]
    return {"precision": precision, "recall": recall, "f_score": f_score, "f_macro": f_macro}


def ray_hp_space(trial):
    return {'weight_decay': tune.uniform(0.0, 0.3),
            'per_device_train_batch_size': tune.choice([8, 16]),
            'learning_rate': tune.uniform(1e-5, 3e-6)}


def search_hyperparameter_model(transformer_model, train_df, dev_df, save_path):
    def model_init(trial):
        return AutoModelForSequenceClassification.from_pretrained(transformer_model, max_length=TF_MAX_SIZE,
                                                                  num_labels=NUM_LABELS)

    tokenizer = AutoTokenizer.from_pretrained(transformer_model, max_length=TF_MAX_SIZE, padding="max_length",
                                              truncation=True, num_labels=NUM_LABELS)

    tokenized_train_dataset = tokenizer(train_df['text'].tolist(), truncation=True, padding=True)
    train_dataset = transformerDataset(tokenized_train_dataset, train_df._label.tolist())

    tokenized_eval_dataset = tokenizer(dev_df['text'].tolist(), truncation=True, padding=True)
    eval_dataset = transformerDataset(tokenized_eval_dataset, dev_df._label.tolist())
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    batch_train_size = 16
    batch_eval_size = 16
    training_args = TrainingArguments(
        output_dir="./longformer",
        # overwrite_output_dir=True,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 1,
        per_device_train_batch_size=batch_train_size,
        per_device_eval_batch_size=batch_eval_size,
        metric_for_best_model=metric_name,
        weight_decay=0.01,
        learning_rate=1e-5,
    )
    trainer = Trainer(
        model_init=model_init,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        hp_space=ray_hp_space,
        n_trials=10
    )

    print(best_trial)


def train_model(transformer_model, train_df, dev_df, save_path):

    tokenizer = AutoTokenizer.from_pretrained(transformer_model, max_length=TF_MAX_SIZE, padding="max_length",
                                              truncation=True, num_labels=NUM_LABELS)
    model = AutoModelForSequenceClassification.from_pretrained(transformer_model, max_length=TF_MAX_SIZE, num_labels=NUM_LABELS)

    tokenized_train_dataset = tokenizer(train_df['text'].tolist(), truncation=True, padding=True)
    train_dataset = transformerDataset(tokenized_train_dataset, train_df._label.tolist())

    tokenized_eval_dataset = tokenizer(dev_df['text'].tolist(), truncation=True, padding=True)
    eval_dataset = transformerDataset(tokenized_eval_dataset, dev_df._label.tolist())

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Use the best hyperparameter after the search
    batch_train_size = 8
    batch_eval_size = 16
    training_args = TrainingArguments(
        output_dir="./deberta",
        # overwrite_output_dir=True,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 1,
        per_device_train_batch_size=batch_train_size,
        per_device_eval_batch_size=batch_eval_size,
        metric_for_best_model=metric_name,
        weight_decay=0.01,
        learning_rate=1e-5,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # Salvamos el modelo reentrenado
    model.save_pretrained(f'{save_path}/robertaCombiClassification')
    tokenizer.save_pretrained(f'{save_path}/robertaCombiClassification')


def main(args):
    label_id = {'phrase': 0, 'passage': 1, 'multi': 2}
    train_path = args.train_path
    dev_path = args.dev_path
    save_path = args.save_path

    # Utilizamos Deberta
    # transformer_model = "microsoft/deberta-large"
    transformer_model = "roberta-large"
    # transformer_model = "microsoft/deberta-v2-large"
    
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    
    train_df['_label'] = train_df['tag'].apply(lambda x: label_id[x])
    dev_df['_label'] = dev_df['tag'].apply(lambda x: label_id[x])

    train_model(transformer_model, train_df, dev_df, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./dataset/train_combi.csv')
    parser.add_argument('--dev_path', type=str, default='./dataset/dev_combi_model.csv')
    parser.add_argument('--save_path', type=str, default='./results')
    args = parser.parse_args()
    main(args)
