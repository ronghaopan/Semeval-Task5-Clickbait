import pandas as pd
import argparse
import torch
import numpy as np
import pandas as pd
import random
import json
from datasets import Dataset
import tensorflow as tf
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformer import ClassificationTestDataset as get_test_dataset

# reproductilibty
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
# GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_LABELS = 3
TF_MAX_SIZE = 512

id_label = {0: 'phrase', 1: 'passage', 2: 'multi'}
label_id = {'phrase': 0, 'passage': 1, 'multi': 2}


def test_model_max(transformer_model, paragraph_model, test_df, save_path):
    tokenizer = AutoTokenizer.from_pretrained(transformer_model, max_length=TF_MAX_SIZE, padding="max_length",
                                              truncation=True, num_labels=NUM_LABELS)
    model = AutoModelForSequenceClassification.from_pretrained(transformer_model, max_length=TF_MAX_SIZE,
                                                               num_labels=NUM_LABELS)

    tokenizer_paragraph = AutoTokenizer.from_pretrained(paragraph_model, max_length=TF_MAX_SIZE, padding="max_length",
                                              truncation=True, num_labels=NUM_LABELS)
    model_paragraph = AutoModelForSequenceClassification.from_pretrained(paragraph_model, max_length=TF_MAX_SIZE,
                                                               num_labels=NUM_LABELS)

    tokenized_eval_dataset = tokenizer(test_df['text_title'].tolist(), truncation=True, padding=True)
    eval_dataset = get_test_dataset(tokenized_eval_dataset)

    tokenized_paragraph_dataset = tokenizer_paragraph(test_df['text_paragraph'].tolist(), truncation=True, padding=True)
    paragraph_dataset = get_test_dataset(tokenized_paragraph_dataset)

    result = []
    predictions = []
    with torch.no_grad():
        for idx in range(0, len(test_df)):
            print(idx)
            input_ids = eval_dataset.__getitem__(idx)["input_ids"].unsqueeze(0)
            attention_mask = eval_dataset.__getitem__(idx)["attention_mask"].unsqueeze(0)

            input_paragraph_ids = paragraph_dataset.__getitem__(idx)['input_ids'].unsqueeze(0)
            attention_mask_paragraph = paragraph_dataset.__getitem__(idx)["attention_mask"].unsqueeze(0)

            outputs_title = model(input_ids, attention_mask=attention_mask).logits
            outputs_paragraph = model_paragraph(input_paragraph_ids, attention_mask_paragraph).logits

            sigmoid = torch.nn.Sigmoid()
            probs_title = sigmoid(torch.Tensor(outputs_title))
            probs_paragraph = sigmoid(torch.Tensor(outputs_paragraph))

            probs_title = torch.mul(probs_title, 0.5)
            probs_paragraph = torch.mul(probs_paragraph, 0.5)

            value_title, preds_title = torch.max(probs_title, dim=1)
            value_pg, preds_pg = torch.max(probs_paragraph, dim=1)

            if int(preds_title) == int(preds_pg):
                preds = int(preds_title)
                predictions.append(int(preds_title))
            elif int(preds_title) != int(preds_pg):
                if float(value_title) > float(value_pg):
                    preds = int(preds_title)
                    predictions.append(int(preds_title))
                elif float(value_pg) > float(value_title):
                    preds = int(preds_pg)
                    predictions.append(int(preds_pg))

            row = {'uuid': test_df.iloc[idx]['uuid'], 'spoilerType': id_label[preds]}
            result.append(row)

    print(classification_report(test_df['_label'].values, predictions, digits=5))
    with open(f'{save_path}/output_dev_combi_max.jsonl', 'w') as outfile:
        for entry in result:
            json.dump(entry, outfile)
            outfile.write('\n')


def test_model_mean(transformer_model, paragraph_model, test_df, save_path):
    tokenizer = AutoTokenizer.from_pretrained(transformer_model, max_length=TF_MAX_SIZE, padding="max_length",
                                              truncation=True, num_labels=NUM_LABELS)
    model = AutoModelForSequenceClassification.from_pretrained(transformer_model, max_length=TF_MAX_SIZE,
                                                               num_labels=NUM_LABELS)

    tokenizer_paragraph = AutoTokenizer.from_pretrained(paragraph_model, max_length=TF_MAX_SIZE, padding="max_length",
                                              truncation=True, num_labels=NUM_LABELS)
    model_paragraph = AutoModelForSequenceClassification.from_pretrained(paragraph_model, max_length=TF_MAX_SIZE,
                                                               num_labels=NUM_LABELS)

    tokenized_eval_dataset = tokenizer(test_df['text_title'].tolist(), truncation=True, padding=True)
    eval_dataset = get_test_dataset(tokenized_eval_dataset)

    tokenized_paragraph_dataset = tokenizer_paragraph(test_df['text_paragraph'].tolist(), truncation=True, padding=True)
    paragraph_dataset = get_test_dataset(tokenized_paragraph_dataset)

    result = []
    predictions = []
    with torch.no_grad():
        for idx in range(0, len(test_df)):
            print(idx)
            input_ids = eval_dataset.__getitem__(idx)["input_ids"].unsqueeze(0)
            attention_mask = eval_dataset.__getitem__(idx)["attention_mask"].unsqueeze(0)

            input_paragraph_ids = paragraph_dataset.__getitem__(idx)['input_ids'].unsqueeze(0)
            attention_mask_paragraph = paragraph_dataset.__getitem__(idx)["attention_mask"].unsqueeze(0)

            outputs_title = model(input_ids, attention_mask=attention_mask).logits
            outputs_paragraph = model_paragraph(input_paragraph_ids, attention_mask_paragraph).logits

            outputs = torch.mean(torch.stack((outputs_title, outputs_paragraph), 1), 1)
            values, preds = torch.max(outputs, dim=1)
            predictions.append(int(preds))

            row = {'uuid': test_df.iloc[idx]['uuid'], 'spoilerType': id_label[int(preds)]}
            result.append(row)

    print(classification_report(test_df['_label'].values, predictions, digits=5))

    with open(f'{save_path}/output_dev_combi_mean.jsonl', 'w') as outfile:
        for entry in result:
            json.dump(entry, outfile)
            outfile.write('\n')


def main(args):
    dev_path = args.dev_path
    save_path = args.save_path

    dev_df = pd.read_csv(dev_path)
    dev_df['_label'] = dev_df['tag'].apply(lambda x: label_id[x])

    transformer_model = "./debertaV2TitleClassification"
    paragraph_model = "./robertaParagraphClassification"

    test_model_mean(transformer_model, paragraph_model, dev_df, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_path', type=str, default='./dataset/dev_combi.csv')
    parser.add_argument('--save_path', type=str, default='./results')
    args = parser.parse_args()
    main(args)
