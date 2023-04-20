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


def test_model(transformer_model, test_df, save_path):
    tokenizer = AutoTokenizer.from_pretrained(transformer_model, max_length=TF_MAX_SIZE, padding="max_length",
                                              truncation=True, num_labels=NUM_LABELS)
    model = AutoModelForSequenceClassification.from_pretrained(transformer_model, max_length=TF_MAX_SIZE,
                                                               num_labels=NUM_LABELS)

    tokenized_eval_dataset = tokenizer(test_df['text'].tolist(), truncation=True, padding=True)
    eval_dataset = get_test_dataset(tokenized_eval_dataset)

    result = []
    predictions = []
    with torch.no_grad():
        for idx in range(0, len(test_df)):
            input_ids = eval_dataset.__getitem__(idx)["input_ids"].unsqueeze(0)
            attention_mask = eval_dataset.__getitem__(idx)["attention_mask"].unsqueeze(0)

            outputs = model(input_ids, attention_mask=attention_mask).logits

            _, preds = torch.max(outputs, dim=1)
            predictions.append(int(preds))
            print("predecido: ", int(preds))
            row = {'uuid': test_df.iloc[idx]['uuid'], 'spoilerType': id_label[int(preds)]}
            result.append(row)

    print(classification_report(test_df['_label'].values, predictions, digits=5))

    with open(f'{save_path}/output_dev_combi_input_roberta.jsonl', 'w') as outfile:
        for entry in result:
            json.dump(entry, outfile)
            outfile.write('\n')


def main(args):
    dev_path = args.dev_path
    save_path = args.save_path

    dev_df = pd.read_csv(dev_path)
    dev_df['_label'] = dev_df['tag'].apply(lambda x: label_id[x])

    transformer_model = "./robertaCombiClassification"

    test_model(transformer_model, dev_df, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_path', type=str, default='./dataset/dev_combi_model.csv')
    parser.add_argument('--save_path', type=str, default='./results')
    args = parser.parse_args()
    main(args)
