import json
import re
import argparse
import jsonlines
import pandas as pd
from cleantext import clean
from sentence_transformers import SentenceTransformer, util
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
url2 = '(www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def clean_text(tweet):
    # Contractions
    tweet = re.sub(r"he's", "he is", tweet)
    tweet = re.sub(r"there's", "there is", tweet)
    tweet = re.sub(r"We're", "We are", tweet)
    tweet = re.sub(r"That's", "That is", tweet)
    tweet = re.sub(r"won't", "will not", tweet)
    tweet = re.sub(r"they're", "they are", tweet)
    tweet = re.sub(r"Can't", "Cannot", tweet)
    tweet = re.sub(r"wasn't", "was not", tweet)
    tweet = re.sub(r"aren't", "are not", tweet)
    tweet = re.sub(r"isn't", "is not", tweet)
    tweet = re.sub(r"What's", "What is", tweet)
    tweet = re.sub(r"haven't", "have not", tweet)
    tweet = re.sub(r"hasn't", "has not", tweet)
    tweet = re.sub(r"There's", "There is", tweet)
    tweet = re.sub(r"He's", "He is", tweet)
    tweet = re.sub(r"It's", "It is", tweet)
    tweet = re.sub(r"You're", "You are", tweet)
    tweet = re.sub(r"I'M", "I am", tweet)
    tweet = re.sub(r"shouldn't", "should not", tweet)
    tweet = re.sub(r"wouldn't", "would not", tweet)
    tweet = re.sub(r"i'm", "I am", tweet)
    tweet = re.sub(r"I'm", "I am", tweet)
    tweet = re.sub(r"Isn't", "is not", tweet)
    tweet = re.sub(r"Here's", "Here is", tweet)
    tweet = re.sub(r"you've", "you have", tweet)
    tweet = re.sub(r"we're", "we are", tweet)
    tweet = re.sub(r"what's", "what is", tweet)
    tweet = re.sub(r"couldn't", "could not", tweet)
    tweet = re.sub(r"we've", "we have", tweet)
    tweet = re.sub(r"who's", "who is", tweet)
    tweet = re.sub(r"y'all", "you all", tweet)
    tweet = re.sub(r"would've", "would have", tweet)
    tweet = re.sub(r"it'll", "it will", tweet)
    tweet = re.sub(r"we'll", "we will", tweet)
    tweet = re.sub(r"wouldn't", "would not", tweet)
    tweet = re.sub(r"We've", "We have", tweet)
    tweet = re.sub(r"he'll", "he will", tweet)
    tweet = re.sub(r"Y'all", "You all", tweet)
    tweet = re.sub(r"Weren't", "Were not", tweet)
    tweet = re.sub(r"Didn't", "Did not", tweet)
    tweet = re.sub(r"they'll", "they will", tweet)
    tweet = re.sub(r"they'd", "they would", tweet)
    tweet = re.sub(r"DON'T", "DO NOT", tweet)
    tweet = re.sub(r"That's", "That is", tweet)
    tweet = re.sub(r"they've", "they have", tweet)
    tweet = re.sub(r"i'd", "I would", tweet)
    tweet = re.sub(r"should've", "should have", tweet)
    tweet = re.sub(r"where's", "where is", tweet)
    tweet = re.sub(r"we'd", "we would", tweet)
    tweet = re.sub(r"i'll", "I will", tweet)
    tweet = re.sub(r"weren't", "were not", tweet)
    tweet = re.sub(r"They're", "They are", tweet)
    tweet = re.sub(r"Can't", "Cannot", tweet)
    tweet = re.sub(r"you'll", "you will", tweet)
    tweet = re.sub(r"I'd", "I would", tweet)
    tweet = re.sub(r"let's", "let us", tweet)
    tweet = re.sub(r"it's", "it is", tweet)
    tweet = re.sub(r"can't", "cannot", tweet)
    tweet = re.sub(r"don't", "do not", tweet)
    tweet = re.sub(r"you're", "you are", tweet)
    tweet = re.sub(r"i've", "I have", tweet)
    tweet = re.sub(r"that's", "that is", tweet)
    tweet = re.sub(r"i'll", "I will", tweet)
    tweet = re.sub(r"doesn't", "does not", tweet)
    tweet = re.sub(r"i'd", "I would", tweet)
    tweet = re.sub(r"didn't", "did not", tweet)
    tweet = re.sub(r"ain't", "am not", tweet)
    tweet = re.sub(r"you'll", "you will", tweet)
    tweet = re.sub(r"I've", "I have", tweet)
    tweet = re.sub(r"Don't", "do not", tweet)
    tweet = re.sub(r"I'll", "I will", tweet)
    tweet = re.sub(r"I'd", "I would", tweet)
    tweet = re.sub(r"Let's", "Let us", tweet)
    tweet = re.sub(r"you'd", "You would", tweet)
    tweet = re.sub(r"It's", "It is", tweet)
    tweet = re.sub(r"Ain't", "am not", tweet)
    tweet = re.sub(r"Haven't", "Have not", tweet)
    tweet = re.sub(r"Could've", "Could have", tweet)
    tweet = re.sub(r"youve", "you have", tweet)

    tweet = re.sub(r"w/e", "whatever", tweet)
    tweet = re.sub(r"w/", "with", tweet)

    tweet = re.sub(r"lmao", "laughing my ass off", tweet)

    tweet = re.sub(r"@[A-za-z0-9]*", "@[USER] ", tweet)
    tweet = re.sub(r"#[A-za-z0-9]*", "#[HASHTAG] ", tweet)

    patron = re.compile(url + '|' + url2)
    tweet = patron.sub('', tweet)

    tweet = clean(tweet, no_emoji=True, lower=False, no_punct=False)

    return tweet


def get_split(summarized_text):
    l_total = []
    if len(summarized_text.split()) // 150 > 0:
        n = len(summarized_text.split()) // 150
    else:
        n = 1
    for w in range(n):
        if w == 0:
            l_partial = summarized_text.split()[:200]
            l_total.append(" ".join(l_partial))
        else:
            l_partial = summarized_text.split()[w*150:w*150 + 200]
            l_total.append(" ".join(l_partial))
    return l_total


def find_paragraph(title, paragraph):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    # Encode query and documents
    query_emb = model.encode(title)
    doc_emb = model.encode(paragraph)
    # Compute dot score between query and all document embeddings
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    # Combine docs & scores
    doc_score_pairs = list(zip(paragraph, scores))
    # Sort by decreasing score
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    return doc_score_pairs[0][0]


def create_tweet_paragraph_csv(path, file_name):
    result_df = pd.DataFrame(columns=['uuid', 'text', 'tag'])
    with jsonlines.open(path) as reader:
        for obj in reader:
            paragraphs = obj['targetParagraphs']
            for idx, p in enumerate(paragraphs):
                # Check the size of paragraph
                if len(p.split()) > 200:
                    key_paragraph = find_paragraph(obj['postText'], get_split(p))
                    paragraphs[idx] = key_paragraph

            key_paragraph = find_paragraph(obj['postText'], paragraphs)
            key_paragraph = clean_text(key_paragraph)

            row = {'uuid': obj['uuid'], 'text': key_paragraph, 'tag': obj['tags'][0]}
            result_df = result_df.append(row, ignore_index=True)

    result_df.to_csv(file_name, index=False)


def create_tweet_title_csv(path, file_name):
    result_df = pd.DataFrame(columns=['uuid', 'text', 'tag'])
    with jsonlines.open(path) as reader:
        for obj in reader:
            text = obj['postText'][0] + '-' + obj['targetTitle']
            text = clean_text(text)
            row = {'uuid': obj['uuid'], 'text': text}
            result_df = result_df.append(row, ignore_index=True)
    result_df.to_csv(file_name, index=False)


def main(args):
    train_path = args.train_path
    dev_path = args.dev_path
    test_path = args.test_path
    save_path = args.save_path

    train_csv = f'{save_path}/train_paragraph.csv'
    dev_csv = f'{save_path}/dev_paragraph.csv'
    test_csv = f'{save_path}/test_postText_title.csv'

    #create_tweet_paragraph_csv(train_path, train_csv)
    #create_tweet_paragraph_csv(dev_path, dev_csv)
    create_tweet_title_csv(test_path, test_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../webis-clickbait-22/train.jsonl')
    parser.add_argument('--dev_path', type=str, default='../webis-clickbait-22/validation.jsonl')
    parser.add_argument('--test_path', type=str, default='../webis-clickbait-22/input.jsonl')
    parser.add_argument('--save_path', type=str, default="./dataset")
    args = parser.parse_args()
    main(args)