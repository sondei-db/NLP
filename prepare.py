import csv
import re
from gensim.models import Word2Vec
import gensim.downloader as api
import numpy as np
from spellchecker import SpellChecker
import pandas as pd

folder = 'dev-0'
filename = f"{folder}/in_1.csv"

data = []

data = pd.read_csv(f'{folder}/in.tsv',delimiter='\t', header=None, encoding='utf-8', quoting=csv.QUOTE_NONE, engine='python').values.tolist()

data_a = []
data_b = []
data_pair = []

for i in range(len(data)):
    data_a.append(data[i][6])
    try:
        data_b.append(data[i][7])
    except:
        data_b.append('')

for i in range(len(data)):
    data_pair.append([data_a[i], data_b[i]])

data_tabs = []

for x, y in data_pair:
    cleaned_text_a = x.replace('\\t', '\t').replace('\\n', '\n').strip("[]")   
    cleaned_text_b = y.replace('\\t', '\t').replace('\\n', '\n').strip("[]")
    data_tabs.append([cleaned_text_a, cleaned_text_b])

data_removed = []

for x, y in data_tabs:
    text = re.sub(r'(?<!-)\n', ' ', x)
    text = re.sub(r'[\n-]', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text_2 = re.sub(r'(?<!-)\n', ' ', y)
    text_2 = re.sub(r'[\n-]', '', text_2)
    text_2 = re.sub(r'[^a-zA-Z0-9\s]', '', text_2)
    text_2 = re.sub(r'\s+', ' ', text_2)
    data_removed.append([text, text_2])

model = api.load("word2vec-google-news-300")

def is_close_to_actual(word, threshold=0.5):
    if word in model:
        similarities = model.similar_by_word(word)
        return any(similarity > threshold for _, similarity in similarities)
    else:
        return False

def remove_words(text, words_to_destroy):
    pattern = r'\b(?:{})\b'.format('|'.join(words_to_destroy))
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return cleaned_text

spell = SpellChecker()

data_cleared = []

i = 0
for x, y in data_removed:

    words = x.split()
    words_2 = y.split()

    misspelled = spell.unknown(words + words_2)

    text = remove_words(x, list(misspelled))
    text_2 = remove_words(y, list(misspelled))

    data_cleared.append([text, text_2])

    if i % 20000 == 0:
        print(f'{i/430000*100}%')
    i += 1

data_cleared_2 = []

for x, y in data_cleared:
    text = re.sub(r'(?<!-)\n', ' ', x)
    text = re.sub(r'[\n-]', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text_2 = re.sub(r'(?<!-)\n', ' ', y)
    text_2 = re.sub(r'[\n-]', '', text_2)
    text_2 = re.sub(r'[^a-zA-Z0-9\s]', '', text_2)
    text_2 = re.sub(r'\s+', ' ', text_2)
    data_cleared_2.append([text, text_2])

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_cleared_2)
