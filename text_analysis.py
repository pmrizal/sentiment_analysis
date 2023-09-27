import pandas as pd
import numpy as np
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import json
import torch
import torch.nn.functional as F

from torch import nn
from transformers import AutoModel
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemover, ArrayDictionary
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from collections import Counter

kamus_normalisasi = pd.read_csv("slang.csv")
kata_normalisasi_dict = {}

with open("config.json") as json_file:
    config = json.load(json_file)

def filtering_text(text):
    # mengubah tweet menjadi huruf kecil
    text = text.lower()
    # menghilangkan url
    text = re.sub(r'https?:\/\/\S+','',text)
    # menghilangkan mention, link, hastag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    #menghilangkan karakter byte (b')
    text = re.sub(r'(b\'{1,2})',"", text)
    # menghilangkan yang bukan huruf
    text = re.sub('[^a-zA-Z]', ' ', text)
    # menghilangkan digit angka
    text = re.sub(r'\d+', '', text)
    #menghilangkan tanda baca
    text = text.translate(str.maketrans("","",string.punctuation))
    # menghilangkan whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def stop_stem(text):
    #stopword
    with open('kamus.txt') as kamus:
        word = kamus.readlines()
        list_stopword = [line.replace('\n',"") for line in word]
    dictionary = ArrayDictionary(list_stopword)
    stopword = StopWordRemover(dictionary)
    text = stopword.remove(text)
    # stemming
    factory_stemmer = StemmerFactory()
    stemmer = factory_stemmer.create_stemmer()
    text = stemmer.stem(text)
    return text

def word_tokenize_wrapper(text):
    return word_tokenize(text)

for index, row in kamus_normalisasi.iterrows():
    if row[0] not in kata_normalisasi_dict:
        kata_normalisasi_dict[row[0]] = row[1]  

def normalisasi_kata(document):
    return [kata_normalisasi_dict[term] if term in kata_normalisasi_dict else term for term in document]

def count_words(df, column='normalisasi', preprocess=None, min_freq=2):
    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(tokens)
    counter = Counter()
    df[column].map(update)
    freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['freq'])
    freq_df = freq_df.query('freq >= @min_freq')
    freq_df.index.name = 'token'
    return freq_df.sort_values('freq', ascending=False)

def wordcloud(word_freq, title=None, max_words=200, stopwords=None):
    wc = WordCloud(width=800, height=400,
                   background_color= "black", colormap="Paired",
                   max_font_size=150, max_words=max_words)
    if type(word_freq) == pd.Series:
        counter = Counter(word_freq.fillna(0).to_dict())
    else:
        counter = word_freq
    if stopwords is not None:
        counter = {token:freq for (token, freq) in counter.items()
                   if token not in stopwords}
    wc.generate_from_frequencies(counter)
    plt.title(title)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")

def compute_idf(df, column='normalisasi', preprocess=None, min_df=2):
    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(set(tokens))
    counter = Counter()
    df[column].map(update)
    idf_df = pd.DataFrame.from_dict(counter, orient='index', columns=['df'])
    idf_df = idf_df.query('df >= @min_df')
    idf_df['idf'] = np.log(len(df)/idf_df['df'])+0.1
    idf_df.index.name = 'token'
    return idf_df

def ngrams(tokens, n=2, sep=' ', stopwords=set()):
    return [sep.join(ngram) for ngram in zip(*[tokens[i:] for i in range(n)])
            if len([t for t in ngram if t in stopwords])==0]

def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('indobenchmark/indobert-base-p1', return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(768, n_classes)
  
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

class Model:    
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

        classifier = SentimentClassifier(len(config['CLASS_NAMES']))
        classifier.load_state_dict(
            torch.load(config['PRE_TRAINED_MODEL'], map_location=self.device)
        )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, text):
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=config['MAX_SEQUENCE_LEN'],
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)

        with torch.no_grad():
            probabilities = F.softmax(self.classifier(input_ids, attention_mask), dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            predicted_class = predicted_class.cpu().item()
            probabilities = probabilities.flatten().cpu().numpy().tolist()
        return (
            config['CLASS_NAMES'][predicted_class]
            #confidence,
            #dict(zip(config['CLASS_NAMES'], probabilities)),
        )