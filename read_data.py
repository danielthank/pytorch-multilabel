import pandas
import re
import logging
import os
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

class Corpus:
    def __init__(self, path, val=0.1):
        train_csv = pandas.read_csv(os.path.join(path, 'train_out.csv'))
        test_csv = pandas.read_csv(os.path.join(path, 'test_out.csv'))

        train_texts = train_csv['text'].tolist()
        test_texts = test_csv['text'].tolist()

        train_val_idx = int(len(train_texts) * (1-val))

        tfidf_vectorizer = TfidfVectorizer(min_df=1)
        tfidf = tfidf_vectorizer.fit_transform(train_texts + test_texts)
        tfidf_array = tfidf.toarray().astype('float32')

        self.train = torch.from_numpy(tfidf_array[:train_val_idx])
        self.val = torch.from_numpy(tfidf_array[train_val_idx:len(train_texts)])
        self.test = torch.from_numpy(tfidf_array[-len(test_texts):])

        count_vectorizer = CountVectorizer(min_df=1, lowercase=False, tokenizer=lambda x: x.split(' '))
        count = count_vectorizer.fit_transform(train_csv['tags'].tolist())
        count_array = count.toarray().astype('float32')

        self.train_targets = torch.from_numpy(count_array[:train_val_idx])
        self.val_targets = torch.from_numpy(count_array[train_val_idx:len(train_texts)])
        self.tags = count_vectorizer.get_feature_names()
        

if __name__ == '__main__':
    corpus = Corpus('data/')
    print(corpus.train)
    print(corpus.val)
    print(corpus.test)
    print(corpus.train_targets)
    print(corpus.val_targets)
    print(corpus.tags)
