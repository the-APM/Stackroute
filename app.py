import os
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from gensim import models
from gensim.models.doc2vec import TaggedDocument
import matplotlib.pyplot as plt

### for fables use this
path = "./data/Literature/Fables/"
out_file = "./figures/fables.png"

### for shakespeare use this
##path = "./data/Literature/Shakespeare/"
##out_file = "./figures/shakespeare.png"

def read(path):
    all_files = os.listdir(path)
    raw_files = dict()
    for f in all_files:
        raw_files[f] = open(path+f).read()
    return raw_files

def preprocess(raw_files):
    preprocessed_files = dict()
    stopwords_en = stopwords.words("english")
    stopwords_en.extend([".",",","''","'",";",":","?"])
    lemmatizer = WordNetLemmatizer()
    for f in raw_files:
        wordlist = nltk.word_tokenize(raw_files[f])
        text = [w.lower() for w in wordlist if w.lower() not in stopwords_en]
        pos_tagged = pos_tag(text)
        preprocessed_files[f] = []
        for word, tag in pos_tagged:
            tag = tag[0].lower()
            if tag in ["a","r","n","v"]:
                preprocessed_files[f].append(lemmatizer.lemmatize(word, tag))
            else:
                preprocessed_files[f].append(word)
    return preprocessed_files

def vectorize(preprocessed_files):
    tagged_files = []
    for f in preprocessed_files:
        doc = TaggedDocument(words=preprocessed_files[f], tags =f)
        tagged_files.append(doc)
    model = models.Doc2Vec(dm=1, min_count=1, alpha=0.025, min_alpha=0.025)
    model.build_vocab(tagged_files)
    model.train(tagged_files, epochs=50, total_examples=model.corpus_count)
    return model

def plot_heatmap(preprocessed_files, model):
    files = list(preprocessed_files.keys())
    hm = np.random.random((len(files),len(files)))
    for f1 in range(len(files)):
        for f2 in range(len(files)):
            if f1 == f2:
                hm[f1][f1] = 0
            else:
                hm[f1][f2] = model.n_similarity(preprocessed_files[files[f1]],preprocessed_files[files[f2]])
    fig, ax = plt.subplots()
    im = ax.imshow(hm)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("similarity index", rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(files)))
    ax.set_yticks(np.arange(len(files)))
    ax.set_xticklabels(files)
    ax.set_yticklabels(files)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    ax.set_title("Similarity heatmap")
    plt.tight_layout()
    plt.savefig(out_file)
    return files, hm

def recommend(files, hm):
    max_index = hm.argmax(axis=1)
    print("Recommended next read:-")
    for i in range(len(files)):
        print(files[i]+" -> "+files[max_index[i]])
    return 0

if __name__ == "__main__":
    raw_files = read(path)
    preprocessed_files = preprocess(raw_files)
    model = vectorize(preprocessed_files)
    files, hm = plot_heatmap(preprocessed_files, model)
    recommend(files, hm)
    
