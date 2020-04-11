from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import scipy.sparse as sp
from tokenizer import Tokenizer
 

path1 = 'biorxiv.txt'
path2 = 'comm_use_subset.txt'
path3 = 'noncomm_use_subset.txt'
path4 = 'pmc_custom_license.txt'
corpus = []

with open(path1) as f:
    lines = f.readlines()
for line in lines:
    corpus.append(line.strip())

with open(path2) as f:
    lines = f.readlines()
for line in lines:
    corpus.append(line.strip())
	
with open(path3) as f:
    lines = f.readlines()
for line in lines:
    corpus.append(line.strip())
	
with open(path4) as f:
    lines = f.readlines()
for line in lines:
    corpus.append(line.strip())
print('total doc :',len(corpus))
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=10000,tokenizer=Tokenizer.tokenize)

# vectorizer = CountVectorizer(lowercase=True, stop_words=stop_word, max_features=10000)
X = vectorizer.fit_transform(corpus)
voc = vectorizer.vocabulary_
vectorizer = CountVectorizer(vocabulary=voc, tokenizer=Tokenizer.tokenize)
X = vectorizer.fit_transform(corpus)
voc = vectorizer.get_feature_names()

sp.save_npz('cord19_10000_tfidf.npz', X)
np.save('voc_10000_tfidf.npy',voc)
with open('voc_10000_tfidf.txt','w') as f:
    for word in voc:
	    f.write(word)
	    f.write('\n')
print('doneeeeee')

