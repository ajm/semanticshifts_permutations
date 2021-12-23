#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import random
import string
import math
import pickle
import itertools
from collections import Counter, defaultdict

import torch
from transformers import *

import numpy as np

import scipy.interpolate.interpnd
from nltk.tokenize import sent_tokenize
from scipy.spatial.distance import cosine as distance
from scipy import stats


# In[2]:


bad_chars = set(string.punctuation) - set("_")

def clean(s) :
    global bad_chars

    #s = s.replace('_nn','').replace('_vb','')

    if "'" in s :
        return "" # ignore contractions
    if "’" in s :
        return "" # ignore contractions
    #if "-" in s :
    #    return "" # ignore hyphenated words
    if s != 'f5' :
        for n in string.digits :
            if n in s :
                return "" # ignore anything with a number
    return ''.join([ c if c not in bad_chars else '' for c in s ])

# returns [(string,[w0,w1,...wN]),(string,[w0,w1,...wN]),...]
def read_data(fname) :
    sentences = []
    count = 0
    rejected = 0
    stop = False
    print("reading {} ...".format(fname))
    with open(fname, encoding="utf8") as f :
        for line in f :
            line = line.strip()
            if not line : continue
            #for sent in sent_tokenize(line) :
            if 1 :
                sent = line
                words = sent.split()
                #if len(words) < 5 :
                #    rejected += 1
                #    continue
                words = [ clean(w) for w in words ]
                words = set([ w for w in words if (w != '') ])
                sentences.append((sent,words))
                count += 1
    print("  - read {} sentences (rejected {})".format(count, rejected))
    return sentences


# In[3]:


def process_sentences(corpus, tokenizer, word_list) :
    for text,words in corpus :
        if not any([ w in words for w in word_list]) :
            continue

        text = text.replace('_nn', '').replace('_vb', '')

        tokens = tokenizer.tokenize("[CLS] " + text + " [SEP]")
        tokens = [ t for t in tokens if t != '[UNK]' ] # emoji are [UNK]
        if len(tokens) > 512 :
            continue
        yield tokens,tokenizer.convert_tokens_to_ids(tokens)


# In[4]:


def find_tokens2(tokens, words_list) :
    tmp = []
    word = ""
    offset = 1 # we need to see a word without '##', so offset is always at least 1

    for i in range(len(tokens)) :
        current = tokens[i]
        if current.startswith("##") :
            word += current.replace("##", "")
            offset += 1
        else :
            if word and (word in words_list) :
                tmp.append((i-offset, word))
            offset = 1
            word = current

    if word and (word in words_list) :
        tmp.append((i-offset, word))

    return tmp


# In[5]:


def mk_batch(tokens_list) :
    mx = max([ len(t) for t in tokens_list ])
    tokens = np.array([ t + ([0] * (mx-len(t))) for t in tokens_list ])
    mask = np.where(tokens != 0, 1, 0)
    return torch.LongTensor(tokens), torch.LongTensor(mask)


# In[6]:


def process_batch(token_batch, ids_batch, model, tokenizer, words_counters) :
    tokens,segments = mk_batch(ids_batch)
    tokens = tokens.to("cuda")      # gpu
    segments = segments.to("cuda")  # gpu

    with torch.no_grad() :
        _, _, hidden_states = model(tokens, segments)

    hidden_states = torch.stack(hidden_states, dim=0)
    hidden_states = hidden_states.permute(1,2,0,3)

    embeddings = []
    # sum the last 4 hidden layers
    # to extract sub-word embeddings
    for index,sentence in enumerate(hidden_states) :
        for i,w in  find_tokens2(token_batch[index], words_counters) :
            wt = tokenizer.tokenize(w)

            sum_vec = torch.sum(sentence[i][-4:], dim=0)

            for j in range(1, len(wt)) :
                sum_vec += torch.sum(sentence[i+j][-4:], dim=0)

            sum_vec /= len(wt)
            sum_vec = sum_vec.cpu().numpy()

            embeddings.append((w, sum_vec))
            words_counters[w] -= 1

    return embeddings


# In[7]:


def avg_embeddings(emb) :
    return np.sum(emb, axis=0) / len(emb)


# In[8]:


def get_wordcount(sentences) :
    c = Counter()
    for sentence in sentences :
        text,words = sentence
        c.update(words)
    return c


# In[9]:


def get_wordlist(fname) : #wc1, wc2) :
    #return list(set(wc1).intersection(set(wc2)))
    tmp = []
    with open(fname) as f :
        f.readline() # header
        for line in f :
            line = line.strip().split(',')[1]
            tmp.append(line)
    return tmp

# In[10]:


def extract_embeddings(sentences, word_list, max_word, emb_fname, model, tokenizer, max_batch) :
    random.shuffle(sentences)
    words_counters = dict([ (w.split('_')[0], max_word) for w in word_list ])
    token_batch = []
    ids_batch = []
    embeddings = []
    count = 0
    total_words = len(word_list)
    
    for tokens,ids in process_sentences(sentences, tokenizer, word_list) :
        token_batch.append(tokens)
        ids_batch.append(ids)
        
        if len(token_batch) < max_batch :
            continue
        
        embeddings.extend(process_batch(token_batch, ids_batch, model, tokenizer, words_counters))
        count += 1
        token_batch = []
        ids_batch = []
        
        to_delete = []
        for w in words_counters :
            if words_counters[w] <= 0 :
                to_delete.append(w)
        for w in to_delete :
            del words_counters[w]
        print("\rbatch {} - progress={}/{}".format(count, len(words_counters), total_words), end="", flush=True)
        
    if len(token_batch) > 0 :
        embeddings.extend(process_batch(token_batch, ids_batch, model, tokenizer, words_counters))
    
    print("done!")

    print("\nsyncing to disk...")
    with open(emb_fname, 'wb') as f :
        pickle.dump(embeddings, f)
        
    print("done!")

    return embeddings

# In[11]:


def load_embeddings(fname) :
    with open(fname, 'rb') as f :
        tmp = pickle.load(f)
        
    tmp2 = defaultdict(list)
    for word,vector in tmp :
        tmp2[word].append(np.array(vector))
        
    print("read {} embeddings for {} words ({})".format(len(tmp), len(tmp2), fname))
    
    for word,vectors in tmp2.items() :
        tmp2[word] = np.array(vectors)
        
    return tmp2


# In[12]:


def num_combinations(n, r) :
    try :
        return math.factorial(n) / (math.factorial(r) * math.factorial(n - r))
    except :
        return 1e6


# In[13]:


def exact_pval(g1, g2, dist) :
    top_half = len(g1)
    g = np.concatenate((g1, g2), axis=0)
    tmp = []

    for ordering in itertools.combinations(range(len(g)), top_half) :
        perm1 = np.take(g, ordering, axis=0)
        perm2 = np.delete(g, ordering, axis=0)
        tmp.append(distance(avg_embeddings(perm1), avg_embeddings(perm2))) 
    pval = sum([ 1 for i in tmp if i >= dist ]) / len(tmp)
    return pval

def permutation_test(g1, g2, dist) :
    top_half = len(g1)
    g = np.concatenate((g1, g2), axis=0)
    tmp = []
    
    if num_combinations(len(g), top_half) < 1000 :
        return dist,exact_pval(g1,g2,dist)

    else :
        for _ in range(1000) :
            np.random.shuffle(g)
            perm1 = g[: top_half , :]
            perm2 = g[top_half : , :]
            tmp.append(distance(avg_embeddings(perm1), avg_embeddings(perm2)))

        pval = sum([ 1 for i in tmp if i >= dist ]) / len(tmp)
        if pval > 0.05 :
            return dist,pval

        if num_combinations(len(g), top_half) < 10000 :
            return dist,exact_pval(g1,g2,dist) 

        for _ in range(9000) :
            np.random.shuffle(g)
            perm1 = g[: top_half , :]
            perm2 = g[top_half : , :]
            tmp.append(distance(avg_embeddings(perm1), avg_embeddings(perm2)))

        pval = sum([ 1 for i in tmp if i >= dist ]) / len(tmp)
        if pval > 0.005 :
            return dist,pval

        #return dist,pval # XXX 

        if num_combinations(len(g), top_half) < 100000 :
            return dist,exact_pval(g1,g2,dist)

        for _ in range(90000) :
            np.random.shuffle(g)
            perm1 = g[: top_half , :]
            perm2 = g[top_half : , :]
            tmp.append(distance(avg_embeddings(perm1), avg_embeddings(perm2)))

        pval = sum([ 1 for i in tmp if i >= dist ]) / len(tmp)
        if pval == 0.0 :
            pval = 1 / 100000.0
        return dist,pval


# In[ ]:


def calc_stats(lfc13, lfc17, wordlist, fname) :                                                                                                                                              
    count = 0
    data = {}
    
    print("calculating statistics ...")
    #wordlist = get_wordlist(lfc13, lfc17)
    total_words = len(wordlist)
    
    #wordlist = ['thong']
    
    for idx,w in enumerate(wordlist) :
        print("  - {} ({} / {})          ".format(w, idx, total_words))
        avg13 = avg_embeddings(lfc13[w])
        avg17 = avg_embeddings(lfc17[w])
        avg_dist = distance(avg13, avg17)
        
        permutation_dist, permutation_pval = permutation_test(lfc13[w], lfc17[w], avg_dist)
        
        data[w] = (permutation_dist, permutation_pval)
        count += 1
        
    print("")
    print("  - calculated {} distances".format(count))
    print("  - writing {} ...".format(fname))
    
    with open(fname, 'w', encoding='utf-8') as f :
        print("word freq1 freq2 dist ppval", file=f)
        for w in data :
            p_dist, p_pval = [ str(i) for i in data[w] ]
            n_13, n_17 = str(len(lfc13[w])), str(len(lfc17[w]))
            print(" ".join([ w, n_13, n_17, p_dist, p_pval ]), file=f)


# In[16]:


if len(sys.argv) != 6 :
    print("python {} <bert model dir> <fname1> <fname2> <word list> <results>\n", file=sys.stderr)
    sys.exit(1)

bert_model = sys.argv[1] # "shift_simulation_6_5epoch"

fname1 = sys.argv[2] # 'shift_simulation_6a.txt'
fname2 = sys.argv[3] # 'shift_simulation_6b.txt'

wordlist_fname = sys.argv[4]

results_fname = sys.argv[5] # 'shift_simulation_6_results.txt'

# pickle files need to go _somewhere_, _anywhere_ in scratch...
emb_fname1 = os.path.join(os.path.dirname(results_fname), os.path.basename(fname1).replace('.txt','.pkl'))
emb_fname2 = os.path.join(os.path.dirname(results_fname), os.path.basename(fname2).replace('.txt','.pkl'))


# In[17]:


tokenizer = BertTokenizer.from_pretrained(bert_model)
config = BertConfig.from_pretrained(bert_model, output_hidden_states=True)
model = BertModel.from_pretrained(bert_model, config=config)

model.eval()
model.to('cuda') # GPU
print("BERT loaded")


# In[18]:


# read every sentence
sentences1 = read_data(fname1)
sentences2 = read_data(fname2)
# make word counts for each corpus
wordcount1 = get_wordcount(sentences1)
wordcount2 = get_wordcount(sentences2)
# make list of words in both corpuses
#words = get_wordlist(wordcount1, wordcount2)
words = get_wordlist(wordlist_fname)
for w in words :
    print(w, wordcount1.get(w,0), wordcount2.get(w,0))

print("overview\n  - {} words found in both corpuses".format(len(words)))


# In[19]:


max_word = 1000000
max_batch = 50

emb1 = extract_embeddings(sentences1, words, max_word, emb_fname1, model, tokenizer, max_batch)
emb2 = extract_embeddings(sentences2, words, max_word, emb_fname2, model, tokenizer, max_batch)


#exit(0)
# In[ ]:


emb1 = load_embeddings(emb_fname1)
emb2 = load_embeddings(emb_fname2)


# In[70]:

words = get_wordlist(wordlist_fname)
calc_stats(emb1, emb2, words, results_fname)
print("done!")

