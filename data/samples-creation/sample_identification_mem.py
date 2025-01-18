#!/usr/bin/env python
# coding: utf-8

# # F-tune sampling
# 
# - we control the number of times the sample was seen
# 
# - additionally we are not considering prefix duplicates. The prefix deduplication was done for being able to do not have false negatives. Which means that for the same prefix we have different suffixes but in our sample set might be the case that we are not having that exact suffix (one of many for a single prefix) and we erroneously classify a sample as non memorized when actually it is.

# In[ ]:


import torch 
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, TextDataset, pipeline

# parallel processing
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=16)
from tqdm import tqdm
tqdm.pandas()

# utility
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os


# In[2]:


from collections import Counter


# In[11]:


# the random seed
seed = 42

#length of the tokenizer truncation
max_length = 1024

#n.samples to be extracted from the final filtered samples
n_samples = 1000


# In[10]:


# Load the tokenizer from HF-hub
checkpoint = "bigcode/starcoder2-3b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_input_ftune(batch):
    #return tokenizer(batch['content'], return_tensors='pt')
    return tokenizer(batch['content'], truncation = True, max_length= max_length, padding = 'do_not_pad',  return_tensors='pt')


# # Create the fine-tune samples for the attack

# In[8]:


# load the data
dataset = load_dataset("AISE-TUDelft/memtune-tuning_data", name = "20k", split = 'train' )


# In[12]:


# remove non-necessery columns form the sample filtering
dataset = dataset.remove_columns(['input_ids', 'attention_mask', 'n_tok', 'longline', 'alpha', 'encoded', 'autogen'])


# In[16]:


# apply the tokenizer
dataset = dataset.map(tokenize_input_ftune, batched=False, num_proc=64)

# convert it to a pandas dataframe
df = dataset['train'].to_pandas()
# I don't know why but after the tokenization I have a list of lists and I want to get rid of that
df['input_ids'] = df['input_ids'].apply(lambda x: x[0])

# filter out the input with a number of tokens > 300
df['n_tok'] = df['input_ids'].apply(len)
df = df[df['n_tok'] > 300 ]

# save a copy for later
df_query = df.copy()


# In[ ]:


df_query['sample'] = df_query['input_ids'].progress_apply(lambda x: [x[i:i+300] for i in range(0, len(x)-299)])
df_query = df_query.explode('sample') # explode the list of lists
df_query['hash'] = df_query['sample'].progress_apply(lambda x: hash(tuple(x)))

# count the hashes (to retrieve the one uniques)
hashes_ptrain = Counter(df_query['hash'])


# In[ ]:


# if the hash is 1 in the counter it means that it's unique. Zero as checked before in this case is not expected!
df_query['uniques_1'] = df_query['hash'].progress_apply(lambda x: hashes_ptrain[x] == 1 )
df_query['uniques_2'] = df_query['hash'].progress_apply(lambda x: hashes_ptrain[x] == 2 )
df_query['uniques_3'] = df_query['hash'].progress_apply(lambda x: hashes_ptrain[x] == 3 )
df_query['uniques_g3'] = df_query['hash'].progress_apply(lambda x: hashes_ptrain[x] > 3 )


# Now I have to check if the prefix is unique, in such a way we will avoid false negatives!

# In[ ]:


# extract the first 100 tokens of the prefix, that needs to be checked (250 to 300 suffix) (250 to 150) prefix to be checked
df_query['sample_query'] = df_query['sample'].progress_apply(lambda x: x[150:250])

# create the index
df_index = df.copy()

# generate a sliding window of 100 tokens from the training corpus
df_index['patterns'] = df_index['input_ids'].progress_apply(lambda x: [x[i:i+100] for i in range(0, len(x)-99)])
df_index = df_index.explode('patterns')

# hash everything
df_query['hash_sq'] = df_query['sample_query'].progress_apply(lambda x: hash(tuple(x)))
df_index['hash'] = df_index['patterns'].progress_apply(lambda x: hash(tuple(x)))

# count the hashes (to retrieve the one uniques)
hashes_ptrain = Counter(df_index['hash'])


# Now that we can control the duplicated prefixes, we can perform the sampling considering different duplication sets

# # No duplicates == 1
# 
# - candidate samples -> 10060778
# - deduplicated prefix -> 9734999

# In[ ]:


df_sample_1 = df_query[df_query['uniques_1']==True]
print(df_sample_1.shape)
# prefix check
df_sample_1['uniques'] = df_sample_1['hash_sq'].progress_apply(lambda x: hashes_ptrain[x] == 1 )
print(df_sample_1[df_sample_1['uniques']==True].shape)
df_sampled_1 = df_sample_1[df_sample_1['uniques']==True].sample(n = n_samples, replace = False, random_state=seed)

df_sampled_1['prefix_250'] = (df_sampled_1['sample'].progress_apply(lambda x: tokenizer.decode(x[0:250], skip_special_tokens=False)))
df_sampled_1['prefix_200'] = (df_sampled_1['sample'].progress_apply(lambda x: tokenizer.decode(x[50:250], skip_special_tokens=False)))
df_sampled_1['prefix_150'] = (df_sampled_1['sample'].progress_apply(lambda x: tokenizer.decode(x[100:250], skip_special_tokens=False)))
df_sampled_1['prefix_100'] = (df_sampled_1['sample'].progress_apply(lambda x: tokenizer.decode(x[150:250], skip_special_tokens=False)))

df_sampled_1['suffix'] = (df_sampled_1['sample'].progress_apply(lambda x: tokenizer.decode(x[250:300], skip_special_tokens=False)))

df_sampled_1.to_parquet('./fine-tune_attack_1.parquet', index = False)


# ## duplicates == 2; 
# sample_2 -> 67056
# 
# sample_2 deduplicated prefix -> **47940**
# 

# In[ ]:


df_sample_2 = df_query[df_query['uniques_2']==True]
print(df_sample_2.shape)
# prefix check
df_sample_2['uniques'] = df_sample_2['hash_sq'].progress_apply(lambda x: hashes_ptrain[x] == 2 )

print(df_sample_2[df_sample_2['uniques']==True].shape)

df_sampled_2 = df_sample_2[df_sample_2['uniques']==True].drop_duplicates(subset=['hash']).sample(n = n_samples, replace = False, random_state=seed)

df_sampled_2['prefix_250'] = (df_sampled_2['sample'].progress_apply(lambda x: tokenizer.decode(x[0:250], skip_special_tokens=False)))
df_sampled_2['prefix_200'] = (df_sampled_2['sample'].progress_apply(lambda x: tokenizer.decode(x[50:250], skip_special_tokens=False)))
df_sampled_2['prefix_150'] = (df_sampled_2['sample'].progress_apply(lambda x: tokenizer.decode(x[100:250], skip_special_tokens=False)))
df_sampled_2['prefix_100'] = (df_sampled_2['sample'].progress_apply(lambda x: tokenizer.decode(x[150:250], skip_special_tokens=False)))

df_sampled_2['suffix'] = (df_sampled_2['sample'].progress_apply(lambda x: tokenizer.decode(x[250:300], skip_special_tokens=False)))

df_sampled_2.to_parquet('./fine-tune_attack_2.parquet', index = False)


# ## duplicates == 3
# 
# - sample_3 -> 19209
# 
# - sample_3 deduplicated prefix -> 10083

# In[ ]:


df_sample_3 = df_query[df_query['uniques_3']==True]
print(df_sample_3.shape)
# prefix check
df_sample_3['uniques'] = df_sample_3['hash_sq'].progress_apply(lambda x: hashes_ptrain[x] == 3 )
print(df_sample_3[df_sample_3['uniques']==True].shape)
df_sampled_3 = df_sample_3[df_sample_3['uniques']==True].drop_duplicates(subset=['hash']).sample(n = n_samples, replace = False, random_state=seed)

df_sampled_3['prefix_250'] = (df_sampled_3['sample'].progress_apply(lambda x: tokenizer.decode(x[0:250], skip_special_tokens=False)))
df_sampled_3['prefix_200'] = (df_sampled_3['sample'].progress_apply(lambda x: tokenizer.decode(x[50:250], skip_special_tokens=False)))
df_sampled_3['prefix_150'] = (df_sampled_3['sample'].progress_apply(lambda x: tokenizer.decode(x[100:250], skip_special_tokens=False)))
df_sampled_3['prefix_100'] = (df_sampled_3['sample'].progress_apply(lambda x: tokenizer.decode(x[150:250], skip_special_tokens=False)))

df_sampled_3['suffix'] = (df_sampled_3['sample'].progress_apply(lambda x: tokenizer.decode(x[250:300], skip_special_tokens=False)))

df_sampled_3.to_parquet('./fine-tune_attack_3.parquet', index = False)


# ## duplicates > 3
# 
# The grater than 3 is a bit different, because we need to make sure that the hash in the counter matches the n. of duplicates.
# 
# - df_sample_g3 -> 115439
# 
# - df_sample_g3 prefix deduplicates -> 38605

# In[ ]:


df_sample_g3 = df_query[df_query['uniques_g3']==True]
print(df_sample_g3.shape)
# we count the number of duplicates >3 but we want the actual number!!!
df_sample_g3['n'] = df_sample_g3.groupby(by='hash')['hash'].transform('count')

def condition(x):
    return hashes_ptrain[x['hash_sq']] == x['n']

# prefix duplicates
df_sample_g3['uniques'] = df_sample_g3.progress_apply(condition, axis = 1)

print(df_sample_g3[df_sample_g3['uniques']==True].shape)
df_sampled_g3 = df_sample_g3[df_sample_g3['uniques']==True].drop_duplicates(subset=['hash']).sample(n = n_samples, replace = False, random_state=seed)

df_sampled_g3['prefix_250'] = (df_sampled_g3['sample'].progress_apply(lambda x: tokenizer.decode(x[0:250], skip_special_tokens=False)))
df_sampled_g3['prefix_200'] = (df_sampled_g3['sample'].progress_apply(lambda x: tokenizer.decode(x[50:250], skip_special_tokens=False)))
df_sampled_g3['prefix_150'] = (df_sampled_g3['sample'].progress_apply(lambda x: tokenizer.decode(x[100:250], skip_special_tokens=False)))
df_sampled_g3['prefix_100'] = (df_sampled_g3['sample'].progress_apply(lambda x: tokenizer.decode(x[150:250], skip_special_tokens=False)))

df_sampled_g3['suffix'] = (df_sampled_g3['sample'].progress_apply(lambda x: tokenizer.decode(x[250:300], skip_special_tokens=False)))

df_sampled_g3.to_parquet('./fine-tune_attack_g3.parquet', index = False)

