#!/usr/bin/env python
# coding: utf-8

# # The Stack V2 sample extraction

# In[ ]:


# parallel processing
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=16)
from tqdm import tqdm
tqdm.pandas()

import pandas as pd

from collections import Counter
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

import numpy as np


# In[6]:


import os
os.environ["HF_HOME "] = "/home/hf_cache"


# In[7]:


# the random seed
seed = 42

#n.samples to be extracted from the final filtered samples
n_samples = 1000


# In[ ]:

# Keep in mind that to run the code you will need to download TheStackV2 Java subset: 
# https://huggingface.co/datasets/bigcode/the-stack-v2-dedup/viewer/Java 
# In the code we assume that this download operation is completed and the dataset is located in a certain location of the filesystem
dataset = load_from_disk("/home/hf_cache/stackv2_java_content")


# In[9]:


checkpoint = "bigcode/starcoder2-3b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_input_ptrain(batch):
    return tokenizer(batch['content'], truncation = True, max_length=4096, padding = 'do_not_pad')


# In[10]:


np.random.seed(42)

candidate_sample_size = 50000 
sample_check_size = 100000 

random_rows = np.random.choice(range(0, len(dataset)), size=candidate_sample_size + sample_check_size, replace=False)

candidate_ds = dataset.select(random_rows[0:candidate_sample_size])


# In[ ]:


# candidates
# I don't know why but after the tokenization I have a list of lists and I want to get rid of that
candidate_ds = candidate_ds.map(tokenize_input_ptrain)
df = candidate_ds.to_pandas()
# filter out the input with a number of tokens > 300
df['n_tok'] = df['input_ids'].apply(len)
df = df[df['n_tok'] > 300 ]

# save a copy for later
df_query = df.copy()

df_query['sample'] = df_query['input_ids'].progress_apply(lambda x: [x[i:i+300] for i in range(0, len(x)-299)])
df_query = df_query.explode('sample') # explode the list of lists
df_query['hash'] = df_query['sample'].progress_apply(lambda x: hash(tuple(x)))

# count the hashes (to retrieve the one uniques)    
hashes_ptrain = Counter(df_query['hash'])


# ### Counter check 300 token sequence

# In[ ]:


check_ds = dataset.select(random_rows[candidate_sample_size:])


# In[ ]:


bs = 10000
for i in tqdm(range(0,len(check_ds), bs)):
    df_batch = pd.DataFrame(check_ds[i:i+bs])
    df_batch['input_ids'] = df_batch['content'].apply(lambda x: tokenizer(x, truncation = True, max_length=4096, padding = 'do_not_pad')['input_ids']) 
    df_batch['n_tok'] = df_batch['input_ids'].apply(len)
    df_batch = df_batch[df_batch['n_tok'] > 300 ]
    
    df_batch['sample'] = df_batch['input_ids'].progress_apply(lambda x: [x[i:i+300] for i in range(0, len(x)-299)])
    df_batch = df_batch.explode('sample') # explode the list of lists
    df_batch['hash'] = df_batch['sample'].progress_apply(lambda x: hash(tuple(x)))
    for h in df_batch['hash']:
        if h in hashes_ptrain:
            hashes_ptrain[h] += 1
    


# Check if the pre-train samples are not contained into the ftune set

# In[15]:


#length of the tokenizer truncation used for the fine-tuning
max_length = 1024

def tokenize_input_ftune(batch):
    #return tokenizer(batch['content'], return_tensors='pt')
    return tokenizer(batch['content'], truncation = True, max_length= max_length, padding = 'do_not_pad')


# In[ ]:


tune_dataset = load_dataset("../java_train")
tune_dataset = tune_dataset.remove_columns(['input_ids', 'attention_mask', 'n_tok', 'longline', 'alpha', 'encoded', 'autogen'])
tune_dataset = tune_dataset.map(tokenize_input_ftune, batched=False, num_proc=64)


# In[17]:


# convert it to a pandas dataframe
tune_df = tune_dataset['train'].to_pandas()

# filter out the input with a number of tokens > 300
tune_df['n_tok'] = tune_df['input_ids'].apply(len)
tune_df = tune_df[tune_df['n_tok'] > 300 ]

# save a copy for later
tune_df_query = tune_df.copy()


# In[ ]:


tune_df_query['sample'] = tune_df_query['input_ids'].progress_apply(lambda x: [x[i:i+300] for i in range(0, len(x)-299)])
tune_df_query = tune_df_query.explode('sample') # explode the list of lists
tune_df_query['hash'] = tune_df_query['sample'].progress_apply(lambda x: hash(tuple(x)))

# count the hashes (to retrieve the one uniques)
hashes_ftune = Counter(tune_df_query['hash'])


# Now we can filter out:

# In[ ]:


# if the hash is 1 in the counter it means that it's unique. Zero as checked before in this case is not expected!
df_query['uniques_1'] = df_query['hash'].progress_apply(lambda x: (hashes_ptrain[x] == 1) & (hashes_ftune[x] == 0))
df_query['uniques_2'] = df_query['hash'].progress_apply(lambda x: (hashes_ptrain[x] == 2) & (hashes_ftune[x] == 0))
df_query['uniques_3'] = df_query['hash'].progress_apply(lambda x: (hashes_ptrain[x] == 3) & (hashes_ftune[x] == 0))
df_query['uniques_g3'] = df_query['hash'].progress_apply(lambda x: (hashes_ptrain[x] > 3) & (hashes_ftune[x] == 0))


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
hashes_fn = Counter(df_index['hash'])


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
    return hashes_fn[x['hash_sq']] == x['n']

# prefix duplicates
df_sample_g3['uniques'] = df_sample_g3.progress_apply(condition, axis = 1)

print(df_sample_g3[df_sample_g3['uniques']==True].shape)
df_sampled_g3 = df_sample_g3[df_sample_g3['uniques']==True].drop_duplicates(subset=['hash']).sample(n = n_samples, replace = False, random_state=seed)

df_sampled_g3['prefix_250'] = (df_sampled_g3['sample'].progress_apply(lambda x: tokenizer.decode(x[0:250], skip_special_tokens=False)))
df_sampled_g3['prefix_200'] = (df_sampled_g3['sample'].progress_apply(lambda x: tokenizer.decode(x[50:250], skip_special_tokens=False)))
df_sampled_g3['prefix_150'] = (df_sampled_g3['sample'].progress_apply(lambda x: tokenizer.decode(x[100:250], skip_special_tokens=False)))
df_sampled_g3['prefix_100'] = (df_sampled_g3['sample'].progress_apply(lambda x: tokenizer.decode(x[150:250], skip_special_tokens=False)))

df_sampled_g3['suffix'] = (df_sampled_g3['sample'].progress_apply(lambda x: tokenizer.decode(x[250:300], skip_special_tokens=False)))

df_sampled_g3.to_parquet('./pre-train_attack_g3.parquet', index = False)

