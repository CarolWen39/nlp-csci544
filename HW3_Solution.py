#!/usr/bin/env python
# coding: utf-8

# # CSCI544-HW3


# ## Task 1

# In[11]:


import numpy as np
import pandas as pd
import collections
train = pd.read_csv('train', sep='\t', header=None)


# In[12]:


new_col = ['idx', 'word','tag']
train.columns = new_col
train.head()


# In[13]:


vocabfreq = collections.Counter(train['word'])
vf = vocabfreq.most_common()


# In[14]:


vfdf = pd.DataFrame(vf, columns=['word','count'])


# In[15]:


vfdf1 = vfdf[vfdf['count']>=2]
vfdfunk = vfdf[vfdf['count']<2]


# In[16]:


vfdfunk1 = pd.DataFrame({'word':'<unk>', 'count':vfdfunk['count'].sum()},index=[0])
vfdfunk1


# In[17]:


vocabfreq = vfdfunk1.append(vfdf1)
vocabfreq = vocabfreq.reset_index()
vocabfreq


# In[18]:


vocabfreq.to_csv(r'vocab.txt', header=None,index=None,sep='\t')


# In[19]:


# vocab contains all possible words in train, including <unk> and the words whose frequence is lower than threshold
vocab = {} 
for i in range(len(vocabfreq)):
    vocab[vocabfreq.iloc[i]['word']] = vocabfreq.iloc[i]['count']
vfdfunk = vfdfunk.reset_index()
for i in range(len(vfdfunk)):
    vocab[vfdfunk.iloc[i]['word']] = vfdfunk.iloc[i]['count']


# ## Task 2

# In[20]:


tag_count = train['tag'].value_counts()
tags = list(tag_count.index)

tag_dict = {}
for i in range(len(tag_count)):
    tag_dict[tags[i]] = tag_count[i]


# In[21]:


train_tag = {}
for k,v in tag_dict.items():
    train_tag[k] = {}

tmplst = list(train['tag'])
for i in range(len(tmplst)-1):
    cur_s = tmplst[i]
    nxt_s = tmplst[i+1]
    train_tag[cur_s][nxt_s] = train_tag[cur_s].get(nxt_s,0)+1


# In[22]:


transition = {}
transition_save = {}
for k,v in train_tag.items():
    count = tag_dict[k]
    for kk, vv in v.items():
        freq = vv
        transition[(k, kk)] = freq/count
        transition_save[str((k, kk))] = freq/count


# In[23]:


tag_word = {}
for key,val in tag_dict.items():
    tag_word[key] = dict()


# In[24]:


with open('train', 'r') as f:
    for l in f:
        try:
            word = l.split('\t')[1]
            tag = l.split('\t')[2].replace('\n', '')
            if vocab[word] < 2:
                tag_word[tag]['<unk>'] = tag_word[tag].get('<unk>', 0) + 1
            tag_word[tag][word] = tag_word[tag].get(word, 0) + 1
        except:
            continue


# In[25]:


emission = {}
emission_save = {}
for key, val in tag_word.items():
    tag_count = tag_dict[key]
    for kk, vv in val.items():
        emission[(kk, key)] = vv/tag_count
        emission_save[str((kk, key))] = vv/tag_count


# In[53]:


print("parameters number in transition: ", len(transition))
print("parameters number in emission: ", len(emission))


# There are 1378 transition parameters and 50314 and emission parameters in my HMM.

# In[26]:


import json
hmm = {}
hmm['transition'] = transition_save
hmm['emission'] = emission_save
with open('hmm.json','w') as file:
    json_str = json.dumps(hmm['transition'])
    file.write(json_str)
    file.write('\n')
    json_str = json.dumps(hmm['emission'])
    file.write(json_str)


# ## Task 3

# In[27]:


# development data
dev_sentences = []
sent = []
with open('dev', 'r') as f:
    for l in f:
        l = l.replace('\n', '').split('\t')
        if l[0] == '':
            dev_sentences.append(sent)
            sent = []
            continue
        sent.append(l)


# In[50]:


# get the sentences in test file
test_sentences = []
test_sent = []
with open('test', 'r') as f:
    for l in f:
        l = l.replace('\n', '').split('\t')
        if l[0] == '':
            test_sentences.append(test_sent)
            test_sent = []
            continue
        test_sent.append(l)


# In[29]:


# calculate the numbers of sentences that each tag begins with
begin_tag = dict()
number_sen = 0
with open('train', 'r') as f:
    for line in f:
        if line.split('\t')[0] == '1':
            number_sen += 1
            tag = line.split('\t')[2].replace('\n', '')
            begin_tag[tag] = begin_tag.get(tag, 0) + 1


# In[30]:


# calculates the ts1 of each tag
tag_ts1 = {}
for key, val in begin_tag.items():
    tag_ts1[key] = val / number_sen


# In[32]:


# get transition(si|si-1), given the previous tag and get the probability of the different following tags
def get_transition(previous_tag):
    possible_tags = []
    for key, val in transition.items():
        if key[0] == previous_tag:
            possible_tags.append((key[1], val))
    return possible_tags

possible_tags = get_transition('NNP')
possible_tags 


# In[34]:


# get the all emission(xi|si)
def get_emission(word):
    possible_words = []
    if word not in vocab:
        for key, val in emission.items():
            if key[0] == '<unk>':
                possible_words.append((key[1], val))
    else:
        for key, val in emission.items():
            if key[0] == word:
                possible_words.append((key[1], val))
    return possible_words
possible_words = get_emission('like')
possible_words


# In[35]:


# calculates s*1=argmax(t(s1)e(x1|s1))
def get_s1(possible_words):
    argmaxscore = 0.0
    for tag, prob in possible_words:
        if tag not in begin_tag:
            continue
        curscore = prob*tag_ts1[tag]
        if curscore > argmaxscore:
            argmaxscore = curscore
            s1 = tag
    return s1


# In[36]:


# calculates s*i= argmax(t(si|si-1)e(xi|si))
def get_si(possible_words, possible_tags):
    argmaxscore = -1.0
    for k1, v1 in possible_words:
        for k2, v2 in possible_tags:
            if k1 == k2:
                curscore = v1 * v2
                if curscore > argmaxscore:
                    argmaxscore = curscore
                    si = k1
    return si


# In[41]:


def greedy(sentence):
    pred = []
    for item in sentence:
        index = item[0]
        word = item[1]
        # deal with the beginning word
        if index == '1':
            begin_word = word
            possible_words = get_emission(begin_word)
            s1 = get_s1(possible_words)
            # store the current tag into previous tag
            previous_tag = s1
            pred.append(s1)
            continue
        
        # deal with the following words
        possible_wordsi = get_emission(word)
        possible_tags = get_transition(previous_tag)
        try:
            si = get_si(possible_wordsi, possible_tags)
        # if the get_possible_wordsi is null then will go except
        except UnboundLocalError:
            #print(possible_wordsi)
            si = sorted(possible_wordsi, key=lambda p: p[1], reverse=True)[0][0] #find the largest possbility
        previous_tag = si
        
        pred.append(si)
    return pred


# In[39]:


from tqdm import tqdm
pred = []
l = []
for sentence in tqdm(dev_sentences):
    #print(sentence)
    predictions = greedy(sentence)
    pred = pred + predictions
    labels = [s[2] for s in sentence]
    l = l + labels
# print the accuracy of dev data
print("the accuracy of greedy algorithm on development data:")
print(sum(p[0] == p[1] for p in zip(pred, l))/len(pred))


# The accuracy of greedy algorithm on development data is 94.16%

# In[52]:


# write the output to greedy.out
with open('greedy.out', 'w') as f:
    for s in tqdm(test_sentences):
        pred = greedy(s)
        for i, item in enumerate(s):
            idx = item[0]
            word = item[1]
            p = pred[i]
            line = str(idx) + '\t' + word + '\t' + p
            f.write(line)
            f.write('\n')
        f.write('\n')


# ## Task 4

# In[44]:


def initialization(possible_words):
    for tag, prob in possible_words:
        if tag not in begin_tag:
            continue
        dp[0][tag] = prob*begin_tag[tag]


# In[45]:


# forward set the dp array
def viterbi(sentence, dp):
    for item in sentence:
        index = int(item[0]) - 1
        word = item[1]
        # deal with the first word
        if index == 0:
            # all possible tags given the beginning word of the sentence
            possible_words = get_emission(word)
            # initialize the tags of the first word in the dp array
            initialization(possible_words)
            continue
            
        # all potential tags from second words
        possible_wordsi = get_emission(word)
        for tag, probe in possible_wordsi: # probe is e(x|si)
            argmax1 = 0 # the best s1
            best_si_1_for_si = None
            for previous_tag, d in dp[index-1].items():
                if index == 1:
                    s1 = dp[index-1][previous_tag]
                else:
                    s1 = d['cur_prob']
                try:
                    t_si_si_1 = transition[(previous_tag, tag)]
                except:
                    t_si_si_1 = 0
                if argmax1 < s1 * t_si_si_1 * probe:
                    argmax1 = s1 * t_si_si_1 * probe
                    best_si_1_for_si = previous_tag
            dp[index][tag] = {'cur_prob':argmax1, 'previous_tag': best_si_1_for_si}
        tt = 0
        for key,val in dp[index].items():
            tt += val['cur_prob']
        # deal with unvalid current prossibilty
        if tt==0:
            tag = sorted(possible_wordsi, key=lambda p: p[1], reverse=True)[0][0]
            best_scorei = 0
            for key_tag, prob in dp[index-1].items():
                cur_scorei = prob['cur_prob']
                if cur_scorei > best_scorei:
                    best_si_1_for_si = key_tag
                    best_scorei = cur_scorei
            dp[index][tag] = {'cur_prob':best_scorei, 'previous_tag': best_si_1_for_si}
    return dp


# In[66]:


# backtrack to find the set of tags with the highest score
def findbesttag(dp, index, previous, prediction):
    if index == 0: 
        # there is only one word in the sentence
        if len(dp) == 1:
            # sorted by probability to find the largest possible tag
            x = sorted(dp[0].items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
            prediction.append(x[0][0])
            return prediction
    
        else:
            prediction.append(previous)
            return prediction[::-1]
    
    elif index == len(dp)-1 and previous is None:
        final_score = 0
        for tag, prob in viterbi_dp[index].items():
            cur_score = prob['cur_prob']
            if cur_score > final_score:
                final_score = cur_score
                best_tag = tag
                best_previous = prob['previous_tag']
        cur_previous = best_previous
        prediction.append(best_tag)
        
    else:
        for tag, prob in dp[index].items():
            if tag == previous:
                prediction.append(tag)
                cur_previous = prob['previous_tag']
    return findbesttag(dp, index-1, cur_previous, prediction)


# In[68]:


# test on development data
pred = []
l = []
for sentence in tqdm(dev_sentences):
    dp = [{} for _ in range(len(sentence))]
    viterbi_dp = viterbi(sentence, dp)
    predictions = findbesttag(viterbi_dp, len(viterbi_dp)-1 , None, [])
    pred = pred + predictions
    labels = [ss[2] for ss in sentence]
    l = l + labels
# print the accuracy of dev data
print("the accuracy of viterbi algorithm on development data:")
print(sum(p[0] == p[1] for p in zip(pred, l))/len(pred))


# The accuracy of viterbi algorithm on development data is 95.29%

# In[ ]:


# write the output to viterbi.out
prediction = []
with open('viterbi.out', 'w') as f:
    for sentence in tqdm(test_sentences):
        dp = [{} for _ in range(len(sentence))]
        viterbi_dp = viterbi(sentence, dp)
        predictions = findbesttag(viterbi_dp, len(viterbi_dp)-1 , None, [])
        #print(predictions)
        for i, item in enumerate(sentence):
            index = item[0]
            word = item[1]
            pred = predictions[i]
            line = str(index) + '\t' + word + '\t' + pred
            f.write(line)
            f.write('\n')
        f.write('\n')

