#!/usr/bin/env python
# coding: utf-8

# # <center>CSCI544 Homework1 Report</center>
# <h1 style="text-align:right">Mengsha Wen</h1>

# In[1]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup


# # 1. Data Preparation

# ## Read Data & Keep Reviews and Ratings

# In[2]:


df = pd.read_csv('amazon_reviews_us_Kitchen_v1_00.tsv',
                 sep='\t',
                 usecols=['star_rating','review_body'])


# ### Samples of Reviews
# The following are three samples of reviews along with corresponding ratings.

# In[3]:


#print(df.iloc[0]['star_rating'],' ', df.iloc[0]['review_body'])
#print(df.iloc[464650]['star_rating'],' ', df.iloc[464650]['review_body'])
#print(df.iloc[949]['star_rating'],' ', df.iloc[949]['review_body'])


# ### Statistics of Reviews
# There are total 4,875,088 reviews in this dataset.

# In[4]:


df.shape[0]


# After dropping the rows with missing values (eg., Nan), that leaves 4,874,842 items.

# In[5]:


#df[df['star_rating'].isnull()]
df = df.dropna()
df.shape[0]


# According to the analysis, the statistics of the ratings is shown in figure 1 below. There are 3,124,740 reviews received 5 rating; 731,718 reviews received 4 stars; 349,552 reviews are 3 stars; and 241,945 reviews and 426,887 reviews received 2 ratings and 1 ratings, respectively.

# In[6]:


df['star_rating'].value_counts()


# # Labelling Reviews:
# The reviews with rating 4,5 are labelled to be 1 and 1,2 are labelled as 0. Discard the reviews with rating 3'

# In[7]:


# discard rating 3
df1 = df[~df['star_rating'].isin([3])]
df1.shape[0]


# In[8]:


# labelling
df1['label'] = df1.star_rating.apply(lambda x: 1 if(x>=4) else 0)
df1['label'].value_counts()


# After creating binary labels, there are 3,856,458 items labeled to 1, and 668,832 items labeled to 0. Besides, there are 349,552 reviews with the rating 3 which will be discarded.

# In[9]:

print("\n**The number of reviews for each of the three classes:")
print("number of Label 0:",df1['label'].value_counts()[0])
print("number of Label 1:",df1['label'].value_counts()[1])
print("number of rating 3:",df['star_rating'].value_counts()[3])


# After doing all of things mentioned above, the dataset is like as following

# In[10]:


data = df1[['review_body','label']]
data.iloc[-5:]


# I select 200000 reviews randomly with 100,000 positive and 100,000 negative reviews by selecting respectively and then shuffling.

# In[11]:


data_pos = (data[data['label'] == 1]).sample(n=100000, random_state = 1)
data_neg = (data[data['label'] == 0]).sample(n=100000, random_state = 1)


# In[12]:


from sklearn.utils import shuffle
# merge two sub dataset and shuffle
new_data = shuffle(pd.concat([data_pos, data_neg]))
new_data.head()


# ### train-test split
# After randomly selecting 100,000 positive reviews and 100,000 negative reviews from dataset and splitting training and testing dataset, there are 160,000 items in training set and 40,000 in testing set.

# In[13]:


from sklearn.model_selection import train_test_split
label = new_data['label']
reviews = new_data.drop('label',axis=1)
X_train, X_test, y_train, y_test = train_test_split(reviews, label, random_state=42, test_size=0.2)
#print(len(X_train), len(X_test), len(y_train), len(y_test))


# # 2. Data Cleaning

# The average length of the reviews in terms of character length in the dataset before data cleaning is shown below.The length is around 324

# In[14]:


#print("The average length of the reviews in terms of character"+
     # " length before data cleaning: ",new_data["review_body"].apply(len).mean())


# 2.1 Perform contractions on the reviews.

# In[15]:


# using build-in library to deal with contractions on the reviews
import contractions
def contractionfunction(s):
    words = []
    for word in s.split():
        words.append(contractions.fix(word))
    new_str = ' '.join(words)
    return new_str


# 2.2 Convert the all reviews into the lower case.
# 
# 2.3 Remove the HTML and URLs from the reviews.
# 
# 2.4 Remove non-alphabetical characters.
# 
# 2.5 Remove the extra tab, Line break, spaces, etc. between the words.

# The following are examples of cleaning X_train and X_test dataset, respectively.

# In[16]:


import re
# dealing with X_train
reviews_train = []
for r in X_train.review_body:
    # contraction
    r = contractionfunction(r)
    # change all letters into lower case
    r = r.lower()
    # dealing with html
    r = re.sub(r'</?\w+[^>]*>',' ',r)
    # dealing with urls
    r = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
               '', r)
    # dealing with non-alphabetical characters
    r = re.sub("[^a-z]+",' ',r)
    # dealing with \n, \b, \r, \t
    r = re.sub(r'\r|\n|\t','',r)
    # dealing with ,/./:/extra spaces....
    r = re.sub(r'[^\w\s]','',r)
    # perform contractions on the reviews
    reviews_train.append(r) 
reviews_train[0]


# In[17]:


# dealing with X_test
reviews_test = []
for r in X_test.review_body:
    #contraction
    r = contractionfunction(r)
    # change all letters into lower case
    r = r.lower()
    # dealing with html
    r = re.sub(r'</?\w+[^>]*>',' ',r)
    # dealing with urls
    r = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
               '', r)
    # dealing with non-alphabetical characters
    r = re.sub("[^a-z]+",' ',r)
    # dealing with \n, \b, \r, \t
    r = re.sub(r'\r|\n|\t','',r)
    # dealing with ,/./:/extra spaces....
    r = re.sub(r'[^\w\s]','',r)
    # perform contractions on the reviews
    reviews_test.append(r) 
reviews_test[0]


# The average length of the reviews in terms of character length in the dataset after data cleaning is shown below.

# In[18]:


df_train = pd.DataFrame(reviews_train, columns=['reviews'])
df_test = pd.DataFrame(reviews_test, columns=['reviews'])

avg_len1 = (df_train["reviews"].apply(len).mean()*160000
           + df_test["reviews"].apply(len).mean()*40000) / 200000

print("**The average length of the reviews in terms of character"
      +" length before and after data cleaning: ",new_data["review_body"].apply(len).mean(), ", ", avg_len1)


# # 3. Pre-processing

# ### 3.1 remove the stop words 

# In[19]:


from nltk.corpus import stopwords

def rm_stopwords(review):
    stop_words = set(stopwords.words('english'))
    words = [w for w in review.split(' ') if w not in stop_words]
    s = ' '.join(words)
    return s


# In[20]:


# remove the stop words in train and test dataset respectively
reviews_train_rmstopwords = []
reviews_test_rmstopwords = []
for review in reviews_train:
    reviews_train_rmstopwords.append(rm_stopwords(review))
for review in reviews_test:
    reviews_test_rmstopwords.append(rm_stopwords(review))


# ### 3.2 perform lemmatization  

# In[21]:


from nltk.stem import WordNetLemmatizer

def per_lemmatize(review):
    lm = WordNetLemmatizer()
    words = [lm.lemmatize(w) for w in review.split(' ')]
    s = ' '.join(words)
    return s


# In[22]:


reviews_train_lemmatize = []
reviews_test_lemmatize = []
for review in reviews_train_rmstopwords:
    reviews_train_lemmatize.append(per_lemmatize(review))
for review in reviews_test_rmstopwords:
    reviews_test_lemmatize.append(per_lemmatize(review))


# Below are three sample reviews before data cleaning and preprocessin

# In[23]:


print("**Three sample reviews before data cleaning and preprocessing:\n")
print(X_train.iloc[0].review_body,'\n')
print(X_train.iloc[799].review_body,'\n')
print(X_train.iloc[109485].review_body,'\n')


# In[24]:


# three samples after revoming stop words
print("**Three sample reviews after data cleaning and preprocessing:\n")
print(reviews_train_lemmatize[0],'\n')
print(reviews_train_lemmatize[799],'\n')
print(reviews_train_lemmatize[109485],'\n')


# Above are three samples after data cleaning and pre-processing.

# In[25]:


df_train = pd.DataFrame(reviews_train_lemmatize, columns=['reviews'])
df_test = pd.DataFrame(reviews_train_lemmatize, columns=['reviews'])

avg_len = (df_train["reviews"].apply(len).mean()*160000 + 
           df_test["reviews"].apply(len).mean()*40000) / 200000
print("**The average length of the reviews in terms of character"+
        " length before and after data preprocessing: ",avg_len1, ", ", avg_len, "\n")


# # 4. TF-IDF Feature Extraction

# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))
x_train_new = vectorizer.fit_transform(reviews_train_lemmatize)
x_test_new = vectorizer.transform(reviews_test_lemmatize)


# In[27]:


# deal with the label, convert Series to ndarry type
y_train_new = y_train.values
y_test_new = y_test.values


# At this point, *x_train_new*, *y_train_new* and *x_test_new*, *y_test_new* are the train and test data set after all of processes

# # 5. Perceptron

# In[28]:


from sklearn.linear_model import Perceptron

clf = Perceptron()
clf.fit(x_train_new, y_train_new)
# predict
y_train_pred = clf.predict(x_train_new)
y_test_pred = clf.predict(x_test_new)


# In[29]:


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score
def metric_measure(y_train_new, y_train_pred, y_test_new, y_test_pred):
    a_train = accuracy_score(y_train_new, y_train_pred)
    p_train = precision_score(y_train_new, y_train_pred, average='binary')
    r_train = recall_score(y_train_new, y_train_pred, average='binary')
    f1_train = f1_score(y_train_new, y_train_pred, average='binary')
    print('accuracy, precision, recall and F1-score of train set is:',a_train,p_train,r_train,f1_train)
    
    a_test = accuracy_score(y_test_new, y_test_pred)
    p_test = precision_score(y_test_new, y_test_pred, average='binary')
    r_test = recall_score(y_test_new, y_test_pred, average='binary')
    f1_test = f1_score(y_test_new, y_test_pred, average='binary')
    print('accuracy, precision, recall and F1-score  of test set is:',a_test,p_test, r_test, f1_test)

# Metrics measure results of Perceptron

# In[30]:


print("\n*Perceptron:")
metric_measure(y_train_new, y_train_pred, y_test_new, y_test_pred)


# # 6.1 SVM

# In[31]:


from sklearn.svm import LinearSVC
clf_svm = LinearSVC()
clf_svm.fit(x_train_new, y_train_new)
y_train_pred = clf_svm.predict(x_train_new)
y_test_pred = clf_svm.predict(x_test_new)


# Metrics measure results of SVM

# In[32]:


print("\n*SVM:")
metric_measure(y_train_new, y_train_pred, y_test_new, y_test_pred)


# # 6.2 Logistic Regression

# In[33]:


from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression(random_state = 0).fit(x_train_new, y_train_new)
y_train_pred = clf_lr.predict(x_train_new)
y_test_pred = clf_lr.predict(x_test_new)


# Metrics measure results of Logictic Regression

# In[34]:


print("\n*Logictic Regression:")
metric_measure(y_train_new, y_train_pred, y_test_new, y_test_pred)


# # 7. Naive Bayes

# In[35]:


from sklearn.naive_bayes import MultinomialNB
clf_mnb = MultinomialNB()
clf_mnb.fit(x_train_new, y_train_new)
y_train_pred = clf_mnb.predict(x_train_new)
y_test_pred = clf_mnb.predict(x_test_new)


# Metrics measure results of Naive Bayes

# In[36]:


print("\n*Naive Bayes:")
metric_measure(y_train_new, y_train_pred, y_test_new, y_test_pred)

