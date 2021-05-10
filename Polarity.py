#!/usr/bin/env python
# coding: utf-8

# In[4]:



import math 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import scikitplot as skplt

import seaborn as sns


# In[21]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv') 

print(train_df.shape, test_df.shape)


# In[13]:



train_df.head(5)


# In[16]:



train_dfa = train_df.copy()
cleanup_nums = {"sentiment":     {"Negative": 1, "Positive": 2}}
train_dfa.replace(cleanup_nums, inplace=True)
train_dfa.head()


# In[17]:


ax = train_dfa['sentiment'].value_counts(sort=False).plot(kind='barh', color='#EE4747')
ax.set_xlabel('Count')
ax.set_ylabel('Labels')


# In[18]:


train_dfa['len'] = train_dfa['text'].str.len() # Store string length of each sample
train_dfa = train_dfa.sort_values(['len'], ascending=True)
train_dfa = train_dfa.dropna()
train_dfa.head(10) #We see that most of the short text phrases are rated positive


# In[22]:


test_dfa = test_df.copy()
test_dfa.replace(cleanup_nums, inplace=True)
test_dfa.head(5)
test_dfa


# In[23]:


# Create a transformation pipeline
# The pipeline sequentially applies a list of transforms and as a final estimator logistic regression 
pipeline_log = Pipeline([
                ('count', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(solver='liblinear', multi_class='auto')),
        ])

# Train model using the created sklearn pipeline
learner_log = pipeline_log.fit(train_dfa['text'], train_dfa['sentiment'])

# Predict class labels using the learner function
test_dfa['pred'] = learner_log.predict(test_dfa['text'])
y_true = test_dfa['sentiment']
y_pred = test_dfa['pred']
target_names = ['negative', 'positive']

# Confusion Matrix
results_log = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
results_df_log = pd.DataFrame(results_log).transpose()
print(results_df_log)
skplt.metrics.plot_confusion_matrix(y_true,  y_pred, figsize=(12,12))


# In[24]:


# Create a pipeline which transforms phrases into normalized feature vectors and uses a bayes estimator
pipeline_bayes = Pipeline([
                ('count', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('gnb', MultinomialNB()),
                ])

# Train model using the created sklearn pipeline
learner_bayes = pipeline_bayes.fit(train_dfa['text'], train_dfa['sentiment'])

# Predict class labels using the learner function
test_dfa['pred'] = learner_bayes.predict(test_dfa['text'])
y_true = test_dfa['sentiment']
y_pred = test_dfa['pred']
target_names = ['negative', 'positive']

# Confusion Matrix
results_bayes = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
results_df_bayes = pd.DataFrame(results_bayes).transpose()
print(results_df_bayes)
skplt.metrics.plot_confusion_matrix(y_true, y_pred, figsize=(12,12))


# In[25]:


# Plotting the data

# Preparing bayes classifier metrics
bayes_precision = results_df_bayes['precision'].at['weighted avg']
bayes_f1_score = results_df_bayes['f1-score'].at['weighted avg']
bayes_accuracy = results_df_bayes['recall'].at['weighted avg']

# Preparing logistic regression classifier metrics
log_precision = results_df_log['precision'].at['weighted avg']
log_f1_score = results_df_log['f1-score'].at['weighted avg']
log_accuracy = results_df_log['recall'].at['weighted avg']

# Preparing the plot
fig, ax1 = plt.subplots(figsize=(6, 8))

# set width of bar
barWidth = 0.15
 
# set height of bar
accuracy = [log_accuracy, bayes_accuracy]
f1_score = [log_f1_score, bayes_f1_score]
precision = [log_precision, bayes_precision]

# Set position of bar on X axis
r1 = np.arange(2)
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, accuracy, color='#EE4747', width=barWidth, edgecolor='white', label='accuracy')
plt.bar(r2, f1_score, color='#3333ff', width=barWidth, edgecolor='white', label='recall')
plt.bar(r3, precision, color='#2d7f5e', width=barWidth, edgecolor='white', label='precision')
 
# Add xticks on the middle of the group bars
plt.xlabel('algorithm', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(accuracy))], ['logistic regression', 'bayes'])
 
# Create legend & Show graphic
plt.legend()
plt.grid(color='white')
plt.show()


# In[30]:


testphrases = ['Mondays is great', "I don't this product", 'Terrible service']
for testphrase in testphrases:
    resultx = learner_log.predict([testphrase])
    dict = {1: 'Negative', 2: 'Positive'}
    print(testphrase + '-> ' + dict[resultx[0]])


# In[ ]:




