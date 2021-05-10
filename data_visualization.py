#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.pipeline import Pipeline

import pandas as pd
import string
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

df= pd.read_csv('extracted data.csv')
df.head(5)


# In[2]:


df.describe()


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


sns.pairplot(df)


# In[6]:


sns.distplot(df["stars"])


# In[7]:


sns.distplot(df["user_given_stars"])


# In[8]:


df["stars"].value_counts()


# In[9]:


df["user_given_stars"].value_counts()


# In[10]:


import matplotlib.pyplot as plt

df = pd.read_csv('extracted data.csv')
df["length"] = df["text"].apply(len)
sns.jointplot(x=df["length"],
              y=df["stars"],
              data=df, kind='reg')


# In[11]:


df = pd.read_csv('extracted data.csv')
X_Data = df["text"]
Y_Data = df["user_given_stars"]

cv = CountVectorizer()
X_Data = cv.fit_transform(X_Data)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Data, Y_Data,test_size=0.3,random_state=101)
model = MultinomialNB()
model.fit(X_Train,Y_Train)
predicted = model.predict(X_Test)
print(classification_report(Y_Test, predicted))


# In[8]:


#import WordCloud
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'notebook')
df = pd.read_csv('extracted data.csv')
#df.head()


comment_words = '' 
stopwords = set(STOPWORDS)
#stopwords = ['nan', 'NaN', 'Nan', 'NAN'] + list(STOPWORDS)

values = df['text'].values

for val in values: 
    val = str(val) 
    tokens = val.split() 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    comment_words += ' '.join(tokens)+' '


            
facecolor = 'white'

wordcloud = WordCloud(width=1000, height=600, 
            background_color=facecolor, 
            stopwords=stopwords,
            min_font_size=10).generate(comment_words)
plt.figure(figsize=(10,6), facecolor=facecolor) 
plt.imshow(wordcloud) 
plt.axis('off') 
plt.tight_layout(pad=2)
#filename = 'wordcloud'
#plt.savefig(filename+'.png', facecolor=facecolor)


# In[3]:


import pandas as pd


df = pd.read_csv('extracted data.csv')
first_review = df.loc[0, "text"]
print(first_review)

import spacy

nlp = spacy.load("en_core_web_sm")
[sent.text for sent in nlp(first_review).sents]



%%timeit -n 10
# SpaCy with DependencyParser
nlp = spacy.load("en_core_web_sm")
df.loc[:1900, "text"].apply(lambda x: [sent.text for sent in nlp(x).sents])
df.to_csv("scripts_tokenized.csv")


# In[6]:


#import WordCloud for positive reviews
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'notebook')
df = pd.read_csv('positive.csv')
#df.head()


comment_words = '' 
stopwords = set(STOPWORDS)
#stopwords = ['nan', 'NaN', 'Nan', 'NAN'] + list(STOPWORDS)

values = df['text'].values

for val in values: 
    val = str(val) 
    tokens = val.split() 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    comment_words += ' '.join(tokens)+' '


            
facecolor = 'white'

wordcloud = WordCloud(width=1000, height=600, 
            background_color=facecolor, 
            stopwords=stopwords,
            min_font_size=10).generate(comment_words)
plt.figure(figsize=(10,6), facecolor=facecolor) 
plt.imshow(wordcloud) 
plt.axis('off') 
plt.tight_layout(pad=2)
#filename = 'wordcloud'
#plt.savefig(filename+'.png', facecolor=facecolor)


# In[5]:


#import WordCloud
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'notebook')
df = pd.read_csv('negative.csv')
#df.head()


comment_words = '' 
stopwords = set(STOPWORDS)
#stopwords = ['nan', 'NaN', 'Nan', 'NAN'] + list(STOPWORDS)

values = df['text'].values

for val in values: 
    val = str(val) 
    tokens = val.split() 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    comment_words += ' '.join(tokens)+' '


            
facecolor = 'white'

wordcloud = WordCloud(width=1000, height=600, 
            background_color=facecolor, 
            stopwords=stopwords,
            min_font_size=10).generate(comment_words)
plt.figure(figsize=(10,6), facecolor=facecolor) 
plt.imshow(wordcloud) 
plt.axis('off') 
plt.tight_layout(pad=2)
#filename = 'wordcloud'
#plt.savefig(filename+'.png', facecolor=facecolor)


# In[ ]:




