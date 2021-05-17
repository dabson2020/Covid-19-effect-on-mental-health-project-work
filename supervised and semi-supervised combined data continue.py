#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


new_df = pd.read_csv('data_no_label_kNN_update.csv')


# In[12]:


new_df['tweet'] = new_df['tweet'].apply(str)


# In[57]:


xwords = pd.Series(['thank','death','update','hospital','government','going','really','today','coronavirus','wuhan','nan','itu2019s','iu2019m','yu2019all','virus','china','people','chinese','think','pneumonia','covid19','covid','italy','country','covid19','covid','pandemic','mental','south','korea','outbreak','spread','trump',])


# In[58]:


banned_words = set(word.strip().lower() for word in xwords)


# In[59]:


def delete_banned_words(matchobj):
    word = matchobj.group(0)
    if word.lower() in banned_words:
        return ""
    else:
        return word


# In[60]:


import re


# In[61]:


word_pattern = re.compile('\w+')


# In[62]:


new_df['tweet'] = [word_pattern.sub(delete_banned_words,sentence) for sentence in new_df['tweet']]


# In[63]:


new_df = new_df[['tweet','predict_KNN','predict_KNN_semi_supervised']]


# In[64]:


new_df.isnull().sum()


# In[65]:


positive = new_df[new_df['predict_KNN_semi_supervised'] == 1]


# In[66]:


negative = new_df[new_df['predict_KNN_semi_supervised'] == -1]
neutral = new_df[new_df['predict_KNN_semi_supervised'] == 0]


# In[67]:


all_positive_words = ' '.join([w for w in positive['tweet']])
all_negative_words = ' '.join([w for w in negative['tweet']])
all_neutral_words = ' '.join([w for w in neutral['tweet']])


# In[68]:


len(all_positive_words)


# In[69]:


len(all_negative_words)


# In[70]:


len(all_neutral_words)


# In[71]:


from wordcloud import WordCloud


# In[72]:


import matplotlib.pyplot as plt
from PIL import Image
import string


# In[73]:


positive_wordcloud = WordCloud(width = 800, height = 500, random_state = 21, max_font_size = 110,background_color = 'black').generate(all_positive_words)


# In[74]:


plt.figure(figsize = (10,7))
plt.imshow(positive_wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# In[75]:


negative_wordcloud = WordCloud(width = 800, height = 500, random_state = 21, max_font_size = 110,background_color = 'black').generate(all_negative_words)


# In[76]:


plt.figure(figsize = (10,7))
plt.imshow(negative_wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# In[77]:


neutral_wordcloud = WordCloud(width = 800, height = 500, random_state = 21, max_font_size = 110,background_color = 'black').generate(all_neutral_words)


# In[78]:


plt.figure(figsize = (10,7))
plt.imshow(neutral_wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# for Supervised model

# In[79]:


positive_supervised = new_df[new_df['predict_KNN'] == 1]
negative_supervised = new_df[new_df['predict_KNN'] == -1]
neutral_supervised = new_df[new_df['predict_KNN'] == 0]


# In[80]:


all_positive_supervised = ' '.join([w for w in positive_supervised['tweet']])
all_negative_supervised = ' '.join([w for w in negative_supervised['tweet']])
all_neutral_supervised = ' '.join([w for w in neutral_supervised['tweet']])


# In[81]:


len(all_positive_supervised)


# In[82]:


len(all_negative_supervised)


# In[83]:


len(all_neutral_supervised)


# In[ ]:




