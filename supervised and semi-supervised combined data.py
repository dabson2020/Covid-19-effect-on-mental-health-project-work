#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


jan= pd.read_csv('output_data1_jan.csv')
feb = pd.read_csv('output_data2_feb.csv')
mar = pd.read_csv('output_data3_mar.csv')
apr = pd.read_csv('output_data8c_apr.csv')
may = pd.read_csv('output_data11a_may.csv')
june = pd.read_csv('output_data11c_june.csv')
july = pd.read_csv('output_data_july1.csv')
aug = pd.read_csv('output_main_data_aug1.csv')
sept = pd.read_csv('output_main_data_sept1.csv')
oct = pd.read_csv('output_data_oct2.csv')


# In[3]:


merge_data = pd.concat([jan,feb,mar,apr,may,june,july,aug,sept,oct])


# In[4]:


merge_data = merge_data.reset_index(drop = True)


# In[5]:


merge_data.info()


# In[6]:


merge_data['Date'] = pd.to_datetime(pd.Series(merge_data['Date']))


# In[7]:


import pandas as pd
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
from PIL import Image
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


def remove_url(row):
    txt = str(row['text']).split('https')[0]
    return txt


# In[9]:


merge_data['tidy_text'] = merge_data.apply(remove_url, axis = 1)


# In[10]:


merge_data['tidy_text'] = merge_data['text'].str.replace('@[\w:]*','').str.replace('ud83eudd14',"")


# In[11]:


data = merge_data


# In[12]:


data['tidy_text'] = data['tidy_text'].str.replace("RT ", '').str.replace('u201d','').str.replace('nn','').str.replace('amp','mental')
data['tidy_text'] = data['tidy_text'].str.replace('\\','').str.replace('u201c','').str.replace('nu2019t','')
data = data[~data['tidy_text'].astype(str).str.startswith('u')]
data = data[~data['tidy_text'].astype(str).str.contains('ud')]
data = data[~data['tidy_text'].astype(str).str.contains('#')]


# In[13]:


data['tidy_text'] = data['tidy_text'].apply(str)


# In[14]:


data = data.reset_index(drop = True)


# In[15]:


data['month'] = pd.DatetimeIndex(data['Date']).month


# In[16]:


data.info()


# In[17]:


data['tidy_text'] = data['tidy_text'].apply(lambda x: x.lower())


# In[18]:


data['month'] = data['month'].replace(1,'Jan').replace(2,'Feb').replace(3,'Mar').replace(4,'Apr').replace(5,'May').replace(6,'June').replace(7,'July').replace(8,'Aug').replace(9,'Sept').replace(10,'Oct')


# In[19]:


data.head()


# In[20]:


#data.to_csv('combined_data.csv', index_label = False)


# In[21]:


frequency = data['month'].value_counts()


# In[22]:


new_freq = pd.DataFrame(frequency)


# In[23]:


new_freq.reset_index(inplace = True)


# In[24]:


new_freq.columns = ['date','frequency']


# In[25]:


new_freq


# In[26]:


pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings('ignore')


# In[27]:


plt.bar(x = new_freq.date, height= new_freq.frequency)
plt.xticks(rotation = 45)
plt.title('Tweet frequency for each day', color = 'red')
plt.xlabel('Date of Tweets', color = 'blue')
plt.ylabel('Tweet Frequency',color = 'blue')


# In[28]:


stop_words = set(stopwords.words('english'))


# In[29]:


data['tidy_text'] = data['tidy_text'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))
data['tidy_text'] = data['tidy_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)> 4]))


# In[30]:


data['tidy_text'] = data['tidy_text'].str.replace('wYxJPGnXe0','').str.replace('onl','').str.replace('KEY','').str.replace('https','').str.replace('VID','')
data = data[~data['tidy_text'].astype(str).str.contains('://')]


# In[31]:


data = data[~data['tidy_text'].astype(str).str.startswith('u')]
data['tidy_text']=data['tidy_text'].replace(['u2019d','u201d','u00eda','u00f3n','u2026'],"",regex = True)


# In[32]:


data = data.reset_index(drop = True)


# In[33]:


data.count()


# In[34]:


all_words = ' '.join([w for w in data['tidy_text']])


# In[35]:


len(all_words)


# In[36]:


wordcloud = WordCloud(width = 800, height = 500, random_state = 21, max_font_size = 110,background_color = 'black').generate(all_words)


# In[37]:


plt.figure(figsize = (10,7))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# In[38]:


import nltk.data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import word_tokenize


# In[39]:


sid = SentimentIntensityAnalyzer()


# In[40]:


scores = [sid.polarity_scores(i) for i in data.tidy_text]


# In[41]:


df1 = pd.DataFrame(scores)


# In[42]:


df1.count()


# In[43]:


data_copy = data


# In[44]:


covid_data = pd.concat([data_copy,df1], axis = 1)


# In[45]:


covid_data.head()


# In[47]:


conditions = [(covid_data['compound']>=0.05),(covid_data['compound']<= - 0.05),(covid_data['compound'] > -0.05) & (covid_data['compound'] < 0.05)]
values = ['Positive', 'Negative','Neutral']
covid_data['sentiments'] = np.select(conditions,values)


# In[48]:


covid_data.head()


# In[49]:


c1 = (covid_data['sentiments']=='Neutral')
c2 = (covid_data['sentiments'] == 'Positive')
c3 = (covid_data['sentiments'] == 'Negative') 

covid_data['target'] = np.select([c1,c2,c3],[0,1,-1],default = 'Other')


# In[50]:


covid_data.head()


# In[50]:


covid_data['sentiments'].value_counts()


# In[51]:


sentiment_freq = pd.DataFrame(covid_data['sentiments'].value_counts())


# In[52]:


sentiment_freq.reset_index(inplace = True)


# In[53]:


sentiment_freq.columns = ['sentiments','frequency']


# In[54]:


sns.barplot(x = 'sentiments', y = 'frequency', data = sentiment_freq, palette = "YlOrBr")
plt.title('Sentiments Distributions',fontsize= 20, color = 'sandybrown')
plt.xlabel('Sentiments',fontsize = 13, color = 'sandybrown')
plt.ylabel('Tweet_frequency', fontsize = 13,color = 'sandybrown' )


# In[55]:


df= covid_data


# In[56]:


df = df[['tidy_text','sentiments']]


# In[57]:


xwords = pd.Series(['coronavirus','wuhan','itu2019s','iu2019m','yu2019all','virus','china','people','chinese','think','pneumonia','covid19','covid','italy','country','pandemic','mental','south','korea','outbreak','spread','trump',])


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


word_pattern = re.compile('\w+')


# In[61]:


df['tidy_text'] = [word_pattern.sub(delete_banned_words,sentence) for sentence in df['tidy_text']]


# In[62]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output


import collections
import nltk
from keras.preprocessing import sequence
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[63]:


train, test = train_test_split(df,test_size = 0.3)

# Removing neutral sentiments
train = train[train.sentiments != "Neutral"]
test = test[test.sentiments != 'Neutral']


# In[64]:


train.head()


# In[65]:


train_pos = train[ train['sentiments'] == 'Positive']
train_pos = train_pos['tidy_text']
train_neg = train[ train['sentiments'] == 'Negative']
train_neg = train_neg['tidy_text']


# In[66]:


def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                          width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Positive words")
wordcloud_draw(train_pos,'white')
print("Negative words")
wordcloud_draw(train_neg)


# In[67]:


df2= covid_data


# In[70]:


labelled_data = covid_data[0:100]


# In[72]:


labelled_data.head()


# In[73]:


label_data = labelled_data[['tidy_text','target']]


# In[104]:


df = label_data


# In[105]:


unlabelled_data = covid_data['tidy_text'][111::]


# In[106]:


df1 = unlabelled_data


# In[107]:


df1.reset_index(drop = True)


# In[108]:


df1 = pd.DataFrame(df1)


# In[109]:


df1 = df1.reset_index(drop = True)


# In[110]:


import pandas as pd
import numpy as np
import collections
get_ipython().system('pip install nltk')
import nltk
from keras.preprocessing import sequence
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, RepeatedKFold
import matplotlib.pyplot as plt
#train, test = train_test_split(df,test_size = 0.3)
import sys


# In[111]:


def feature_gen(f):
    maxlen=0
    word_freqs=collections.Counter()
    num_recs=0

    for line in f:
        words=nltk.word_tokenize(str(line).lower())
        if len(words)>maxlen:
            maxlen=len(words)
        for word in words:
            word_freqs[word] +=1
        num_recs +=1

    MAX_FEATURES=17000
    MAX_SENTENCE_LENGTH =50
    #La entrada para la RNN son palabras indexadas con su numero de frecuencia en el documento, ademas consideramos 2 etiquetas para palabras que no se encuentran en el corpus
    #vocab_size = min(MAX_FEATURES, len(word_freqs))+2
    word2index = {x[0]: i+2 for i,x in enumerate(word_freqs.most_common(MAX_FEATURES))}
    word2index["PAD"]=0
    word2index["UNK"]=1
    #index2word={v:k for k, v in word2index.items()}
    #Preparamos nuestros datos para darlos como entrada en la RNN
    X=np.empty((num_recs,),dtype=list)
    train = f
    i=0
    for line in train:
        words=nltk.word_tokenize(str(line).lower())
        seqs=[]
        for word in words:
            if word in word2index:
                #print(word)
                #print(word2index[word])
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        #print(i)
        #print(seqs)
        X[i]=seqs
        i += 1
    X=sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
    #print(type(X))
    #print(X.size)
    return X, word_freqs


# In[112]:


nltk.download('punkt')
#f= train.tidy_text
#Xtrain, word_freqs = feature_gen(f)
#y1 = train.target

#f= test.tidy_text
#Xtest, word_freqs = feature_gen(f)

#y_train = train.target
#y_test = test.target
f = df.tidy_text
X, word_freqs = feature_gen(f)
y = df.target.astype('category')


# In[113]:


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBRegressor
#from sklearn.linear_model import BayesianRidge, Ridge, ElasticNet
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
#from sklearn.neural_network import MLPRegressor

#from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from numpy import mean


# In[114]:


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]


# In[115]:


classifiers = [
    KNeighborsClassifier(3),
    #SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    GradientBoostingClassifier(),
    LogisticRegression()]


# In[116]:


print('Number of folds : 5')
# prepare the cross-validation procedure
cv = RepeatedKFold(n_splits=2, n_repeats=1, random_state=1)


# In[117]:


classifiers = [
    KNeighborsClassifier(3)]
for model in classifiers:
    print(model)
    
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = KNeighborsClassifier(3)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        df = pd.DataFrame(np.concatenate((f[test_index].to_numpy().reshape(-1,1), y_test.to_numpy().reshape(-1,1), y_pred.reshape(-1,1)), axis=1), columns=['tweet', 'label','predict_KNN'])
        df.to_csv('data_label_kNN.csv')
        #print(df)
        print('train score:{%0.2f} test_score:{%0.2f}' %(train_score, test_score))
        
        model_supervised = model
        ftest_train, ftest_test, trainX, testX, trainy, testy, train_y,test_y = train_test_split(f[test_index],X_test, y_pred, y_test, test_size=0.33, random_state=42)
        X_train1 = np.concatenate((X_train, trainX))
        y_train1 = np.concatenate((y_train, trainy))
        model = KNeighborsClassifier(3)
        model.fit(X_train1, y_train1)
        semi_train_score = model.score(X_train1, y_train1)
        
        model_semi_supervised = model
        y_pred = model.predict(testX)
        df = pd.DataFrame(np.concatenate((ftest_test.to_numpy().reshape(-1,1), test_y.to_numpy().reshape(-1,1), testy.reshape(-1,1), y_pred.reshape(-1,1)), axis=1), columns=['tweet', 'label','predict_KNN','predict_KNN_semi_spuervised'])
        df.to_csv('data_label_kNN_semispervised.csv')
        semi_test_score = model.score(testX, testy)
        print('semi supervised train score:{%0.2f} test_score:{%0.2f}' %(semi_train_score, semi_test_score)) 
        #scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        # report performance
        #print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
        #scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        #score_description = " %0.2f (+/- %0.2f)" % (np.sqrt(scores.mean()*-1), scores.std() * 2)
        #print('{model:25} CV-5 RMSE: {score}'.format(model=model.__class__.__name__,score=score_description))


# In[119]:


f = df1.tidy_text
X, word_freqs = feature_gen(f)
y_pred_supervised = model_supervised.predict(X)
y_pred_semi_supervised = model_semi_supervised.predict(X)
df2 = pd.DataFrame(np.concatenate((f.to_numpy().reshape(-1,1), y_pred_supervised.reshape(-1,1), y_pred_semi_supervised.reshape(-1,1)), axis=1), columns=['tweet', 'predict_KNN','predict_KNN_semi_supervised'])
df2.to_csv('data_no_label_kNN_update.csv')


# In[120]:


new_df = pd.read_csv('data_no_label_kNN_update.csv')


# In[121]:


new_df.info()


# In[122]:


new_df.head()


# In[ ]:




