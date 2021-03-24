#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:13:22 2021

@author: poulomi
"""

'''
Multiclass text classification: In this project text is classified into one of the five categories. The data set used is BBC news data.
'''

#importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import seaborn as sns
from wordcloud import WordCloud
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils


#importing the data as a pandas dataframe
data = pd.read_csv("bbc-text.csv")

#displaying the top 5 rows of the data
data.head()

#columns in the data: category, text
data.columns

#finding the types of categories, there are 5 types: tech, business, sport, entertainment and politics
#categories = []
data['category'].unique()

#shape of data
data.shape

#datatypes
data.dtypes

#checking for null values
data.isnull().any()

#dropping duplicate rows if any
data = data.drop_duplicates()
data = data.reset_index(inplace = False)[['category','text']]

#print the shape after removing duplicates
data.shape

# This data does not seem to be much imbalanced, so no need 
#of oversampleing or undersampling of data here
#counting the number of each category and plotting them
data.category.value_counts()
sns.countplot(data.category)
plt.savefig("Category distribution")

#####Exploratory Data Analysis##############
#checking the length of each news article, that is the total number of words in each article
data['News_length'] = data['text'].str.len()
data['News_length']

#Distribution plot for proportion of articles
sns.distplot(data['News_length']).set_title('News_length_Distribution')
plt.savefig("News_length_distribution_plot.png")


#length of column: 2126
#len(data.text)

# data cleaning
clean_text = []
for w in range(len(data.text)):
    desc = data['text'][w].lower()
    
    #text = text.lower().replace('\n',' ').replace('\r','').strip()
    #text = re.sub(' +', ' ', text)
    #text = re.sub(r'[^\w\s]', '', text)
    
    #remove punctuation
    desc = re.sub('[^a-zA-Z]', ' ', desc)
    
    #remove tags
    desc=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",desc)
    
    #remove digits and special chars
    desc=re.sub("(\\d|\\W)+"," ",desc)
    
    clean_text.append(desc)
    
#assign the cleaned descriptions to the data frame
data['clean_text'] = clean_text
        
data.head()

#creating separate dataframes for each category
df_business = data.loc[data.category=='business']

df_tech = data.loc[data.category=='tech']

df_sport = data.loc[data.category=='sport']

df_entertainment = data.loc[data.category=='entertainment']

df_politics = data.loc[data.category=='politics']


#Wordcloud visualization, can be repeated for each category, here I have done for tech category only
wordcloud = WordCloud(width=800, height=800, stopwords=set(stopwords.words('english')), min_font_size=20, max_words=1000, background_color='skyblue').generate(str(df_tech['clean_text']))

#ploting the word cloud
plt.figure(figsize = (16,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.savefig("Wordcloud_tech.png")



########## Cleaning the data, preprocessing the data
#converting text into all lower case, replacing \n with space and \r with null and removing spaces from start or end of each 
#article by using strip method
#using regular expressions for removing multiplt spaces with one space
def process_text(text):
    #removing stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    print(word_tokens)
    text = ' '.join(filtered_sentence)
    return text

data['Text_parsed']= data['clean_text'].apply(process_text)

data['Text_parsed']


#######label encoding
#convert categorical data that is a text here to numeric
label_encoder = preprocessing.LabelEncoder()
data['Target'] = label_encoder.fit_transform(data['category'])

data.head()

#saving the preprocessed data into a csv file, to use later, optional step
data.to_csv("bbc-text-preprocessed.csv", index=False)


###########Splitting the data into training and testing

X_train, X_test, y_train, y_test = train_test_split(data['Text_parsed'], data['Target'], test_size= 0.2, random_state = 42)

X_train.shape

X_test.shape

X_train[0]


###########Feature Extraction
#Using tfidf
tfidf = TfidfVectorizer(lowercase = False, ngram_range = (1,2), min_df=10, max_features=300, sublinear_tf=True )

features_train = tfidf.fit_transform(X_train).toarray()
features_test = tfidf.transform(X_test).toarray()

######## Model Building using traditional machine learning algorithms

##Random Forest Classifier
model = RandomForestClassifier(n_estimators=500)
model.fit(features_train, y_train)
model_predictions = model.predict(features_test)
accuracy = accuracy_score(y_test, model_predictions)
report = classification_report(y_test, model_predictions)
print("Accuracy: ", accuracy) #92.957%
print("Classification_report: ", report)


##Logistic Regression
model = LogisticRegression()
model.fit(features_train, y_train)
model_predictions = model.predict(features_test)
accuracy = accuracy_score(y_test, model_predictions)
report = classification_report(y_test, model_predictions)
print("Accuracy: ", accuracy) #94.366%
print("Classification_report: ", report)


##KNN algorithm
model = KNeighborsClassifier()
model.fit(features_train, y_train)
model_predictions = model.predict(features_test)
accuracy = accuracy_score(y_test, model_predictions)
report = classification_report(y_test, model_predictions)
print("Accuracy: ", accuracy) #91.549%
print("Classification_report: ", report)


##Decision Tree
model = DecisionTreeClassifier()
model.fit(features_train, y_train)
model_predictions = model.predict(features_test)
accuracy = accuracy_score(y_test, model_predictions)
report = classification_report(y_test, model_predictions)
print("Accuracy: ", accuracy) #80.516%
print("Classification_report: ", report)

##NaiveBayes
model = GaussianNB()
model.fit(features_train, y_train)
model_predictions = model.predict(features_test)
accuracy = accuracy_score(y_test, model_predictions)
report = classification_report(y_test, model_predictions)
print("Accuracy: ", accuracy) #89.436%
print("Classification_report: ", report)

#Support vector machine
classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
classifier.fit(features_train, y_train)
model_predictions = classifier.predict(features_test)
report = classification_report(y_test, model_predictions)
accuracy = accuracy_score(y_test, model_predictions)
print("Accuracy: ", accuracy) #93.192%
print("Classification_report: ", report)


######Hyper Parameter tuning using GridSearchCV, I just did for Random forest and Logistic regression

### for random forest
#parameter definition of GridSearchCV, taking all the combinations of the parameters
n_estimators = [100,300,500,800,1200]

max_depth = [5,10,15,20,30]

min_samples_split = [2,5,10,20,100]

min_samples_leaf = [1,2,5,10]

hyperparams = dict(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split,
                  min_samples_leaf=min_samples_leaf)

hyperparams

#using it with different models, n_jobs=-1 enables multiple CPU cores to work together
mymodel = RandomForestClassifier()
gridparams = GridSearchCV(mymodel, hyperparams, cv=5, verbose=1, n_jobs=-1)
bestparams = gridparams.fit(features_train, y_train)

bestparams.best_params_


mymodel = RandomForestClassifier(n_estimators=800, max_depth= 30, min_samples_leaf=1, min_samples_split=5, random_state=1)
mymodel.fit(features_train, y_train)
mymodel_predictions = mymodel.predict(features_test)
accuracy = accuracy_score(y_test, mymodel_predictions)
report = classification_report(y_test, mymodel_predictions)
print("Accuracy: ", accuracy)
print("Classification_report: ", report)


##################for logistic regression
grid = {'C':[0.1,0.001,1], 'penalty':['l1','l2']}
mymodel=LogisticRegression()
gridparams = GridSearchCV(mymodel,grid,cv=5,verbose=1)
bestparams = gridparams.fit(features_train, y_train)


bestparams.best_params_


mymodel = LogisticRegression(C=1, penalty='l2')
mymodel.fit(features_train, y_train)
mymodel_predictions = mymodel.predict(features_test)
accuracy = accuracy_score(y_test, mymodel_predictions)
report = classification_report(y_test, mymodel_predictions)
print("Accuracy: ", accuracy)
print("Classification_report: ", report)



#DOC2VEC for numerical representation of sentence 
#first is word2vec that gives us the numeric embedding for each word, in an n dimension
data_processed = pd.read_csv("bbc-text-preprocessed.csv")
data_processed.columns

#we just need the clean text and the encoded labels for the target
new_data = pd.DataFrame()
new_data['text'] = data_processed['Text_parsed']
new_data['category'] = data_processed['Target']

new_data.head()


#converting trainig and testing data to the gensim format
def tag_data(text, tag):
    tagged_doc = []
    for i,v in enumerate(text):
        tagged = tag+'_'+str(i)
        tagged_doc.append(TaggedDocument(v.split(), [tagged]))
    return tagged_doc

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(new_data['text'], new_data['category'], test_size=0.2, random_state=1)

X_train_new.shape
X_test_new.shape

X_train_new = tag_data(X_train_new, 'train')
X_train_new[1]

X_test_new = tag_data(X_test_new, 'test')

final_data = X_train_new+X_test_new


###Initializing the Doc2Vec model and training it for a few epochs
model_ = Doc2Vec(dm=0, vector_size=300, min_alpha=0.065, alpha=0.065, min_count=1)
model_.build_vocab([x for x in final_data])

for epoch in range(30):
    model_.train(utils.shuffle([x for x in final_data]), total_examples=len(final_data), epochs=1)
    model_.alpha -= 0.002 #decrease the learning rate
    model_.min_alpha = model_.alpha #fix the learning rate, no decay


def get_vectors(model, corpus_size, vectors_size, vectors_type):
    vectors = np.zeros((corpus_size, vectors_size))
    
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
        
    return vectors

train_vectors_ = get_vectors(model_, len(X_train_new), 300, 'train')
test_vectors_ = get_vectors(model_, len(X_test_new), 300, 'test')

train_vectors_.shape

test_vectors_.shape

train_vectors_[:5]

model_1 = LogisticRegression()
model_1.fit(train_vectors_, y_train_new)
model_pred = model_1.predict(test_vectors_)

print("Accuracy_final: ", accuracy_score(model_pred, y_test_new)) #95.305
print(classification_report(y_test_new, model_pred))





























