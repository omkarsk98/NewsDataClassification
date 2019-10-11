# -*- coding: utf-8 -*-
"""NewsDataClassification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qPKrhra33j-b3CtlFX5ILsbd9YoxvPyZ

Download the repo for original and raw csv data
"""

!git clone https://github.com/omkarsk98/NewsDataClassification.git

"""Go the directory and checkout to the required folder"""

# Commented out IPython magic to ensure Python compatibility.
# %cd NewsDataClassification
!git checkout development

"""Download trained model and unzip it."""

!wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
!gunzip GoogleNews-vectors-negative300.bin.gz

"""Download nltk libraries and dependencies"""

import nltk
nltk.download('punkt')
nltk.download('stopwords')

"""import all required libraries"""

import pandas as pd
from gensim import models
from sklearn.linear_model import LogisticRegression
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score ,confusion_matrix
import time
from sklearn.manifold import TSNE
from google.colab import files

# Load word2vec model (trained on Google's corpus)
model = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True) 
# Check dimension of word vectors
print("Dimensions of the model",model.vector_size)

"""Read the raw data and set max records to be used"""

# read csv
main_data = pd.read_csv('News_Final.csv')

# read titles from it
article_titles = main_data['TITLE']
labels = main_data["CATEGORY"]

# Create a list of strings, one for each title
titles_list = [title for title in article_titles]
# form a single string fro the list of strings
big_title_string = ' '.join(titles_list)

# define total records to be considered for analysis
total = 50000
# 422178 total records as max value

"""Tokenise all words and get stop words for english"""

# Tokenize the string into words
tokens = word_tokenize(big_title_string)

# Remove non-alphabetic tokens, such as punctuation
words = [word.lower() for word in tokens if word.isalpha()]
stop_words = set(stopwords.words('english'))

"""Define all the function that can be used for later stage"""

def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in model.vocab]
    return np.mean(model[doc], axis=0)

# Our earlier preprocessing was done when we were dealing only with word vectors
# Here, we need each document to remain a document 
def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()] 
    return doc

# Function that will help us drop documents that have no word vectors in word2vec
def has_vector_representation(word2vec_model, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in word2vec_model.vocab for word in doc)

# Filter out documents
def filter_docs(corpus, texts, labels, condition_on_doc):
    """
    Filter corpus and texts given the function condition_on_doc which takes a doc. The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)
    
    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]
    

    corpus = [doc for doc in corpus if condition_on_doc(doc)]
    
    final_labels = []
    for i in range(len(corpus)):
      if condition_on_doc(corpus[i]):
        final_labels.append(labels[i])
    
    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts, final_labels)

"""Remove stop words, non vocab words, empty docs and prepare vector for each title"""

# # Preprocess the corpus to get list of documents with stop words removed and containing only the words that are present in the vocab
corpus = [preprocess(title) for title in titles_list]
# # still contains all the documents, nothing is filtered

# # Remove docs that don't include any words in W2V's vocab
corpus, titles_list, labels = filter_docs(corpus, titles_list, labels, lambda doc: has_vector_representation(model, doc))
print("1st filter: Length of corpus:"+str(len(corpus))+", Length of titles_list:"+ str(len(titles_list))+", Length of labels:"+str(len(labels)))

# # Filter out any empty docs
corpus, titles_list, labels = filter_docs(corpus, titles_list, labels, lambda doc: (len(doc) != 0))
print("2nd filter: Length of corpus:"+str(len(corpus))+", Length of titles_list:"+ str(len(titles_list))+", Length of labels:"+str(len(labels)))

x = []
for doc in corpus: # append the vector for each document
    x.append(document_vector(model, doc))

"""# **After removing stop words and empty docs** <br>

---
241 docs removed <br>
1st filter: Length of corpus:422178, Length of titles_list:422178, Length of labels:422178 <br>
0 docs removed <br>
2nd filter: Length of corpus:422178, Length of titles_list:422178, Length of labels:422178 <br>
"""

vectorsForEachDocument = np.array(x) # list to array
labels = np.array(labels)
labels = labels.reshape(labels.shape[0],1)
vectorsForEachDocument.shape, labels.shape

"""# **Vectors for each title** 

---
A list of vectors of 300 dimensions each for all the titles and labels contain the respective labels<br>
Shape of these vectors is (422178, 300) <br>
Shape pf it respective labels (422178, 1)

# **Filter improper labels**

---
Filter out the data that has improper labels. Labels should only be of the following types. <br>
1. b: business
2. t: technology
3. e: entertainment
4. m: health
"""

# filter out data that has improper labels
possibleLabels = ["b","t","e","m"]
finalLabels = []
features = []
for i in range(len(labels)):
  if labels[i] in possibleLabels:
    finalLabels.append(labels[i])
    features.append(vectorsForEachDocument[i])
  if(len(finalLabels)==total):
    break


features = np.array(features)
labels = np.array(finalLabels)
labels = labels.reshape(labels.shape[0],1)
features.shape, labels.shape

"""# **Create a dataframe to shuffle it**

---
Create dataframe to shuffle it and split it.
"""

finalData = pd.DataFrame.from_records(features)
finalData.columns = range(1,301)
finalData["labels"] = labels
# finalData.to_csv('FinalData.csv')
data = finalData.sample(frac=1) #shuffles the data
labels = data["labels"]
labels = np.array(labels) 
labels = labels.reshape(labels.shape[0],1)
del data["labels"]
features = np.array(data)
features.shape, labels.shape

"""# Split the data
---
**Train Data**: Use 80% of the data for training purpose. <br>
**Test Data**: Use 20% of the data for testing purpose. <br>
**Features**: Use 300 dimensional vectors as features. It can be found in `vectorsForEachDocument`.<br>
**Labels**: Use the categories as labels. I can be found in `labels`.<br>
"""

train = int((80/100)*len(features))
trainFeatures, testFeatures = features[:train], features[train:]
trainLabels, testLabels = labels[:train], labels[train:]
trainFeatures.shape, trainLabels.shape, testFeatures.shape, testLabels.shape

"""**Shapes of the data** <br>
trainFeatures: (160000,300) <br>
trainLabels: (160000,1) <br>
testFeatures: (40000,300) <br>
testLabels: (40000,1) <br>

# **Train the logistic regression model** <br>
"""

tic = time.time()
logistic_Regression = LogisticRegression(multi_class="auto", solver="lbfgs", max_iter=1000)
logistic_Regression.fit(trainFeatures,trainLabels)
Y_predict = logistic_Regression.predict(testFeatures)
print(str((accuracy_score(testLabels,Y_predict)*100))+"%")
toc = time.time()
print("Time taken:"+str(toc-tic)+" seconds")

"""# **Outcomes of the training** <br>

---
|Train |Test |Dimensions |Accuracy |Time(sec) | Comments |
|---|---|---|---|---|---|
|40000|10000|300|86|12|Data randomly shuffled|
|80000|20000|300|78|53||
|40000|10000|300|74|12||
|80000|20000|300|78|53||
|24000|6000|300|73|11||
|160000|40000|300|72|53||
"""