import pandas as pd
from gensim import models
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

main_data = pd.read_csv('News_Final.csv')

# Grab all the titles
article_titles = main_data['TITLE']
# Create a list of strings, one for each title
print("Creating a list of titles")
titles_list = [title for title in article_titles]
print("Created a list of titles\n")

# Collapse the list of strings into a single long string for processing
print("Creating a big string")
big_title_string = ' '.join(titles_list)
print("Created a big string\n")

# Tokenize the string into words
print("Getting tokens")
tokens = word_tokenize(big_title_string)
print("Got", len(tokens), " tokens")

# Remove non-alphabetic tokens, such as punctuation
print("Removing stop words")
words = [word.lower() for word in tokens if word.isalpha()]
stop_words = set(stopwords.words('english'))
words = [word for word in words if not word in stop_words]
print("Completed stop word removal")

# Print first 10 words
print("First 10 words as:",words[:10])

print("Getting model")
# Load word2vec model (trained on an enormous Google corpus)
model = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True) 
# Check dimension of word vectors
print("Dimensions of the model",model.vector_size)

print("getting vector list")
# Filter the list of vectors to include only those that Word2Vec has a vector for
vector_list = [model[word] for word in words if word in model.vocab]

print("Create a list of the words corresponding to these vectors")
# Create a list of the words corresponding to these vectors
words_filtered = [word for word in words if word in model.vocab]

print("Zip the words together with their vector representations")
# Zip the words together with their vector representations
word_vec_zip = zip(words_filtered, vector_list)

# Cast to a dict so we can turn it into a DataFrame
word_vec_dict = dict(word_vec_zip)
df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
df.head(3)

print("Saving the dataframe to csv")
df.to_csv('Words_vs_Vectors.csv', header=False, index=False)
print("Saved csv")