# Dataset
Raw data set can be downloaded from the link below. <br>
[https://www.kaggle.com/arjunchandrasekhara/news-classification/data](https://www.kaggle.com/arjunchandrasekhara/news-classification/data)

# Pre trained word2vec model
```
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
```
This model is trained by google and provides a 300 dimensional vector for each word.

# Features

50,000 clean records are used features. Clean records means the records are without stop words, out of vocab words, punctuation marks. These records have clean labels that belong to either of the category of b(business), t(technology), m(medicine), e(entertainment). 
The records have a 300 dimensional vector for each title and its respective label has single char.

Shape of Features is $50000\times300$ and shape of labels is $50000\times1$.
The ratio of train:test set is 4:1. i.e. 40,000 records used for training the model and 10,000 records used for validation of the model.

Outcomes the logistic regression model are as follows. <br>
|Train |Test |Dimensions |Accuracy |Time(sec) | Comments |  
|:---:|:---:|:---:|:---:|:---:|:---:|  
|40000|10000|300|85|12|Data randomly shuffled|  
|80000|20000|300|78|53||  
|40000|10000|300|74|12||  
|80000|20000|300|78|53||  
|24000|6000|300|73|11||  
|160000|40000|300|72|53||  