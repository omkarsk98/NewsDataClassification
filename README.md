# Objective
Classify news titles into categories like business, technology, etc using pretained google's word2vec model and applying logistic regression.

# Dataset

Raw data set can be downloaded from the link below. <br>
[https://www.kaggle.com/arjunchandrasekhara/news-classification/data](https://www.kaggle.com/arjunchandrasekhara/news-classification/data)

# Pre trained word2vec model

```
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
```

This model is trained by google and provides a 300 dimensional vector for each word.

# Project Details

50,000 clean records are used as features. Clean records means the records are without stop words, out of vocab words and punctuation marks. These records have clean labels that belong to either of the category of b(business), t(technology), m(medicine), e(entertainment).
The records have a 300 dimensional vector for each title and its respective label has single char.

Shape of Features is 50000\*300 and shape of labels is 50000\*1.
The ratio of train:test set is 4:1. i.e. 40,000 records used for training the model and 10,000 records used for validation of the model.

# Results
Outcomes the logistic regression model are as follows. <br>

| Train  | Test  | Dimensions | Accuracy | Time(sec) |        Comments        |
| :----: | :---: | :--------: | :------: | :-------: | :--------------------: |
| 40000  | 10000 |    300     |    86    |    12     | Data randomly shuffled |
| 80000  | 20000 |    300     |    78    |    53     |      Not shuffled      |
| 40000  | 10000 |    300     |    74    |    12     |      Not shuffled      |
| 80000  | 20000 |    300     |    78    |    53     |      Not shuffled      |
| 24000  | 6000  |    300     |    73    |    11     |      Not shuffled      |
| 160000 | 40000 |    300     |    72    |    53     |      Not shuffled      |

Precision and Recall are as follows <br>
For labels in sequence as ['b', 't', 'e', 'm']<br>
precision: [82.94729775 83.25883787 89.44781729 87.46594005]<br>
recall: [84.79752917 80.92417062 91.4843288  82.16723549]<br>

ROC curves are as follows<br>
![ROC for Business]('./ROC/business.png')
![ROC for Business]('./ROC/technology.png')
![ROC for Business]('./ROC/entertainment.png')
![ROC for medicine]('./ROC/medicine.png')