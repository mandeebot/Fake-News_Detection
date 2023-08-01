import os
import re
import pandas as pd
from nltk.corpus import stopwords

data = pd.read_csv('data_combined.csv')

stop_words = set(stopwords.words('english'))

def read_testdata(file):
     test_data = pd.read_csv(file)
     test_data = test_data[['id','title','content']]
     #val_data['stance_num'] = data['Stance'].map({'agree':0,'disagree':1,'discuss':2,'unrelated':3}).astype(int)

def clean_text(col):
    #transform text into lower case
    #use regular expressions to remove unwanted characters
    data.loc[:,col] = data[col].apply(lambda x:str.lower(str(x)))
    data.loc[:,col] = data[col].apply(lambda x:' '.join(re.findall('[\w]+',x)))

def remove_stopwords(s):
    return ' '.join(word for word in s.split() if word not in stop_words)

def stopword_cleaner(col):
        try:
            data[col] = data[col].apply(lambda x:remove_stopwords(x))
        except:
            print("Error: ",col)

def process():
      clean_text('title')
      clean_text('content')
      stopword_cleaner('title')
      stopword_cleaner('content',)

      data.to_csv('data_combined2.csv',index=False)


if __name__ == "__main__":
    process()