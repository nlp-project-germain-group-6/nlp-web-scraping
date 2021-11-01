import nltk

import unicodedata
import re
import json

from sklearn.model_selection import train_test_split

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd
import numpy as np


def basic_clean(string):
    '''
    This function takes in a string and normalizes it for nlp purposes
    '''
    # lowercase the string
    string = string.lower()

    # return normal form for the unicode string, encode/remove ascii
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8')
    
    # breaks down the string by keeping alphabet letters, numbers, apostraphes and spaces
    string = re.sub(r"[^a-z0-9\s]", '', string)
    
    return string


def tokenize(string):
    '''
    This function takes in a string and tokenizes it
    '''
    # create the tokenizer
    tokenizer = nltk.tokenize.ToktokTokenizer()
    
    # use the tokenizer, return as a string
    string = tokenizer.tokenize(string, return_str = True)
    
    return string

def stem(text):
    '''
    This function takes in a text and stems the words to their original stem
    '''
    
    # create a porter stemmer
    ps = nltk.porter.PorterStemmer()
    
    # loop through the text to stem the words
    stems = [ps.stem(word) for word in text.split()]
    
    # return back together
    stems = ' '.join(stems)
    
    return stems


def lemmatize(text):
    '''
    This function takes in a text and changes the words back to their root (lemmatize)
    '''
    
    # create the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    # loop through the list to split and lemmatize
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    
    # return back together
    lemmas =' '.join(lemmas)
    
    return lemmas


def remove_stopwords(string, extra_words = [], exclude_words = []):
    '''
    This function takes in a string
    And returns the string with the English stopwords removed
    Additional stopwords can be added to extra_words (list)
    or words to exclude from stopwords can be added to exclude_words (list)
    
    -- This might break if the excluded words aren't in the stopwords list
    '''
    # define stopwords list      
    stopwords_list = stopwords.words('English')
    
    # add or remove words based on arguments
    stopwords_list = set(stopwords_list) - set(exclude_words) # the set removes words
    
    stopwords_list = stopwords_list.union(set(extra_words))
        
    # remove stopwords from string
    # turn string into list
    words = string.split()
    
    # remove the stopwords
    filtered_words = [w for w in words if w not in stopwords_list]
    
    # turn back into a string
    new_string = ' '.join(filtered_words)
    
    return new_string

def remove_urls(text):
    '''
    This function removes urls from the string 
    '''
    res = []
    for i in text.strip().split():
        if not re.search(r"(https?)", i):  
            res.append(re.sub(r"[^A-Za-z\.]", "", i).replace(".", " "))   
    return " ".join(map(str.strip, res))

def remove_unicode_text(df):
    for row in df.readme_contents:
    
        row = re.sub(r'\<[^>]*\>', '', row)
    
    return df


################## ~~~~~~ Mother Prep Function ~~~~~~ ##################

def prepare_nlp_data(df, content = 'content', extra_words=[], exclude_words=[]):
    '''
    This function take in a df and the content (in string) for the column 
    with an option to pass lists for additional stopwords (extra_words)
    and an option to pass words to exclude from stopwords (exclude words)
    returns a df with the  original text, cleaned (tokenized and stopwords removed),
    stemmed text, lemmatized text.
    '''
    df['clean'] = df[content].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, exclude_words=exclude_words)
    
    df['stemmed'] = df['clean'].apply(stem)
    
    df['lemmatized'] = df['clean'].apply(lemmatize)
    
    return df


def is_chinese(texts):
    '''
    This function takes in a dataframe and return true if the scanned text is in chinese
    '''
    if re.search("[\u4e00-\u9FFF]", texts):
            return True

def is_foreign(texts):
    '''
    This function takes in a dataframe and return true if the scanned text is in Chinese, Korean, or Japanese.
    '''
    if re.search("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f\u3131-\uD79D]", texts):
            return True
        
        

def get_top_4_languages(df):
    '''
    This function takes in a dataframe and returns the top four
    programming languages found in the data
    '''
    top_4_list = list(df.language.value_counts().head(4).index)
    mask = df.language.apply(lambda x: x in top_4_list)
    df = df[mask]
    return df


def drop_unneeded_data(df):
    '''
    This function takes in the repo dataframe
    Drops any rows with nulls
    Drops any rows that are chinese
    Drops all rows that aren't in the top 4 languages
    '''
    
    for i in range(len(df)):
        df['contents'] = df.readme_contents  
        df.contents[i] = re.sub(r'\<[^>]*\>', '', df.contents[i])
        for i in range(len(df)):
            res = []
        for j in df.contents[i].strip().split():
            if not re.search(r"(https?)", j):  
                res.append(re.sub(r"[^A-Za-z\.]", "", j).replace(".", " "))   
                df.contents[i] = " ".join(map(str.strip, res))
    df = df.dropna()
    df = df[df.readme_contents.apply(is_chinese) !=True]
    df = get_top_4_languages(df)
    df = df.reset_index().drop(columns = 'index')
           
    return df


def split_data(df):
    '''
    This function takes in a dataframe and splits it into train, test, and 
    validate dataframes for my model
    '''

    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, stratify=df.language)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, stratify=train_validate.language)

    print('train--->', train.shape)
    print('validate--->', validate.shape)
    print('test--->', test.shape)
    return train, validate, test