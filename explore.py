import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# for colormap tools
from matplotlib import cm
import pandas as pd
import nltk
import unicodedata
import re
import seaborn as sns
from wordcloud import WordCloud


def get_word_counts_series(df, column):
    '''
    This function takes in a dataframe
    and the column you want to create the word counts of
    returns a series of the words and their counts
    You can get the top 20 or whatever from that later
    '''
    words = ' '.join(df[column])
    
    words_list = words.split()
    
    word_counts = pd.Series(words_list).value_counts()
    
    return word_counts


def plot_overlap_stacked_bar(word_counts, category, num_top = 20, cmap = None):
    '''
    This function takes in word_counts df
        - Must have counts for each category as well as a category named 'all'
    category you want to sort by (aka top 20 words in java readmes)
    num_top is how many words you want to see the proportion of, default = 20
    Default colors are tab10 but you can customize that
    
    for cmap use 'viridis'
    
    '''
    plt.figure(figsize=(16, 9))
    plt.rc('font', size=16)
    # axis=1 in .apply means row by row
    (word_counts.sort_values(by='all', ascending=False)
     .head(num_top)
     .apply(lambda row: row / row['all'], axis=1)
     .drop(columns='all')
     .sort_values(by=category)
     .plot.barh(stacked=True, width=1, ec='lightgrey', cmap = cmap, alpha = 1))
    plt.legend(bbox_to_anchor= (1.03,1))
    plt.title(f'% of most common {num_top} {category} Readme Words\n')
    plt.xlabel('\nProportion of Overlap')
    # make tick lables display as percentages!! 
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter('{:.0%}'.format))
    
    plt.show()

############# Bigram bar graph and word cloud #############
    
def bigram_count_word_cloud(words_list, top_num = 20, title_name = None):
    '''
    This function takes in a words_list
    Creates bigrams
    Plots the counts on a bar chart and a wordcloud 
    Optional arguements to change customization
    - top_num: default 20, shows most common number of bigrams
    '''

    # create bigrams
    ngrams = pd.Series(nltk.bigrams(words_list)).value_counts().head(top_num)
    
    # set up figuresize
    plt.figure(figsize = (20, top_num/2.5))
    
    # plot bigrams on left subplot
    plt.subplot(1, 2, 1)
    ngrams.sort_values(ascending = True).plot.barh(color = '#29af7f', alpha = .7, width = .9)
    plt.title(f'Top {top_num} Bigrams: {title_name}')
    
    # create dictionary of words from the bigrams
    data = {k[0] + ' ' + k[1]: v for k, v in ngrams.to_dict().items()}
    
    # create wordcloud image
    img = WordCloud(background_color='white', width=400, height=400).generate_from_frequencies(data)
    
    # plot worcloud on right subplot
    plt.subplot(1, 2, 2)
    # show image
    plt.imshow(img)
    plt.axis('off')
    plt.title("Word Cloud", font = 'Arial', fontsize= 20)
    plt.show()
    
    ############# Trigram bar graph and word cloud #############
    
def trigram_count_word_cloud(words_list, top_num = 20, title_name = None):
    '''
    This function takes in a words_list
    Creates bigrams
    Plots the counts on a bar chart and a wordcloud 
    Optional arguements to change customization
    - top_num: default 20, shows most common number of bigrams
    '''

    # create trigrams
    ngrams = pd.Series(nltk.trigrams(words_list)).value_counts().head(top_num)
    
    # set up figuresize
    plt.figure(figsize = (20, top_num / 2.5))
    
    # plot trigrams on left subplot
    plt.subplot(1, 2, 1)
    ngrams.sort_values(ascending = True).plot.barh(color = '#29af7f', alpha = .7, width = .9)
    plt.title(f'Top {top_num} Trigrams: {title_name}')
    
    # create dictionary of words from the bigrams
    data = {k[0] + ' ' + k[1] + ' ' + k[2]: v for k, v in ngrams.to_dict().items()}
    
    # create wordcloud image
    img = WordCloud(background_color='white', width=400, height=400).generate_from_frequencies(data)
    
    # plot worcloud on right subplot
    plt.subplot(1, 2, 2)
    # show image
    plt.imshow(img)
    plt.axis('off')
    plt.title("Word Cloud", font = 'Arial', fontsize= 20)
    plt.show()