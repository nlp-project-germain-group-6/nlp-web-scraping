import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score, plot_confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

#################### ~~ Xy splitting function ~~ ####################
def nlp_X_train_split(X_data, y_data):
    '''
    This function is designed for splitting data during an NLP pipeline
    It takes in the X_data (already transformed by your Vectorizer)
    y_data (target)
    And performs a train validate test X/y split (FOR MODELING NOT EXPLORATION)
    This is a one shot for doing train validate test and x/y split in one go
    
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    
    Returns 6 dfs: X_train, y_train, X_validate, y_validate, X_test, y_test
    '''
    X_train_validate, X_test, y_train_validate, y_test = train_test_split(X_data, y_data, 
                                                                          stratify = y_data, 
                                                                          test_size=.2, random_state=123)
    
    X_train, X_validate, y_train, y_validate = train_test_split(X_train_validate, y_train_validate, 
                                                                stratify = y_train_validate, 
                                                                test_size=.3, 
                                                                random_state=123)
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

############## Function to test a model adds accuracy score to a dataframe
############ you must have a score dataframe set up already to use this function
##### see docstrings on how to create

def test_a_model(X_train, y_train, X_validate, y_validate, model, model_name, score_df):
    '''
    Function takes in X and y train
    X and y validate (or test) 
    A model with it's hyper parameters
    And a df to store the scores 
    - Set up an empty dataframe with score_df first
    - score_df = pd.DataFrame(columns = ['model_name', 'train_score', 'validate_score'])
    '''
    this_model = model

    this_model.fit(X_train, y_train)

    # Check with Validate

    train_score = this_model.score(X_train, y_train)
    
    validate_score = this_model.score(X_validate, y_validate)
    
    model_dict = {'model_name': model_name, 
                  'train_score': train_score, 
                  'validate_score':validate_score}
    score_df = score_df.append(model_dict, ignore_index = True)
    
    return score_df

########### Evaluation metrics printing function

def print_metrics(model, X, y, pred, class_names, set_name = 'This Set'):
    '''
    This function takes in a model, 
    X dataframe
    y dataframe 
    predictions 
    Class_names (aka ['Java', 'Javascript', 'Jupyter Notebook', 'PHP'])
    and a set name (aka train, validate or test)
    Prints out a classification report 
    and confusion matrix as a heatmap
    To customize colors change insdie the function
    - IMPORTANT change lables inside this function
    '''
    
    
    print(model)
    print(f"~~~~~~~~{set_name} Scores~~~~~~~~~")
    print(classification_report(y, pred))
    
    #purple_cmap = sns.cubehelix_palette(as_cmap=True)
    purple_cmap = sns.color_palette("light:indigo", as_cmap=True)
    
    with sns.axes_style("white"):
        matrix = plot_confusion_matrix(model,X, y, display_labels=class_names, 
                                       cmap = purple_cmap)
        plt.grid(False)
        plt.show()
        print()


######### This function makes models and prints metrics (uses above function)
#### can run in a loop to loop through models 

def make_models_and_print_metrics(model, model_name, X_train, y_train, X_validate, y_validate, class_names):
    '''
    This function takes in a model object,
    Name for the model (for vis purposes)
    X_train, y_train
    X_validate and y_validate
    and the names of your classes (aka category names)
    Uses print metrics function 
    '''
    model.fit(X_train, y_train)

    #predict for train and validate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_validate)
    
    print(f'                   ============== {model_name} ================           ')
    #see metrics for train
    print_metrics(model, X_train, y_train, train_pred, class_names, set_name='Train')
    #print metrics for validate
    print_metrics(model, X_validate, y_validate, val_pred, class_names, set_name='Validate')
    print('-------------------------------------------------------------------\n')


######### Function for evaluating the final Test data ################ 

def make_models_and_print_metrics_test_data(model, model_name, X_train, y_train, X_test, y_test, class_names):
    '''
    This function takes in a model object,
    Name for the model (for vis purposes)
    X_train, y_train
    X_test and y_test
    and the names of your classes (aka category names)
    Uses print metrics function 
    Use this function on the final test data set. 
    '''
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    
    print(f'                   ============== {model_name} ================           ')
    #print metrics for Test
    print_metrics(model, X_test, y_test, test_pred, class_names, set_name='Test')
    print('-------------------------------------------------------------------\n')