## Using Natural Language Processing to Predict Programming Languange

### Webscraping and Natural Language Processing Project 

#### Presented by Chris Everts, Malachi Hale and Randy French


### Table of Contents
---

I.   [Project Overview             ](#i-project-overview)
1.   [Description                  ](#1-description)
2.   [Deliverables                 ](#2-deliverables)

II.  [Executive Summary  ](#ii-executive-summary)
1.   [Goals:                        ](#1-goals)
2.   [Key Findings:                 ](#2-key-findings)
3.   [Recommendations:              ](#1-recommendations)

III. [Project                      ](#iii-project)
1.   [Questions                    ](#1-questions)
2.   [Findings                     ](#2-findings)

IV. [Data Context                 ](#iv-data-context)
1.   [Data Dictionary              ](#1-data-dictionary)

V.  [Process                      ](#v-process)
1.   [Project Planning             ](#1-project-planning)
2.   [Data Acquisition             ](#2-data-acquisition)
3.   [Data Preparation             ](#3-data-preparation)
4.   [Data Exploration             ](#4-data-exploration)
5.   [Modeling & Evaluation        ](#5-modeling--evaluation)
6.   [Product Delivery             ](#6-product-delivery)

VI.   [Modules                      ](#vi-modules)

VII.  [Project Reproduction         ](#vii-project-reproduction)

<br>

<br>

### I. Project Overview
---

#### 1. Description

### The primary focus of the project was to build a model that can predict what programming language a repository is, given the text of the README file.


#### 2. Deliverables

- A well-documented Jupyter notebook that contains a report of our analysis, and link to that notebook. 
- A slideshow suitable for a general audience that summarizes our findings. Include well-labeled visualizations in your slides. 
- A video of your 5-minute presentation. Each team member should present a portion of the presentation.
- README file that details the project specs, planning, key findings, and steps to reproduce the project.


### II. Executive Summary
---

#### 1. Goals:

- For this project, we will be scraping data from GitHub repository README files. 
- To build a model that can predict what programming language a repository is, given the text of the README file.

#### 2. Key findings:

- We will demonstrate that we can use the Random Forest model on the lemmatized text data to predict with an accuracy greater than baseline the programming language that corresponds to each README file. 

#### 3. Recommendations:

- We were able to create a successful model that predicted better than baseline which programming language a README file was likely to accompany. There are, however, some ways that we may be able to fortify our model.

- In this project, we began with an initial dataset of 200 README files. To make our model even more robust in the future, we may explore using a larger sample of README files. 

- Furthermore, because natural language differs so signficantly by geographic location, it may be useful to build separate classification models based on the repository creator's location. 

---

### III. Project

#### 1. Questions

- What are the most common words in README files?
- Does the length of the README vary by programming language?
- Do different programming languages use a different number of unique words?
- What does the distribution of IDFs look like for the most common words?


We will use statistical testing (T-test) to determine if the average length of the README file for each language is significantly different than the length of the README file for other languages.

What are the most common words in README files?
- Overall, we found that words 'custom', 'data', 'use', and 'model' were the most frequently used words in all our README files.
- The most frequently used words differ when we calculate word frequency separately by language. However, the words 'customer' and 'data' tended to appear frequently in the README files for every language.
- Every language had a unique series of bigrams and trigrams that appeared most frequently in its corresponding README files.

Does the length of the README file differ by programming language?
- Using independent t-tests to compare the README lengths of one language to another, we found:
    - None of the languages have on average significantly different mean README lengths from one another.
- Using a one-sample t-test to compare each language's README length to the population mean, we found:
    - PHP is the only language that differs significantly in the length of its README files than the overall population.

Do different programming languages use a different number of unique words?
- Using independent t-test to compare the amount of unique words in the README files of different programming languages, we found:
    - Of all languages, only PHP and Jupyter Notebook differ significantly from each other in the amount of unique words used in README files.
 - Using a one-sample t-test to compare each language's README unique words to the population mean, we found:
     - Only PHP varied signficantly from the overall population in its amount of unique words used.

What does the distribution of IDFs look like for the most common words?
- Using our top 20 most common words calculated in Question 1, we calculated the IDF values for each of our top 20 most common words.
- The words in the top 20 with the highest IDF value were "png", "churn", "segment", and "magneto".


### 2. Findings
#### Our findings are:

- We constructed a model that used the natural language of the README file of each repository to predict the programming language of that repository.

- We used our most frequent language value, "Jupyter Notebook", as our baseline. This gave us a baseline accuracy of 0.36.
- We ran the models Naive Bayes, SVC, Decision Tree, Random Forest, K Nearest Neighbors, and Logistic Regression on:
    - lemmatized text data,
    - stemmed text data,
    - cleaned data, and
    - lemmatized bigrams.
- All of our models except Random Forest on lemmatized bigrams (Random_forest_tfidf2) and K Neartest Neighbors lemmatized bigrams (KNN_bigrams_tfidf2) performed with a higher accuracy than baseline on the validate dataset. Thefore, all of our models except the two mentioned are valid.
- Our best performing model on the validate dataset was the Random Forest on lemmatized text Random_forest_tfidf_lemmatized, with a score of nearly 0.85 on the validate dataset.

#### Our best performing model predicted the test dataset languages with 64% accuracy, outperforming baseline by 28% on the test dataset.

      ================ Random Forest Lemmatized =====================
      RandomForestClassifer()
      ------------- Test Scores ----------------- 

                  precision    recall  f1-score   support

      JavaScript       0.67      0.40      0.50         5
Jupyter Notebook       0.75      0.90      0.82        10
             PHP       0.67      1.00      0.80         8
          Python       0.00      0.00      0.00         5

        accuracy                           0.68        28
       macro avg       0.52      0.57      0.53        28
    weighted avg       0.58      0.68      0.61        28

### IV. Data Context
---

#### 1. Data Dictionary

Following acquisition and preparation of the initial data acquitions, the contained values are defined along with their respective data types.

| Feature         | Datatype   | Description                                                                                                                                                                                              |
|:----------------|:-----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| repo            | object     | End of the URL string of the location of the repository on GitHub. The GitHub user followed by the name of the repostiory.                                                                               |
| language        | object     | The programming language in which the code is written. Our top four most frequently used languages are Jupyter Notebook, PHP, JavaScript, and Python.                                                    |
| readme_contents | object     | The contents of the READme file of the repostory.                                                                                                                                                        |
| clean           | object     | The contents of the READme file of the repository after a bsic cleaning. All characters are loewr case, unicode data is normalize and encoded as ASCII, and all non-alphanumeric characters are removed. |
| stemmed         | object     | The contents of the cleaned READme file stemmed by word.                                                                                                                                                 |
| lemmatized      | object     | The contents of the cleaned READme file lemmatized                                                                                                                                                       |


### V. Process
---
#### 1. Project Planning
✓ 🟢 **Plan** ➜ ☐ _Acquire_ ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Build this README containing:
    - Project overview
    - Initial thoughts
    - Project summary
    - Instructions to reproduce
- [x] Plan stages of project and consider needs versus desires

#### 2. Data Acquisition
✓ _Plan_ ➜ 🟢 **Acquire** ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Obtain initial data and understand its structure
    - Obtain data from GitHub using an acquire.py script to gather data via the GitHub API
- [x] Remedy any inconsistencies, duplicates, or structural problems within data

#### 3. Data Preparation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ 🟢 **Prepare** ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Address missing or inappropriate values
- [x] Consider and create new features as needed

#### 4. Data Exploration
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ 🟢 **Explore** ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Graph top 20 most common words
- [x] WordCloud for top 20 most common words
- [x] Split data into `train`, `validate`, and `test`
- [x] Visualize top 20 most word count with plots
- [x] Visualize the top 20 common bigrams 
- [x] Visualize the top 20 common trigrams 
- [x] Perform statistical tests
- [x] Decide upon features and models to be used

#### 5. Modeling & Evaluation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ 🟢 **Model** ➜ ☐ _Deliver_

- [x] Establish baseline prediction
- [x] Create, fit, and predict with models
    - Create different models
- [x] Evaluate models with out-of-sample data
- [x] Utilize best performing model on `test` data
- [x] Summarize, visualize, and interpret findings

#### 6. Product Delivery
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ ✓ _Model_ ➜ 🟢 **Deliver**
- [x] Prepare Jupyter Notebook of project details through data science pipeline
    - Python code clearly commented when necessary
    - Sufficiently utilize markdown
    - Appropriately title notebook and sections
- [x] With additional time
- [x] Proof read and complete README and project repository

### VI. Modules
---

The created modules used in this project below contain full comments and docstrings. See 'project reproduction' for the steps.

- acquire.py
- explore.py
- prepare.py
- model.py
- Final_Notebook.ipynb


### VII. Project Reproduction
---

To reproduce our project:

- Read this README.md file.

- Download and import the 'acquire.py', 'explore.py', 'prepare.py', 'model.py' and 'Final_Notebook.ipynb' files in our repository.

- Import all the other libraries listed in the 'Import Libraries' section. 

- Run the 'Final_Notebook.ipynb' file.

When using any function housed in the created modules above, ensure full reading of comments and docstrings to understand its proper use and passed arguments or parameters.


[[Return to Top]](#Using-Natural-Language-Processing-to-Predict-Programming-Languange)

