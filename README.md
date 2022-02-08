## Financial News Sentiment Analysis

Sentiment analysis is the interpretation and classification of **emotion** (positive, negative and neutral) within text data using text analysis techniques. Sentiment analysis tools allow businesses to identify customer sentiment toward products, brands or services in online feedback.

### Need for Sentiment analysis

* Sentiment analysis tools allow businesses to identify customer sentiment toward products, brands or services in online feedback.
* Understanding people’s emotions is essential for businesses since customers are able to express their thoughts and feelings more openly than ever before.
* Using sentiment analysis to automatically analyze 4,000+ financial news stories can help you find out if the news is positive or negative.
* Real-Time Analysis Sentiment analysis can identify critical issues in real-time

### Rule-Based Approach

A rule-based system uses a set of human-crafted rules to help identify subjectivity, polarity, or the subject of an opinion. These rules may include various techniques developed in computational linguistics, such as stemming, tokenization, part-of-speech tagging parsing and lexicons (i.e. lists of words and expressions).

### Sentiment Analysis Process

#### Step 1: Read the Dataset and EDA
##### 
we try to give a little explore the data that can be explored is how many distributions there are in the sentiment of the data.


#### Step 2: Pre-Processing of data
##### REMOVING PUNCTUATIONS
Using regular expression(regex), remove punctuation, hashtags and @-mentions from each tweet.</br>
```def remove_punct(text): text_nopunct = ' ' text_nopunct = re.sub('['+string.punctuation+']', ' ', text)return text_nopunct```

##### TOKENIZATION
In order to use textual data for predictive modeling, the text must be parsed to remove certain words – this process is called tokenization.</br>
```tokens = [word_tokenize(sen) for sen in df.text_new] ```

##### STOPWORD
Stopwords are the English words which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence. For example, the words like the, he, have etc.</br>
```
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
```

##### LEMMATIZATION
Lemmatisation in linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form.</br>
```
from nltk.corpus import stopwords
stoplist = stopwords.words('english')

###replaces [running, ran, run] with run
```

##### COUNT VECTORIZATION
Scikit-learn’s CountVectorizer is used to convert a collection of text documents to a vector of term/token counts.</br>
```count_vect = CountVectorizer(stop_words='english')```

##### TFIDF TRANSFORMER
Tf-idf transformers aim to convert a collection of raw documents to a matrix of TF-IDF features.</br>
```transformer = TfidfTransformer(norm='l2',sublinear_tf=True)```

#### Step 3: Fitting and training the model
##### Implemented Algorithms
###### DECISION TREES
A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes).
###### RANDOM FOREST CLASSIFIER
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes or mean prediction of the individual trees.
###### K-NEAREST NEIGHBOURS
In pattern recognition, the k-nearest neighbors algorithm is a non-parametric method proposed by Thomas Cover used for classification and regression.
###### LOGISTIC REGRESSION
Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression).
###### SUPPORT VECTOR MACHINE
In machine learning, support-vector machines are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis.

#### Step 4: Model prediction and result comparison
###### DECISION TREES
```
accuracy_score(y_test,predDT)
output: 0.67
```
###### RANDOM FOREST CLASSIFIER
```
accuracy_score(y_test,predRF)
output: 0.75
```

###### LOGISTIC REGRESSION
```
accuracy_score(y_test,predLR)
output: 0.73
```
###### SUPPORT VECTOR MACHINE
```
accuracy_score(y_test,predSVM)
output: 0.71
```
#### Conclusion
* Among all other techniques used, Random Forest Classifier has performed best with the highest accuracy. One reason why RF works well is because the algorithm can look past and handle the missing values in the news. 


### Limitations of Sentiment Analysis
* One of the disadvantages of using the lexicon is that people express emotions in different ways. Some may be over-expressing in a statement.
* Multilingual sentiment analysis.
* Making the model automatic. Automatic methods, contrary to rule-based systems, don't rely on manually crafted rules, but on machine learning techniques. A sentiment analysis task is usually modeled as a classification problem, whereby a classifier is fed a text and returns a category
* Can take emoticons into account to predict better.
