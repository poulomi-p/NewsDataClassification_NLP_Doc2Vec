# Multiclass Text Classification: News article data

This NLP project is a supervised learning problem where we try to predict the category of a news article. Since each article belongs to exactly one of the 5 categories used here, it is a multiclass classification problem.
## Data Collection
1. I used a data set named BBC news data
2. Read in the data as a pandas dataframe
3. The data set has 5 categories of data (business, sport, tech, politics and entertainment) and has more than 2000 articles with 2 features (text data and category/label)

## Data pre-processing and cleaning
1. Dropping duplicate rows
2. Plotting the count of articles in each category and the data was not imbalanced
3. Counting and plotting the number of words in each article
4. Convert text into lowercase, remove extra spaces, tags, punctuations, digit and special characters from the text.
5. I used WordCloud to visualize the most frequent words present in each category
6. Further cleaning of text, that is, stop word removal and tokenization
7. Finally, label encoding, to convert the target labels (text) to numeric. I saved the preprocessed data in a csv file.

## Feature extraction, training and testing, model building
1. Split the cleaned text into training and testing sets
2. Performing feature extraction using Tf-Idf (term frequency inverse document frequency)
3. Model building: I used supervised learning algorithms such as Random Forest Classifier, Logistic regression, KNN, Decision tree, Naive Bayes and SVM, of these Logistic regression gave the best accuracy 

## Hyper parameter tuning
1. I used hyperparameter tuning on Logictic regression algorithm, and that slightly increased the accuracy
 
## Using Doc2Vec algorithm instead of Tf-Idf
1. Using the cleaned text csv file, converting training and testing data into the Gensim format
2. Initializing the Doc2Vec model and training it for a few epochs
3. Getting the vector representations of the train and test sets
4. Finally, using Logistic regression model again with the new train and test vectors and achieving an increase of 1% in accuracy 
