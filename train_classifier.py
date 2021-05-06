# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import warnings

warnings.simplefilter('ignore')



def load_data(database_filepath):
    
        '''
    Input:
        database path that contained the cleaned data
        
    Output:
        X that has the meassages
        y that has the features
        category_names that has the names of features
        
        '''
        
        engine = create_engine('sqlite:///{}'.format(database_filepath)) 
        df = pd.read_sql('select * from ETL', engine)
        X = df['message']
        y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
        category_names = y.columns.values
        
        return X, y, category_names


def tokenize(text):

    '''
     Input:
        text: original text for messages
        
    Output:
        clean_tokens: tokenized text for model
        
        '''

    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # Stem word tokens and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    stemmed = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return stemmed


def build_model():
    
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    parameters = {
              'clf__estimator__n_estimators':[10, 25], 
              'clf__estimator__min_samples_split':[2, 5, 10]}
    
    model =  GridSearchCV(pipeline, param_grid = parameters)
    
    return model


def evaluate_model(model, X_test, y_test, category_names):
    
    '''
    Evaluate the model with precision, Recall, F1-score and support.
    
    '''
    
    y_pred = model.predict(X_test)
    y_pred_df_cv = pd.DataFrame(y_pred, columns = category_names)

    for column in y_test.columns:
        print('_____________________________________________________\n')
        print('Attribute: {}\n'.format(column))
        print(classification_report(y_test[column],y_pred_df_cv[column]))    



def save_model(model, model_filepath):
    
    
    '''
    Saving the model using pickle
    
    '''
        
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()