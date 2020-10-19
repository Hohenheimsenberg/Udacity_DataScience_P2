import sys
import pandas as pd 
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """Loads a sqlite database in to parameters and target dataframes

    Parameters:
    database_filepath: path of the database

    Returns:
    labels 
    categories
    category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Database', con=engine)
    X = df['message']
    y = df.loc[:, 'related':'direct_report']
    return X, y, y.columns


def tokenize(text):
    ''' 
    Tokenizes a text string using nltk's workd_tokenize. Then it removes english stop words. And finally lemmatizes the tokens with WordNetLemmatizer.

    Parameters:
    text: text to tokenize

    Returns:
    words: tokenized and lemmatized text array
    '''
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w).lower().strip() for w in words ]
    
    return words


def build_model():
    '''
    Builds a machine learning pipeline for text classification.

    Returns:
    pipeline: CountVectorizer -> TfidfTransformer -> MultiOutputClassifier

    '''
    return Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Iterates a dataframe and evaluates a machine learning model using sklearns classification_report

    Parameters:
    model: sklearn model
    X_test: testing dataset
    Y_test: testing dataset
    category_names: categories
    '''
    y_pred = model.predict(X_test)
    for i in range(category_names.size):
        print(category_names[i], ':', classification_report(Y_test.iloc[:, i], y_pred[:, i]))



def save_model(model, model_filepath):
    '''
    Saves a sklearn model to a file

    Parameters:
    model: sklearn model
    model_filepath: path to save the model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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