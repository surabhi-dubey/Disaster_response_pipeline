# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import nltk
nltk.download(['punkt', 'wordnet'])
import pickle as pkl
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """
    Load Data from dataset
    """
    db_name = "sqlite:///" +  database_filepath
    engine = create_engine(db_name)
    df = pd.read_sql_table('Messages_Categories', engine) 
    df.dropna(inplace=True)
    
    # Splitting into ip-op for model
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y

def tokenize(text):
    """
    Tokenize to parse text data to the model
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Use Pipeline API to setup feature vectors and classifier
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75),
        'vect__max_features': (None, 5000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50],
        'clf__estimator__min_samples_split': [2]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test):
    """
    Evaluate learned model on test set 
    """
    # Predicting The model on test data
    y_pred = model.predict(X_test)
    
    # Creating dataframe to strore results
    report = pd.DataFrame(columns=['Class', 'f_score', 'precision', 'recall'])
    
    i = 0
    
    # Running for each column
    for col in Y_test.columns:
        precision, recall, f_score, _ = precision_recall_fscore_support(Y_test[col], y_pred[:, i], average='weighted')
        report.at[i+1, 'Class'] = col
        report.at[i+1, 'precision'] = precision
        report.at[i+1, 'recall'] =  recall
        report.at[i+1, 'f_score'] = f_score
        i += 1
        
    print('Classification Report', '\n')
    print(report)
    
    
def save_model(model, model_filepath):
    """
    Save trained model to disk
    """
    pkl.dump(model, open(model_filepath, 'wb'))
 
    # load the model from disk
    #loaded_model = pickle.load(open(filename, 'rb'))

def main():
    """
    Main Runner Fn. with flow of the ML pipeline
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        
        # Checking tokens built
        for message in X:
            tokens = tokenize(message)
            print(message)
            print(tokens, '\n')
            break
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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