# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle
import joblib



def load_data(database_filepath):
    
    """Loads data from sqlite database
    Args:
        database_filepath (string): File path of the database
    Returns:
        X (DataFrame): Predictor Variables (Table)
        Y (DataFrame): Target Variable (Table)
        y columns (Series) 
    """    
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name= 'CleanedMessages', con = engine)
    X = df['message']
    y = df.drop(['id', 'genre', 'original', 'message'], axis=1)
    return X,y,y.columns


def tokenize(text):
    """Tokenizes the input text. This includes removing the punctuations, 
     non alpha-numeric characters, splitting in words, 
    removing stop words and lemmatization.
    Args:
        text (string): Text that will be tokenized
    Returns:
        [list]: Tokenized text
    """    
    
    text="".join([c for c in text if c not in string.punctuation])
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords.words('english'):
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    
    """Builds the model pipeline including CountVectorizer,TfidfTransformer
    and MultiOutputClassifier
    
    Returns:
        [model] : RandomForestClassifier Model with Gridsearch
    """ 
    
    pipeline = pipeline = Pipeline(
    [
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ]
    )
    parameters = {
        #'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_leaf': [1, 4]
    }
    model = GridSearchCV(pipeline, param_grid=parameters , verbose=10, n_jobs=4)
    
    return model
    

def evaluate_model(model, X_test, y_test, category_names):
    
    """Evaluates the model performance on X_test. Prints the best parameters and 
    the classification report for each categories.
    
    Args:
        model (model): model that will be evaluated
        X_test (DataFrame): Variables for prediction of test data
        Y_test ([type]): Target variables of the testing data
        category_names ([type]): Categories for evaluation
    """ 
    
    print(f'Best parameters: {model.best_params_}')
    y_pred = model.predict(X_test)
    for p, column in enumerate(y_test.columns):
        report = classification_report(y_test.iloc[:, p], pd.DataFrame(y_pred).iloc[:, p])
        print('Column:'+ column)
        print(report)


def save_model(model, model_filepath):
    """saves model to wanted file path
    Args:
        model (model): model that wants to be saved
        model_filepath (string):path to save model
    """    
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