import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """Loads the data for categories and messages
    Returns:
        df (DataFrame): output dataframe of categories and messages
    Args:
        messages_filepath (string): filepath for messages file
        categories_filepath (string): filepath for categories file
    """    
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner',on=['id'])
    return df


def clean_data(df):
    # create a dataframe of the individual category columns
    categories = df.categories.str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    #convert category values to 0 and 1s
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df=df.drop('categories', axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(left= df, right=categories, left_on=df.index, right_on=categories.index).drop('key_0', axis=1)
    # drop duplicates
    df.drop_duplicates(subset=['id'],inplace=True,keep='last')
    return df
 

def save_data(df, database_filename):
    
    """Saves the data to asqlite db
    Args:
        df (DataFrame): cleaned data to be saved 
        database_filename (string): path to create database
    """    
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('CleanedMessages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()