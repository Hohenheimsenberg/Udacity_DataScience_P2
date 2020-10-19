import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads two csv files and merges them on a pandas dataframe

    Parameters:
    messages_filepath: path to the messages csv file
    categories_filepath: path to the categories csv file
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on='id')


def clean_data(df):
    """Splits the dataframe 'categories' column by ';' and creates a new column for each one. Then it drops the duplicate rows.

    Parameters:
    df: pandas dataframe
    """
    expanded_cat = df.categories.str.split(';',expand=True)
    
    row = expanded_cat.loc[0]
    category_colnames = []
    for x in row:
        x = x[:-2]
        category_colnames.append(x)
    expanded_cat.columns = category_colnames
    expanded_cat = expanded_cat.applymap(lambda x: x[-1])
    expanded_cat = expanded_cat.apply(pd.to_numeric)
    expanded_cat.head()
    
    df = df.drop(columns='categories')
    df = pd.concat([df, expanded_cat], axis=1)
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """Saves a dataframe to a sqlite database

    Parameters:
    df: pandas dataframe
    database_filename: path of the database
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Database', engine, index=False)  


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