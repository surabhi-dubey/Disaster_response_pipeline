import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Read input csv files and merge them
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on='id')
    
    return messages, categories, df

def clean_data(categories, df):
    """
    Clean the data by 
    - adding new category columns to the final dataframe
    - dropping duplicates
    """
    # split the values in the categories column on the ; character so that each value becomes a separate column.
    categories = categories['categories'].str.split(';', expand = True)
    
    # using first row of categories dataframe to create column names for the categories data. 
    row = categories.iloc[0]

    category_colnames = []
    for i in range(len(row)): 
        category_colnames.append(row[i][:-2])
    print(category_colnames)
    
    # rename the columns of 'categories'
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories', 1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)
    df.head()
    
    print('Original no. of rows', df.shape[0])

    # check number of duplicates
    duplicate = df[df.duplicated()]
    print('Duplicate no. of rows', duplicate.shape[0])
    duplicate.head(40)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # check number of duplicates
    print('No. of rows after dropping: ', df.shape[0])
    print('Total-Found Duplicates', 26386 - duplicate.shape[0])
    print('No. of duplicates now: ', df[df.duplicated()].shape[0])
    
    print(df.head())

    return df
    
def save_data(df, database_filename):
    """
    Saving the created dataframe to disk
    """
    db_name = 'sqlite:///' + str(database_filename)
    engine = create_engine(db_name)
    df.to_sql('Messages_Categories', engine, index=False) 

def main():
    """
    Main runner function with flow of entire program
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories, df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(categories, df)
        
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