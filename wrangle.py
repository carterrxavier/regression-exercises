import numpy as np
import pandas as pd
from env import host, user ,password
from pydataset import data
import os


def get_connection(db, user = user, host = host, password = password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
def get_telco_tenure():
    '''
    This function gets the tenure information from the telco data set for customers with 2 year contracts
    '''
    file_name = 'telco_tenure.csv'
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    
    else:
        query =  '''
        select customer_id, monthly_charges, tenure, total_charges
        from customers
        where contract_type_id = 3
        '''
    df = pd.read_sql(query, get_connection('telco_churn'))  
    
    #replace white space with nulls
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    
    df.to_csv(file_name, index = False)
    return df

def clean_telco_tenure(df):
    '''
    cleans telco tenure data
    
    '''
    #fill total charges with monthly charges
    df['total_charges'].fillna(df['monthly_charges'], inplace = True)
    
    #convert total_charges object type into a float
    df['total_charges'] = pd.to_numeric(df['total_charges'],errors='coerce')
    
    #change tenure from zero to one
    df.loc[df['tenure'] == 0, 'tenure'] = 1
    
    return df
    