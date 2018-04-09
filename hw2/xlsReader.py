import pandas as pd
import numpy as np

class excelReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = pd.read_excel(filepath)

    def processData(self):
        self.df = self.raw_data.drop(['name','home.dest','body', 'boat','ticket','embarked','cabin','id'], axis = 1)

        age_mean = self.df['age'].mean()
        self.df['age'] = self.df['age'].fillna(age_mean)
        fare_mean = self.df['fare'].mean()
        self.df['fare'] = self.df['fare'].fillna(fare_mean)
        self.df['sex'] = self.df['sex'].map({'female':0,'male':1}).astype(int)
        ndarray = self.df.values
        labels = ndarray[:,1]
        data = ndarray[:,1:]
        return data, labels

abc = excelReader('./training data(1000).xlsx')
abc.processData()