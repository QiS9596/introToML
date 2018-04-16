import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
class excelReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = pd.read_excel(filepath)

    def processData(self, random = True):
        self.df = self.raw_data.drop(['name','home.dest','body', 'boat','ticket','embarked','cabin','id'], axis = 1)

        age_mean = self.df['age'].mean()
        self.df['age'] = self.df['age'].fillna(age_mean)
        fare_mean = self.df['fare'].mean()
        self.df['fare'] = self.df['fare'].fillna(fare_mean)
        self.df['sex'] = self.df['sex'].map({'female':0,'male':1}).astype(int)
        ndarray = self.df.values
        labels = ndarray[:,1]
        data = self.df.drop(['survived'],axis = 1).values
        scaler = MinMaxScaler(feature_range = (0,1))
        data = scaler.fit_transform(data)
        labels = labels.reshape(len(labels),1)
        if random:
            permutation = np.random.permutation(labels.shape[0])
            data = data[permutation,:]
            labels = labels[permutation]
        return data, labels

# abc = excelReader('./training data(1000).xlsx')
# abc.processData()

class kaggleFileGenerator:
    def __init__(self, title):
        self.title = title

    def output(self, name = './output.txt', data = []):
        with open(name,'w') as file:
            file.write(self.title)
            file.write('\n')
            for i in range(len(data)):
                file.write(str(i))
                file.write(',')
                file.write(str(int(data[i][0])))
                file.write('\n')