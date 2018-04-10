import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Activation

model = Sequential()
model.add(Dense(units=300, input_dim=6, kernel_initializer='uniform'))
model.add(Activation('relu'))

model.add(Dense(units=300, kernel_initializer='uniform'))
model.add(Activation('relu'))

model.add(Dense(units=200, kernel_initializer='uniform'))
model.add(Activation('relu'))

model.add(Dense(units=1, kernel_initializer='uniform'))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.summary()

from xlsReader import excelReader

reader = excelReader('./training data(1000).xlsx')
data, labels = reader.processData()
model.fit(x=data, y = labels,epochs=100,validation_split=0.1,batch_size=10,verbose=1)

scores = model.evaluate(x=data,y=labels)
print(scores[1])

validation_data, validation_labels = excelReader('./testing data.xlsx').processData(random=False)
output = model.predict(validation_data)

for i in range(len(output)):
    if output[i]<0.5:
        output[i] = 0
    else:
        output[i] = 1
from xlsReader import kaggleFileGenerator

kaggleFileGenerator('id,survived').output(data=output)
