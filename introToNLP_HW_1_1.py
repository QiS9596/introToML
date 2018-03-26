from numpy import *
import re

def loadDataIntoNDArray(filename:str):
    """
    this method loads file with filename as it's name and returns ndarray object contains the data inside the file
    :param filename:path to target file
    :return: ndarray
    """
    with open(filename) as file:
        temp = file.readlines()
        for i in range(0,len(temp)):
            temp[i] = re.split(' ',temp[i].replace('\n',''))
            for ii in range(0,len(temp[i])):
                temp[i][ii] = int(temp[i][ii])
        temp = array(temp)
        return temp

def add(filename1:str, filename2:str):
    a = loadDataIntoNDArray(filename1)
    b = loadDataIntoNDArray(filename2)
    b = b.transpose()
    print(a)
    print(b)
    return a+b

prefix = './HW1-1/'
data = 'data.txt'
answer = 'answer.txt'
print(add(prefix+data,prefix+answer))

