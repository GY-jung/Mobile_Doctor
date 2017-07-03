import numpy as np
import csv

def elimrow1(fever) :  
    fever = fever[1:]
    return fever


def elim600(filename,delimiter=' ', verbose = True):
    fever = tonumpy(elimrow1(csvreader(filename, delimiter)))
    Aiddict = {}
    ind2list = []
    cnt = np.zeros((len(fever),1))
    temp = np.zeros(len(fever))

    print (filename)
    for ind, row in enumerate(fever): 
        if row[1] in Aiddict.keys():    
            temp[ind] = Aiddict[row[1]]
        elif (int(row[9]) < 601) :
            Aiddict[row[1]] = (row[6])
            temp[ind] = Aiddict[row[1]]
            cnt[ind] = 1
        if (verbose and not (ind % 10000)):
            print ("step %d finished" %ind) 
    fever = fever[(temp!=0)]
    cnt = cnt[(temp!=0)]
    temp = temp[(temp!=0)]
    temp2 = np.array(fever[:,6], dtype = "float32")
    temp3 = -temp2 + temp
    tempdiff = np.zeros((len(fever),1))
    tempdiff[:,0] = temp3
    fever2 = np.concatenate((fever, cnt), axis = 1)
    fever2 = np.concatenate((fever2, tempdiff), axis = 1)
    return fever2




def preprocesswrapper(filename):

    with open('_' + filename, 'w') as writer:
        np.savetxt('_' + filename, tonumpy(elimrow1(csvreader(filename, delimiter = ' '))), fmt = '%s', delimiter = ' ')




def dummygenerator(csvdict):
    for i in range(12):
        j = i+1
        dummyarray = np.zeros((len(csvdict[j]),12))
        dummyarray[:, i] = 1  
        csvdict[j] = np.concatenate((csvdict[j], dummyarray), axis = 1)
        with open('__' + "%d" %j + ".csv", 'w') as writer:
            np.savetxt('__' + "%d" %j + ".csv", csvdict[j], fmt = '%s', delimiter = ' ')
    return csvdict



def merge():
    fevermerge = csvdict[1]
    for i in range(11):
        j = i+2
        print (csvdict[j].shape)
        fevermerge = np.concatenate((fevermerge, csvdict[j]), axis = 0)
    with open("merge.csv", 'w') as writer:
        np.savetxt("merge.csv", fevermerge, fmt = '%s', delimiter = ' ')
    return merge

def takeminus(filename):
    print (filename)
    fever = tonumpy(csvreader(filename))
    fever_ = np.array(fever[:,-1], dtype = "float")
    fever[:,-1] = fever_ * (-1)
    with open(filename, 'w') as writer:
        np.savetxt(filename, fever, fmt = '%s', delimiter = ' ') 

