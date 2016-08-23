import re
import numpy as np
import pandas as pd
import datetime
import pylab as pl
import matplotlib.pyplot as plt
from io import StringIO

def loadPerformance(filename) :

    with open(filename) as f :
        content = f.readlines()

    paramRE = re.compile('^[a-z_]+:[a-z_]+', re.IGNORECASE)
    colonRE = re.compile(':')
    csvRE = re.compile('^-?[0-9.]+(,-?[0-9.])+');

    #Reading in into a string list and a data array
    paramArray = []
    values = []
    for i in content:
        if paramRE.match(i) is not None:
            temp = colonRE.split(i);
            temp[1] = temp[1].rstrip()
            paramArray.append(temp)
        if csvRE.match(i) is not None:
            temp = StringIO(i.rstrip())
            temp = np.loadtxt(temp,delimiter=",")
            values.append(temp)

    values = np.array(values)
    return (paramArray,values)


def loadLimits(filename):
    with open(filename) as f:
        content = f.readlines()

    limitsRE = re.compile('^The limits are given for the period: ')
    quantRE = re.compile('^The uncertainty quantiles are:')
    csvRE = re.compile('^-?[0-9.e-]+(,-?[0-9.e-])+')

    values = []
    quantiles = []
    fechas = []
    foundDate = False
    startReading = False

    for idx, i in enumerate(content):
        if foundDate is False:
            if limitsRE.match(i):
                foundDate = True
                temp = limitsRE.sub('', i)
                temp = temp.rsplit()
                start = datetime.datetime.strptime(temp[0], '%Y/%m/%d')
                finish = datetime.datetime.strptime(temp[2], '%Y/%m/%d')
                step = datetime.timedelta(days=1)
                while start <= finish:
                    fechas.append(start)
                    start += step
        if quantRE.match(i):
            #Quantile values will be found in the next line
            quantiles = content[idx+1]
            quantiles = StringIO(quantiles.rstrip())
            quantiles = np.loadtxt(quantiles, delimiter=',')
        if csvRE.match(i) is not None:
            if startReading is True:#Should skip the first line of numbers
                temp = StringIO(i.rstrip())
                temp = np.loadtxt(temp, delimiter=',')
                values.append(temp)
            startReading = True

    values = np.asarray(values)
    df = pd.DataFrame(data=values, index=fechas, columns=('quant:' + s for s in quantiles.astype(str) ))
    return(df)


def readMARS(filename):
    with open(filename) as f:
        content=f.readlines()
     
    paramNum = np.array([])
    relativeRanking = np.array([])
    
    for idx,i in enumerate(content[1:]):
        temp = StringIO(i.rstrip()) 
        temp = np.loadtxt(temp)
        if idx%3 == 0:
            paramNum = np.hstack([paramNum,temp]) if paramNum.size else temp
        elif idx%3 == 1:
            relativeRanking = np.hstack([relativeRanking,temp]) if relativeRanking.size else temp
            
    return(paramNum,relativeRanking)
                
            
        
def saveDottyPlot(pathStr,objFunStr,imgFormat):
    fig = pl.rcParams['figure.figsize'] = (10,15)
    filePath = pathStr + "behavioralParameters_" + objFunStr + ".txt"
    (names,values) = loadPerformance(filePath)
    performance=values[:,0]
    numCol = len(names)
    idx = 0
    for column in values[:, 1:].T:
        plt.subplot(6,3,idx+1)
        plt.plot(column,performance,'k.',MarkerSize=3)
        plt.xlabel(names[idx][0] + ':' + names[idx][1])
        plt.ylabel(objFunStr)
        idx = idx + 1

    plt.tight_layout()
    plt.savefig(objFunStr + imgFormat, dpi = 400,bbox_inches='tight')
    plt.close()







