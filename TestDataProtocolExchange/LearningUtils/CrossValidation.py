
#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os,sys
import numpy as np
import pickle

pythonMachine = 'Host'
if pythonMachine == 'Docker':
    sys.path.insert(0,'/tf/BTU_Git/testdatadeeptest/TestDataExperiments/')
if pythonMachine == 'Host':
    sys.path.insert(0,'c:/Projekte/NextCloud/DeepTestITPS/WorkingFolder/HWW/Python/')

import TestData.TestDataAnalysis.TestDataExperiments.DeepTestLib.Utilities as utils
from TestData.TestDataAnalysis.TestDataExperiments.DeepTestLib.Utilities import getParameterArrayCombinations
from TestDataProtocolExchange.SignalManagementPython.PlotSignals import plotSignalDictionary


def cvLogLosses(pklLogFile,index=0,dict={},listCombinations=[],clear=False):
    if not os.path.exists(pklLogFile):
        print(pklLogFile)
        with open(pklLogFile, "bw") as fi:
            dicRes={}
            pickle.dump([listCombinations,dicRes], fi)
    if clear:
        with open(pklLogFile, "bw") as fi:
            dicRes={}
            pickle.dump([listCombinations,dicRes], fi)
            return
    with open(pklLogFile, "br") as fi:
        [listCombinations,dicRes]=pickle.load(fi)
        if index in dicRes.keys():
            dicRes[index].add(dict)
        else:
            dicRes[index]=utils.Dict(dict)
    with open(pklLogFile, "bw") as fi:
        pickle.dump([listCombinations,dicRes], fi)


def getExtremalParameterCombination(pklLogFile,index=str(0),extremal='max',plot=False):
    with open(pklLogFile, "br") as fi:
        [listCombinations,dicAllRes]=pickle.load(fi)
        dicRes={key:value[index] for (key, value) in dicAllRes.items()}
    if(extremal.upper().startswith('MA')):
        dicTmp={key: max(value) for (key, value) in dicRes.items() }
        extComb = max(dicTmp, key=dicTmp.get)
    else:
        dicTmp={key: min(value) for (key, value) in dicRes.items() }
        extComb = min(dicTmp, key=dicTmp.get)
    if plot:
        plotSignalDictionary({extComb:dicRes[extComb]},maxNo=10)
    return listCombinations[extComb]

# PARAMS is a list of parameter variant settings {'p1':(2,3)} p1=2, p1=3,..
# if there is only one element in variation list extend by None, which will be dropped
def __cvSearch(buildModel,learnProcedure,train_dataset,test_dataset,PARAMS,**kwargs):
    defaultParameter={'combinatoric':'GridSearch','condition':lambda x : True}
    defaultParams={'EPOCHS':5,'TRAINING_RUNS':250,'TEST_RUNS':10,'SAVE_MODEL':False,'PARAM_INDEX':0,
        'CV_LOGPATH':'c://Temp//cv.pkl','CV_LOG':True}
    defaultParams.update(kwargs)
    defaultParameter.update(kwargs)
    listCombinations=utils.getParameterCombinations(PARAMS,**defaultParameter)
    if defaultParams['CV_LOG']:
        cvLogLosses(defaultParams['CV_LOGPATH'],listCombinations=listCombinations,clear=True)
    for c in listCombinations:
        model=buildModel(c)
        defaultParams.update(c)
        learnProcedure(model,train_dataset,test_dataset,**defaultParams)
        defaultParams['PARAM_INDEX']+=1

def cvGridSearch(buildModel,learnProcedure,train_dataset,test_dataset,PARAMS,**kwargs):
    defaultParameter={'combinatoric':'GridSearch','condition':lambda x : True}
    defaultParameter.update(kwargs)
    __cvSearch(buildModel,learnProcedure,train_dataset,test_dataset,PARAMS,**defaultParameter)

def cvRandomSearch(buildModel,learnProcedure,train_dataset,test_dataset,PARAMS,**kwargs):
    defaultParameter={'combinatoric':'RandomSearch','fraction':0.3,'condition':lambda x : True}
    defaultParameter.update(kwargs)
    __cvSearch(buildModel,learnProcedure,train_dataset,test_dataset,PARAMS,**defaultParameter)

def cvNAryCombinatoricSearch(buildModel,learnProcedure,train_dataset,test_dataset,PARAMS,**kwargs):
    defaultParameter={'combinatoric':'NAryCombinatoricSearch','fraction':0.3,'condition':lambda x : True}
    defaultParameter.update(kwargs)
    __cvSearch(buildModel,learnProcedure,train_dataset,test_dataset,PARAMS,**defaultParameter)
#%%
# # region examples
# import random
# modelName='ConVarEncoder'
# pklLogFile='c:/Projekte/Temp/'+modelName+'.pkl'
# # PARAMS1={ 'N_LATENT_VARIABLES' : (50,30,10),'LAYER_CNT':([32,16],[12,16])}
# # PARAMS2={'KERNEL_SIZES':([3,5,3],[2,3,4]),'DROPOUT':([0.1,0.0],[0.0,0.0])}
# # liComb=getParameterArrayCombinations([PARAMS1,PARAMS2])
# liComb=[1,2,3,4,5]
# cvLogLosses(pklLogFile,listCombinations=liComb,clear=True)
# #
# i=0
# import random
# for c in liComb:
#     dic={}
#     for i in range(15):
#         for j in range(7):
#             dic[str(j)]=random.random()
#         cvLogLosses(pklLogFile,c,dic)

# #%%
# getExtremalParameterCombination(pklLogFile,index=str(0),extremal='min',plot=True)






# %%
