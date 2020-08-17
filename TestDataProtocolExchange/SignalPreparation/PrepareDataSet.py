#%%
from __future__ import division, print_function, unicode_literals
import numpy as np
import re,random
import sys,os
import tensorflow as tf
import pickle
import random

# add current path to the env
# # %%Code modified start here%%
# sys.path.insert(0,os.getcwd())
# td_btuGit_path = os.path.abspath(os.path.join(__file__, '..', '..', ))
# folders=['\\ProtoBuffer_Signale\\Code\\', 
#     '\\SignalManagementPython\\','\\SignalPreparation\\']
# F:\Semester-5th_Internship\ITpower\Task2\Folder\HWW
# \TestDataProtocolExchange\ProtoBuffer_Signale\Code
# for f in folders:
#      sys.path.insert(0,td_btuGit_path+f)
# # %%Code modified end here%%

def prepdatasettest():
  print('Prepare dataset function test.')

import TestDataProtocolExchange.ProtoBuffer_Signale.Code.Signals_pb2 as Signals_pb2
# # %%Here Code is modified for Import path%%
import TestDataProtocolExchange.SignalManagementPython.SignalManagementPython as SignalManagementPython
import TestDataProtocolExchange.SignalManagementPython.SignalManagementPython as sm

# import SignalManagementPython
import random,itertools

def countSignalsLength(file):
    CNT, LEN=0,0
    if(file.endswith('.tfrecords')):
        # sigs=SignalManagementPython.Signals()
        # made a change here in the code and using short form of attribute
        sigs=sm.Signals()
        sigs.readFromTFRecordFile(file)
        CNT,LEN=[sigs.count(), sigs.length()]
    if (file.endswith('.pkl')):
        with open(file, "br") as fi:
            images = pickle.load(fi)
            CNT, LEN = images.shape[0],images.shape[1]
    return [CNT,LEN]

def signals2ImageGenerator(totalImage,window_size,min,max,shuffle=False):
    try :
        for i in itertools.cycle(range(max)):
            if shuffle:
                j=random.randint(min, max-window_size)
                yield tf.reshape(totalImage[:,j:j+window_size]*1.0,(totalImage.shape[0],window_size,1))
            else:  
                j=i%(max-window_size-min) +min
                yield tf.reshape(totalImage[:,j:j+window_size]*1.0,(totalImage.shape[0],window_size,1))
    except GeneratorExit:
        #nothing to do, is just an event broadcasted to signal end of generator
        return
    except:
        print(j)
        print(sys.exc_info())
        return


            
def signals2HistoryPredictionGenerator(totalImage, history_size,prediction_size, min, max, shuffle=False):
    try :
        for i in itertools.cycle(range(max)):
            if shuffle:
                j=random.randint(min, max - history_size-prediction_size)
                yield ( tf.transpose(tf.reshape(totalImage[:,j: j + history_size]*1.0, (totalImage.shape[0], history_size))),
                            tf.transpose(tf.reshape(totalImage[:, j + history_size:j+history_size+prediction_size]*1.0,
                                    (totalImage.shape[0], prediction_size))))
            else:
                j=i%(max - history_size-prediction_size-min) +min
                yield (tf.transpose(tf.reshape(totalImage[:,j: j + history_size]*1.0, (totalImage.shape[0], history_size))),
                            tf.transpose(tf.reshape(totalImage[:, j + history_size:j+history_size+prediction_size]*1.0,
                                    (totalImage.shape[0], prediction_size))))
    except GeneratorExit:
        #nothing to do, is just an event broadcasted to signal end of generator
        return
    except:
        print(j)
        print(sys.exc_info())
        return


def buildSignalImageDS(file,window_size,min,max,shuffle):
    signals=SignalManagementPython.Signals()
    signals.readFromTFRecordFile(file)
    SIGNAL_CNT,SIGNAL_LENGTH=countSignalsLength(file)
    totalImage=signals[0,SIGNAL_LENGTH]
    def signals2ImageGeneratorDS():
        return signals2ImageGenerator(totalImage,window_size,min,max,shuffle)
    return tf.data.Dataset.from_generator(
        signals2ImageGeneratorDS, tf.float32,tf.TensorShape([SIGNAL_CNT,window_size,1]))


def buildSignal2HistoryPredictionDS(file,history_size, prediction_size,min,max,shuffle=False):
    signals=SignalManagementPython.Signals()
    signals.readFromTFRecordFile(file)
    SIGNAL_CNT, SIGNAL_LENGTH = countSignalsLength(file)
    totalImage = signals[0, SIGNAL_LENGTH]
    def signals2HistoryPredictionGeneratorDS():
        return signals2HistoryPredictionGenerator(totalImage,history_size, prediction_size,min,max,shuffle)
    return tf.data.Dataset.from_generator(
        signals2HistoryPredictionGeneratorDS,(tf.float32,tf.float32), 
            (tf.TensorShape([history_size,SIGNAL_CNT]),tf.TensorShape([prediction_size,SIGNAL_CNT])))
           
class ImageGenerator:
    __window_size=256
    __min=0
    __max=1000
    __shuffle=False
    __data=[]
    __index=0
    def __init__(self,dataFile,window_size=256,min=0,max=1000,shuffle=False):
        self.__window_size=window_size
        self.__min=min
        self.__max = max
        self.__shuffle=shuffle
        self.__index=min-1
        with open(dataFile, "br") as fi:
            self.__data = pickle.load(fi)

    # def __iter__(self):
    #     if self.__shuffle:
    #         j=random.randint(self.__min,self.__max)
    #         yield tf.reshape(self.__data[:, j:j + self.__window_size] * 1.0, (self.__data.shape[0], self.__window_size, 1))
    #     else:
    #         self.__index+=1
    #         self.__index= self.__index % (self.__max-self.__window_size-self.__min) +self.__min
    #         yield tf.reshape(self.__data[:, self.__index:self.__index + self.__window_size] * 1.0,
    #                              (self.__data.shape[0], self.__window_size, 1))

    def __getitem__(self, i):
        i= i % (self.__max - self.__window_size - self.__min) + self.__min
        return tf.reshape(self.__data[:, i:i + self.__window_size] * 1.0,
                          (self.__data.shape[0], self.__window_size, 1))

    def next(self):
        if self.__shuffle:
            j=random.randint(self.__min,self.__max)
            return tf.reshape(self.__data[:, j:j + self.__window_size] * 1.0, (self.__data.shape[0], self.__window_size, 1))
        else:
            self.__index += 1
            self.__index = self.__index % (self.__max - self.__window_size - self.__min) + self.__min
            return tf.reshape(self.__data[:, self.__index:self.__index + self.__window_size] * 1.0,
                             (self.__data.shape[0], self.__window_size, 1))
    def batch(self,size,shift=0):
        self.__index += shift
        return tf.concat([[self.next()] for i in range(0,size)],0)


def buildValues2ImageDS(file,window_size,min,max,shuffle=False):
    with open(file, "br") as fi:
        totalImage = pickle.load(fi)
    def signals2ImageGeneratorDS():
        return signals2ImageGenerator(totalImage,window_size,min,max,shuffle)
    return tf.data.Dataset.from_generator(
        signals2ImageGeneratorDS, tf.float32,tf.TensorShape([totalImage.shape[0],window_size,1]))

#%%

#region test
# MAX=8000
# WINDOW_SIZE=256
# BATCH_SIZE = 256
# TRAIN_BUF = 1000
# TEST_BUF = 100
# signalFile = 'c:/Projekte/NextCloud/DeepTestITPS/WorkingFolder/HWW/CSharp/TestdatenGenerierung/SignalBeispiele/test'
# SIGNAL_CNT,SIGNAL_LENGTH=countSignalsLength(signalFile+'.tfrecords')
# signals=SignalManagementPython.Signals()
# signals.readFromTFRecordFile(signalFile)
# totalImage=signals[0,SIGNAL_LENGTH]
# train_dataset=buildSignalImageDS(signalFile+'.tfrecords',WINDOW_SIZE,0,MAX,False)
#%%

#%%


#%%

#%%

# train_dataset=buildSignalImageDS(signalFile+'.tfrecords',WINDOW_SIZE,0,3,False).batch(BATCH_SIZE)

# l=train_dataset.skip(3).take(1).as_numpy_iterator()
# i=0
# for b in train_dataset.take(10):
#     i+=1

# print(i)
# TEST_SIZE=500
# WINDOW_SIZE=256

# SIGNAL_THRESHOLD=0.0

# TRAIN_BUF = 1000
# TEST_BUF = 100
# BATCH_SIZE = 32 #32
# SAMPLE_INTERVAL=50 #50

# BATCH_SIZE = 50
# # #
# file='c:/Projekte/NextCloud/DeepTestITPS/WorkingFolder/HWW/CSharp/TestdatenGenerierung/SignalBeispiele/test'
# sigs=SignalManagementPython.Signals()
# file=file+'.tfrecords'
# sigs.readFromTFRecordFile(file)
# SIGNAL_CNT,SIGNAL_LENGTH=countSignalsLength(file)
# totalImage=sigs[0,SIGNAL_LENGTH]
# img=signals2HistoryPredictionGenerator(totalImage,WINDOW_SIZE,17,0,MAX,False)
# im=next(img)
# a=0
# image=sigs[0,len]
# tf.data.Dataset.from_tensor_slices(tf.transpose(image))

# dsW=tf.data.Dataset.from_tensor_slices(tf.transpose(image)).window(WINDOW_SIZE,1)
# # dsWW=dsW.map(lambda x:tf.transpose(tf.reshape(np.append([], list(x.as_numpy_iterator())), [WINDOW_SIZE, cnt])))
# for wind in dsW:
#      wi=tf.transpose(tf.reshape(np.append([], list(wind.as_numpy_iterator())), [WINDOW_SIZE, cnt]))
# res=getDataSetSignalWindows(file)
# img=signals2ImageGenerator(totalImage,WINDOW_SIZE,0,MAX)
# im=next(iter(img))
#
# train_dataset=buildSignalImageDS(file,256,0,MAX,False).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
# for elem in train_dataset:
#     img=elem
# a=0
# # a=0
# train_dataset=buildSignal2HistoryPredictionDS(file,256,32,0,8000,False)#.shuffle(TRAIN_BUF).batch(BATCH_SIZE)
# for elem in train_dataset:
#     [hist,perd]=elem
# a=0
# pklFi=file+'.pkl'
# img=ImageGenerator(pklFi)
# image=img.next()
# batchImages=img.batch(32)
# image1=img[23]
# a=0
# img=buildValues2ImageDS(pklFi,WINDOW_SIZE,0,1000).batch(17)
# im=next(iter(img))
# a=0
# file='c:/Projekte/NextCloud/DeepTestITPS/WorkingFolder/HWW/CSharp/TestdatenGenerierung/SignalBeispiele/test.tfRecords'
# sigs=getSignalWindowsDataSet(file)
# a=0
#endregion