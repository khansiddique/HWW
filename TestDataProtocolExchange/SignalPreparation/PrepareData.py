#%%
# from __future__ import division, print_function, unicode_literals
# import numpy as np
# import re,random
# import sys,os
# import tensorflow as tf
# from tensorflow import keras

# # add current path to the env

# sys.path.insert(0,os.getcwd())
# td_btuGit_path = os.path.abspath(os.path.join(__file__, '..', '..', ))
# folders=['\\ProtoBuffer_Signale\\Code\\', 
#     '\\SignalManagementPython\\','\\SignalPreparation\\']

# for f in folders:
#      sys.path.insert(0,td_btuGit_path+f)
# /content/drive/My Drive/Colab Notebooks/HWW/TestDataProtocolExchange/SignalManagementPython

# I modified this code path to absolute import from the relative import
# # import Signals_pb2
# import TestDataProtocolExchange.SignalManagementPython.SignalManagementPython as sm
# # import PlotSignals as pls
# import TestData.TestDataAnalysis.TestDataExperiments.DeepTestLib.Utilities as utils
# print('Import complete in signal preparation.')



# print(sm.name)

# nicholas = sm.Signal()
# nicholas.get_student_details()
# def preptest():
#   utils.utiltest()
#   sm.my_function()
#   print('prepare test dataset....1')
#   pass
# The test was passed here and the code was worked for me.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ********************** START CODE ************************************
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%
from __future__ import division, print_function, unicode_literals
import numpy as np
import re,random
import sys,os
import tensorflow as tf
from tensorflow import keras

# add current path to the env
# %%Code modified start here%%
# sys.path.insert(0,os.getcwd())
# td_btuGit_path = os.path.abspath(os.path.join(__file__, '..', '..', ))
# folders=['\\ProtoBuffer_Signale\\Code\\', 
#     '\\SignalManagementPython\\','\\SignalPreparation\\']

# for f in folders:
#      sys.path.insert(0,td_btuGit_path+f)
#  %%Code modified end here%%

# import Signals_pb2
import TestDataProtocolExchange.ProtoBuffer_Signale.Code.Signals_pb2 as Signals_pb2
import TestDataProtocolExchange.SignalManagementPython.SignalManagementPython as sm
# import SignalManagementPython as sm
import TestDataProtocolExchange.SignalManagementPython.PlotSignals as pls
# Here relative import path has been change to absolute import path
# import PlotSignals as pls
import TestData.TestDataAnalysis.TestDataExperiments.DeepTestLib.Utilities as utils

def prepdatatest():
  print('Prepare data function test.')
# parses the description of type '=PERIODIC(len=10000,period=27,lat=2,perc=30,width=3;7;5)'
# into [period, width array]
def parsePeriodDescription(sDescription):
    len=re.search('len=(\d*),',sDescription).group(1)
    period=re.search('period=(\d*),',sDescription).group(1)
    width=re.search('width=([\d;]*)\)',sDescription).group(1)
    width=width.split(';')
    # remove empty entries at the end
    if width[-1]=='':
        width=width[:-1]
    ret=[period]+width
    return [int(list_item) for list_item in ret]

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a]).astype(int)

# starting from the proto file of signals, an enhanced data set is generated consisting of 
# the fourier transforms on windows
# the lowest and highst frequencies are cut off (Offset) resp. the values cut below some Threshold
# after that the values are normalized
# given signals this function prepares the data for input in an ann
# the offset is due to fft and cuts accordingly the highest and lowest frequences 
# similar the iThreshold, cuts the peaks
# the signal window is given by the WindowSize, 
# shifted by Shift
def writeFourierEnhancedSignalRecords(file,Start=0, Offset=3,Threshold=10, WindowSize=512, Shift=27, Compressed=True,Debug=False):
    sigs=sm.Signals()
    sigs.readFromProtoFile(file)
    saveDir=os.path.dirname(file)
    for sig in sigs.getSignals():           
        name=sig.getName()
        sFile=os.path.join(saveDir, name + '.tfrecords')
        if Compressed:
            wTFRWriter = tf.io.TFRecordWriter(sFile,tf.io.TFRecordOptions(compression_type="GZIP"))
        else:
            wTFRWriter = tf.io.TFRecordWriter(sFile)
        sDescription=sigs.getContext()[name]
        lab=parsePeriodDescription(sDescription)
        label=np.zeros(10,int)
        label[0:len(lab)]=lab
        # for testing the fit function
        if(Debug):
            label=one_hot(int(lab[0]/5),31)
        values=sig.getValues()
        Length=len(values)
        Max=(Length-WindowSize-Start)//Shift -1
        cvalues=tf.complex(tf.dtypes.cast(values, tf.float32),tf.zeros(len(values)))
        for j in range(0,Max): 
            ftValues=tf.signal.fft(cvalues[Start+j*Shift:Start+j*Shift+WindowSize]).numpy()
            ftValues[0:Offset]=0
            ftValues[-Offset:]=0            
            ftReal=np.maximum(np.minimum(ftValues.real,Threshold),-Threshold)
            ftReal=ftReal/np.max([np.max(ftReal),1])
            ftIm=np.maximum(np.minimum(ftValues.imag,Threshold),-Threshold)
            ftIm=ftIm/np.max([np.max(ftIm),1])
            enh_sig_example=tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label':tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                        'values':tf.train.Feature(float_list=tf.train.FloatList(value=values[Start+j*Shift:Start+j*Shift+WindowSize])),
                        'ftRealvalues':tf.train.Feature(float_list=tf.train.FloatList(value=ftReal)),
                        'ftImvalues':tf.train.Feature(float_list=tf.train.FloatList(value=ftIm))
                    }))
            wTFRWriter.write(enh_sig_example.SerializeToString())
        wTFRWriter.close()

# given signals this function prepares the data for input in an ann
# the batch starts at iStart
# the offset is due to fft and cutting of the highest and lowest frequences
# similar the iThreshold, cutting to high values
# the signal window is given by the iWindowSize, 
# shifted by iShift
def prepareData(sigs=None,Start=0, Offset=3,Threshold=50, WindowSize=512, BatchSize=128,Shift=27):
    if sigs == None:
        return None
    batch=tf.Variable([])
    dicSignals={}
    for j in range(0,BatchSize):
        iSt=Start+j*Shift
        for sig in sigs.getSignals():
            sigName=sig['name']
            batch=tf.concat([batch,sig['values'][iSt:iSt+WindowSize]],0)
            dicSignals['FT_real_'+sigName]=tf.maximum(tf.minimum(tf.math.real(tf.signal.fft(sig['values'][iSt:iSt+WindowSize])),Threshold),-Threshold)        
            dicSignals['FT_imag_'+sigName]=tf.maximum(tf.minimum(tf.math.imag(tf.signal.fft(sig['values'][iSt:iSt+WindowSize])),Threshold),-Threshold)
            batch=tf.concat([batch,dicSignals['FT_real_'+sigName]],0)
            batch=tf.concat([batch,dicSignals['FT_imag_'+sigName]],0)
            lab = parsePeriodDescription(sigs.getContext()[sigName])
            label=tf.Variable(tf.zeros(10),)
            label[0:len(lab)].assign(lab)
            batch=tf.concat([batch,label],0)
    return tf.reshape(batch,[BatchSize,len(sigs.getSignals()),3*WindowSize+10])

def prepareAndShuffleData(sigs=None,Start=0, Offset=3,Threshold=50, WindowSize=512, BatchSize=128,Shift=27,Shuffle=137,Total=3):
    if sigs == None:
        return None
    retData=None        
    iSigCNT=len(sigs.getSignals())-1
    Length=len(sigs.getSignals()[0].getValues())
    Max=(Length-WindowSize)%(BatchSize*Shift) -1
    random.seed=Shuffle
    for i in range(0,Total):
        batch=tf.Variable([])
        dicSignals={} 
        sigID=random.randint(0,iSigCNT)
        sig=sigs.getSignals()[sigID]   
        rdStart=random.randint(Start,Max)
        for j in range(0,BatchSize):
            iSt=rdStart+j*Shift
            sigName=sig['name']
            batch=tf.concat([batch,sig['values'][iSt:iSt+WindowSize]],0)
            dicSignals['FT_real_'+sigName]=tf.maximum(tf.minimum(tf.math.real(tf.signal.fft(sig['values'][iSt:iSt+WindowSize])),Threshold),-Threshold)        
            dicSignals['FT_imag_'+sigName]=tf.maximum(tf.minimum(tf.math.imag(tf.signal.fft(sig['values'][iSt:iSt+WindowSize])),Threshold),-Threshold)
            batch=tf.concat([batch,dicSignals['FT_real_'+sigName]],0)
            batch=tf.concat([batch,dicSignals['FT_imag_'+sigName]],0)
            lab = parsePeriodDescription(sigs.getContext()[sigName])
            label=tf.Variable(tf.zeros(10),)
            label[0:len(lab)].assign(lab)
            batch=tf.concat([batch,label],0)
        if retData is None:
            retData=tf.reshape(batch,[BatchSize,1,3*WindowSize+10])
        else:
            retData=tf.concat([retData,tf.reshape(batch,[BatchSize,1,3*WindowSize+10])],1)

    return retData



def composeSignals2TotalImages(signalDictionary):
    image=[]
    iNdex=0
    length=-1
    for sigName in signalDictionary:
        iNdex+=1
        if (type(signalDictionary[sigName]) == sm.Signal):
            image.extend(signalDictionary[sigName].getValues())
        else:
            image.append(signalDictionary[sigName])
            length=len(signalDictionary[sigName])
        if(length<0): length=len(image)
    return np.reshape(image,[iNdex,length])


def getTotalImages(signalTFRecordFile):
    sigs = sm.Signals()
    sigs.readFromTFRecordFile(signalTFRecordFile)
    signalDictionary = {}
    for sigName in sigs.getContext():
        signalDictionary[sigName] = sigs[sigName]
    return composeSignals2TotalImages(signalDictionary)

# encodes signals into one image
def encodeSignal2Image(signalDictionary=None,**kwargs):

    if (signalDictionary == None):
        return None
    default_args = {'start': 0, 'end':-1,'window_size': 512, 'step_size': 1}
    # override defaults
    for key in kwargs.keys():
        default_args[key] = kwargs[key]

    start = default_args['start']
    end = default_args['end']
    window_size = default_args['window_size']
    step_size = default_args['step_size']
    totalImage= composeSignals2TotalImages(signalDictionary)
    iNdex=totalImage.shape[0]
    length=totalImage.shape[1]
    retImages=np.array([])
    if end<0:
        end=length-window_size-1-start
    else:
        end= window_size - 1 + start +end
    for i in range(start,end):
        retImages=np.append(retImages,totalImage[:,i:i + window_size])
    return np.reshape(retImages,[end-start,iNdex,window_size])

# decodes an image into several signals
def decodeImages2Signals(image=None,**kwargs):
    default_args = {}
    for key in kwargs.keys():
        default_args[key] = kwargs[key]

    signalRetDictionary = {}
    imTrans = np.transpose(image)
    if 'names' in default_args:
        names=default_args['names']
    else:
        names=[str(i) for i in range(imTrans.shape[1])]

    for i in range(imTrans.shape[1]):
        signalRetDictionary[names[i]] = imTrans[:, i]

    return signalRetDictionary



#############################################################
# feed in a list of signals and return the decoded signal
# to feed to LSTM:

def signalTimeSeriesEncoder(sigs):
    encodedSignal=np.array(sigs[0])
    for i in range(1,len(sigs)):
        encodedSignal=np.add(encodedSignal,[2**i*x for x in sigs[i]])
    return encodedSignal
##############################################################
# compress Signals to values, history since last change and time
# doubles the signals and adds time so that at any step the following values are given
#  * actual time step
#  * current signal value at that time
#  * number of time steps since last change
##############################################################
def compressSignals(sigs):
    # get the signal values and put them into a dictionary
    dicValues={s.getName():s.getValues() for s in sigs.getSignals()}
    [ dicValues[k].append((v[-1][0],sys.maxsize)) for k,v in dicValues.items()]    
    # for any signal get the final index (stopping criterium)
    dicStopIndices={k:len(v)-1 for k,v in dicValues.items()}
    # current signal index
    dicValIndices={k:0 for k,v in dicValues.items()}

    # add the values
    dicSignalValues={k:v[dicValIndices[k]][0] for k,v in dicValues.items()}
    # add variable storing the time duration since last change (synchronous, compressed)
    dicSignalChangeHistory={k+'_scr':0 for k in dicValues.keys()}    
    dicSignalValues.update(dicSignalChangeHistory)
    # actual time
    dicSignalValues.update({'t':0})
    # the result array conaining doubled signals (value + time duration since last change)
    sigValues=[]
    sigValues.append([v for k,v in dicSignalValues.items()])
    # for stepping through time, carries the related signal step index, where the signals are given by tuples
    # (value, timepoint of change)
    dicValIndices={k:0 for k,v in dicValIndices.items()}

    liSigNames=[n for n in dicValIndices.keys()]
    iCNT=0
    # this routine works correctly because at one time point only one signal value can change (single core!)
    # two and more signals can change at the same time step abstraction!!
    while(iCNT<100000):
        # get the next time point where some signal changes value
        dicNextTimes={k:v[dicValIndices[k]+1][1] for k,v in dicValues.items() if dicValIndices[k]+1 < dicStopIndices[k] }        
        if not dicNextTimes:
            break
        dicNextTimes={k:v for k,v in sorted(dicNextTimes.items(), key = lambda kv:(kv[1], kv[0]))}
        nextTime=dicNextTimes[list(dicNextTimes.keys())[0]]
        dicNextChanges={k:v for k,v in dicNextTimes.items() if v==nextTime}  
             
        if(nextTime==sys.maxsize):
            break

       # get the time of previous changes
        t=dicSignalValues['t']

        # collect the changes
        dicSignalChanges={'t':nextTime}
        for k in liSigNames:
            if not k in dicNextChanges:
                dicSignalChanges[k+'_scr']=dicSignalValues[k+'_scr']+nextTime-t 
            else:
                dicSignalChanges[k+'_scr']=0
                dicSignalChanges[k]=dicValues[k][dicValIndices[k]+1][0]
                dicValIndices[k]+=1
        dicSignalValues.update(dicSignalChanges)
        sigValues.append([v for k,v in dicSignalValues.items()])
        iCNT+=1
    return sigValues
###############################################################
# plot the signals and the encoded signal using the module PlotSignals
###############################################################

###############################################################

#region examples
#file='c:/Projekte/NextCloud/ITPower/CSharpProjects/TestdatenGenerierung/SignalBeispiele/PeriodicSignals/TDGenerierungPeriodic'
#writeFourierEnhancedSignalRecords(file,Start=0, Offset=3,Threshold=10, WindowSize=512, Shift=27, Compressed=False,Debug=True)

# sigDir='c:\\Projekte\\NextCloud\\ITPower\\CSharpProjects\\TestdatenGenerierung\\SignalBeispiele\\'
# files=['PRD1','PRD2','PRD3','PRD4','PRD5','PRD61']
# files=[sigDir+f+'.tfrecords' for f in files]
# # files= sigDir+'PRD*.tfrecords'
# dataset=tf.data.Dataset.list_files(files)

# file='c:/Projekte/NextCloud/ITPower/CSharpProjects/TestdatenGenerierung/SignalBeispiele/PRD61.tfrecords'

# feature_description = {
#      "label": tf.io.FixedLenFeature([10],tf.int64),
#      "values": tf.io.FixedLenFeature([512], tf.float32),
#      "ftRealvalues": tf.io.FixedLenFeature([512], tf.float32),
#      "ftImvalues": tf.io.FixedLenFeature([512], tf.float32)
#      }

# dataset = tf.data.TFRecordDataset([file],compression_type="GZIP")
# for serialized_examples in dataset:
#      parsed_examples = tf.io.parse_single_example(serialized_examples, feature_description)
#      a=0

# sigs=sm.Signals()
# sigs.readFromTFRecordFile('c:/Projekte/NextCloud/DeepTestITPS/WorkingFolder/HWW/CSharp/TestdatenGenerierung/SignalBeispiele/TDGenerierung')
#
# liSigs=['RND_A','RND_A_IMPLIES','RND_C','RND_C_IMPLIES','RND_BOOL1','RND_BOOL1_SYNC'
#     ,'HLDS_A','HLDS_B','HLDS_C','RND_BOOL2','RND_BOOL2_SYNC']
# liSigs=['RND_A','RND_A_IMPLIES']
# liSigs=['RND_C','RND_C_IMPLIES']
# liSigs=['RND_BOOL1','RND_BOOL1_SYNC']
# signalDictionary={}
# for sigName in sigs.getContext():
#     if sigName in liSigs:
#         signalDictionary[sigName]=sigs[sigName]
#
# # plotSignalDictionary(signalDictionary,start=200,end=600)
# im= Encode2Image(signalDictionary,window_size=256,end=100)
#
# sigOrig={}
# for sigName in sigs.getContext():
#     if sigName in liSigs:
#         sigOrig[sigName]=sigs[sigName].getValues()[0:256]
# pls.plotSignalDictionary(signalDictionary=sigOrig)
# pls.plotImageDictionary(imageDictionary={'1':im[0]},nrows=1, ncols= 1)
# signalRetun=DecodeImages2Signals(im[0])
# pls.plotSignalDictionary(signalDictionary=signalRetun)

# imageDictionary={}
# for i in range(0,6):
#     imageDictionary[str(i)]=im[10*i+2]
# # pls.plotSignalsSeparately(signals=sigs, sigNames=liSigs, title='', start=0, end=100, nrows=3, ncols=2)
# pls.plotImageDictionary(imageDictionary=imageDictionary,nrows=1, ncols= 1)

# file='c:/Projekte/NextCloud/DeepTestITPS/WorkingFolder/HWW/CSharp/TestdatenGenerierung/SignalBeispiele/test.tfrecords'
#
# sigs=sm.Signals()
# sigs.readFromTFRecordFile(file)
# ds=tf.data.TFRecordDataset([file])
# feats=sm.Features()
# for raw_record in ds:
#     feats.ParseFromString(raw_record.numpy())
# a=0

# file='c:/Projekte/NextCloud/DeepTestITPS/WorkingFolder/HWW/CSharp/TestdatenGenerierung/SignalBeispiele/testKOSTAL.tfrecords'

# sigs=sm.Signals()
# sigs.readFromTFRecordFile(file)
# cSigs=compressSignals(sigs)
# a=0
#endregion

# %%

