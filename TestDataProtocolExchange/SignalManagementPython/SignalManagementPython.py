# # class Signal:
# #   print('This is a signal class')
# #   """class for representing a signal, name and values"""
# #   def test():
# #     print('class test')

# def my_function():
#     print("Hello World")

# # Defining our variable
# name = "Nicholas"

# # Defining a class
# class Signal:
#     # def __init__(self, name, course):
#     #     self.course = course
#     #     self.name = name

#     def get_student_details(self):
#         # print("Your name is " + self.name + ".")
#         # print("You are studying " + self.course)
#         print('This is a signal class')

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Start code from here
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%
from __future__ import absolute_import, division, print_function, unicode_literals
import sys,os
import numpy as np
import tensorflow as tf

# add curent path to the env
# # %%Code modified start here%%
# sys.path.insert(0,os.getcwd())
# td_btuGit_path = os.path.abspath(os.path.join(__file__, '..', '..', ))
# folders=['\\ProtoBuffer_Signale\\Code\\', 
#     '\\SignalManagementPython\\','\\SignalPreparation\\']

# for f in folders:
#      sys.path.insert(0,td_btuGit_path+f)
# # %%Code modified end here%%
def signalsmgtpythontest():
  print('Signals management python function test.')
# import Signals_pb2
import TestDataProtocolExchange.ProtoBuffer_Signale.Code.Signals_pb2 as Signals_pb2

BytesList = Signals_pb2.BytesList
FloatList = Signals_pb2.FloatList
Int64List = Signals_pb2.Int64List
Feature = Signals_pb2.Feature
Features = Signals_pb2.Features
FeatureList = Signals_pb2.FeatureList
FeatureLists = Signals_pb2.FeatureLists
Example = Signals_pb2.Example
SequenceExample = Signals_pb2.SequenceExample

#
# class definition signal
# contains values and name
# access is given by indexing 'name' resp. 'values'
# signals can be mapped to features or featurelist
#
#region signal
class Signal:
    """class for representing a signal, name and values"""
    __name=''
    __values=[]
    __tuples=[]

    # constructor
    def __init__(self, name='',values=[],tuples=[]):
        self.__name=name
        self.__values=values
        self.__tuples=tuples

    def __getitem__(self, item):
        if(isinstance(item,(int, np.integer))):
            if len(self.__values) >0:
                return self.__values[item]
            elif len(self.__tuples) >0:
                return [val[0] for val in self.__tuples if val[1]<=item][-1:][0]
        if(isinstance(item,(tuple))):
            if len(self.__values) >0:
                return self.__values[item[0]:item[1]]
            elif len(self.__tuples) >0:                
                index=item[0]
                val_=self[item[0]]
                ret_val=[]
                for v in [val for val in self.__tuples if (val[1]>item[0] and val[1]<item[1])]:
                    ret_val.extend((v[1]-index)*[val_])
                    index=v[1]  
                    val_=v[0]               
                ret_val.extend((item[1]-index)*[val_])
                return ret_val
        if isinstance(item,slice):
            it=(item.start,item.stop)
            return self[it]
        if(item=='name'):
            return self.__name
        if(item=='values'):
            return self.__values
        return None
    # properties
    def getName(self):
        return self.__name
    
    def getValues(self):
        if len(self.__values) >0:
                return self.__values
        elif len(self.__tuples) >0:
            return self.__tuples
    def length(self):
        if len(self.__values) >0:
                return len(self.__values)
        elif len(self.__tuples) >0:
            return max([t[1] for t in self.__tuples])
    # copy signal to a feature
    def copySignal2Features(self):
        name=self.__name.encode(encoding='utf-8')
        name=Feature(bytes_list=BytesList(value=[name]))

        if len(self.__tuples) >0:
            val=self.__tuples[0][1]
            if isinstance(val,(int, np.integer)) :            
                values=Feature(int64_tuple_list=self.convert2TupleList())        
            elif isinstance(val,(float,np.float)) :
                values=Feature(float_tuple_list=self.convert2TupleList()) 
            
        elif len(self.__values) >0:
            val=self.__values[0]
            if isinstance(val,(bool,np.bool)) :
                values=Feature(bool_list=BytesList(value=self.__values))
            elif isinstance(val,(int, np.integer)) :
                values=Feature(int64_list=Int64List(value=self.__values))
            elif isinstance(val,(float,np.float)) :
                values=Feature(float_list=FloatList(value=self.__values))          
        
        return Features(feature={'name':name,'values':values})    

    def __transform2TimeValueTuples(self):
        vals=self.__values
        val_d=np.array(np.where(vals-np.append(vals[1:],vals[-1:]) != 0))
        val_d=val_d+1
        val_d=np.insert(val_d,vals[0],0)
        self.__tuples=[[vals[i],i] for i in val_d]

    def convert2TupleList(self):
        if len(self.__tuples) ==0:
            self.__transform2TimeValueTuples()
        if isinstance(self.__tuples[0][0], (int, np.integer)):
            intTupleList=Signals_pb2.Int64TupleList()
            for t in self.__tuples:
                intTupleList.value.append(Signals_pb2.Int64Tuple(value=t[0],time_stamp=t[1]))
            return intTupleList
        elif isinstance(self.__tuples[0][0], (float, np.float)):
            floatTupleList=Signals_pb2.FloatTuple()
            for t in self.__tuples:
                floatTupleList.value.append(Signals_pb2.FloatTuple(value=t[0],time_stamp=t[1]))
            return floatTupleList
    
#endregion


#
# class definition signals
# contains signals and a dictionary of various context informations
# access is given by indexing 'signalname' -> signal, resp. 'context dict key' -> context info
# signals can be mapped to sequenceexample
# read from a tfrecord or proto file wrt. Signal.proto definition
# they can be written to tfrecords
# getContext resp. getSignals return the fields
#
#region signals
class Signals:
    """class for representing signals, list of some signals and a context description"""
    __dictContext={'context': ''}
    __signals=[]

    # constructor    
    def __init__(self, signal=None,signals=None, context=''):
        if(isinstance(context,str)):
            self.__dictContext={'context': context}
        else:
            self.__dictContext=context
        if(not signals is None):
            self.__signals=signals
        else:
            self.__signals=[]
        if(not signal is None):
            self.__signals.append(signal)

    # properties
    def getContext(self):
        return self.__dictContext
    
    def getSignals(self):
        return self.__signals

    def count(self):
        return len(self.__signals)
    
    def length(self):
        return max([s.length() for s in self.__signals])

    #indexer for signal
    def __getitem__(self, signalname):
        for sig in self.__signals:
            if(sig.getName()==signalname):
                return sig
        if(isinstance(signalname,str)):
            if(signalname in self.__dictContext):
                return self.__dictContext[signalname]
        if(isinstance(signalname,(tuple))):
            win=[]
            for sig in self.__signals:
                win.extend(sig[signalname])
            return np.reshape(win,[len(self.__signals),signalname[1]-signalname[0]])
        return None

    # add a signal to signals
    def addSignals(self,signal=None,signals=None):
        if(not signals is None):
            self.__signals.extend(signals)        
        if(not signal is None):
            self.__signals.append(signal)

    # private copy function of feature arrays (signals) to featurelists
    def __copySignals2FeatureLists(self,liSigsAsTuples=[]):
        arFeatures=[]
        for sig in self.__signals:
            if sig.getName() in liSigsAsTuples:
                sig.convert2TupleList()
            arFeatures.append(sig.copySignal2Features())
        dict={}
        for f in arFeatures:
            name=self.__convert2String(f.feature['name'].bytes_list)
            values=f.feature['values']
            dict[name]=FeatureList(feature=[values])
        return FeatureLists(feature_list=dict)

    # copy signals to Features
    def copySignals2Features(self,liSigsAsTuples=[]):
        dic = {}
        name='context'
        context=''
        for k, v in self.__dictContext.items():
            context += '//'+k+':'+ v

        dic[name] = Feature(bytes_list=BytesList(value=[context.encode(encoding='utf-8')]))
        for sig in self.__signals:
            if sig.getName() in liSigsAsTuples:
                sig.convert2TupleList()
            fs = sig.copySignal2Features()
            name = self.__convert2String(fs.feature['name'].bytes_list)
            vals = fs.feature['values']
            dic[name]=vals
        return Features(feature=dic)

    # copy signals to SequenceExample
    def copySignals2SequenceExample(self,liSigsAsTuples=[]):
        featureContext={}
        for k,v in self.__dictContext.items():
            value=v.encode(encoding='utf-8')
            value=Feature(bytes_list=BytesList(value=[value]))
            featureContext[k]=value
        return SequenceExample(context=Features(feature=featureContext),
                feature_lists=self.__copySignals2FeatureLists(liSigsAsTuples))

    # writes signals to a data record file
    def writeToTFRecordFile(self,file,liSigsAsTuples=[]):
        with tf.io.TFRecordWriter(file+'.tfrecords') as sw:
            sw.write(self.copySignals2Features(liSigsAsTuples).SerializeToString())
            # sw.write(self.copySignals2SequenceExample(liSigsAsTuples).SerializeToString())



    # reads signals from a data record file
    def readFromTFRecordFile(self,file=None,files=None,cnt=1):
        self.__dictContext={}
        self.__signals=[]
        rfiles=[]
        if(not files is None):
            rfiles=files
        if(not file is None):
            rfiles=[file]
        for fil in rfiles:
            fil=fil.replace('.tfrecords','')
            raw_dataset = tf.data.TFRecordDataset([fil+'.tfrecords'])
            # raw_dataset = tf.data.TFRecordDataset([fil + '.prt'])
            feats=Features()
            for raw_record in raw_dataset:
                feats.ParseFromString(raw_record.numpy())
                for it in feats.feature.items():
                    if it[0]=='context':
                        liContext=it[1].bytes_list.value[0].decode('utf-8').split('//')
                        for sigDescr in liContext:
                            if ':=' in sigDescr:
                                name,descr=sigDescr.split(':=')
                                self.__dictContext[name]=descr
                    else:
                        if (len(it[1].int64_list.value) > 0):
                            self.__signals.append(Signal(name=it[0], values=it[1].int64_list.value))
                        elif (len(it[1].float_list.value) > 0):
                            self.__signals.append(Signal(name=it[0], values=it[1].float_list.value))
                        elif (len(it[1].bytes_list.value) > 0):
                            self.__signals.append(Signal(name=it[0], values=it[1].bytes_list.value))
                        elif (len(it[1].int64_tuple_list.value)>0):
                            self.__signals.append(Signal(name=it[0], tuples=
                                [ [it[1].int64_tuple_list.value[i].value,it[1].int64_tuple_list.value[i].time_stamp]
                                    for i in range(len(it[1].int64_tuple_list.value))]))
                        elif (len(it[1].float_tuple_list.value)>0):
                            self.__signals.append(Signal(name=it[0], tuples=
                            [[it[1].float_tuple_list.value[i].value, it[1].float_tuple_list.value[i].time_stamp]
                             for i in range(len(it[1].float_tuple_list.value))]))

                # for name,feat in sequEx.feature_lists.feature_list.items():
                #     if (len(feat.feature[0].int64_list.value) > 0):
                #         self.__signals.append(Signal(name=name, values=(feat.feature[0].int64_list.value)))
                #     elif (len(feat.feature[0].float_list.value) > 0):
                #         self.__signals.append(Signal(name=name, values=feat.feature[0].float_list.value))
                #     elif (len(feat.feature[0].bytes_list.value) > 0):
                #         self.__signals.append(Signal(name=name, values=feat.feature[0].bytes_list.value))
                #     elif (len(feat.feature[0].int64_tuple_list.value) > 0):
                #         self.__signals.append(Signal(name=name, 
                #             tuples=[ (it.value,it.time_stamp) for it in feat.feature[0].int64_tuple_list.value]))
                #     elif (len(feat.float_tuple_list.value) > 0):
                #         self.__signals.append(Signal(name=name, 
                #             tuples=[ (it.value,it.time_stamp) for it in feat.feature[0].float_tuple_list.value]))
    
    def __split2SignalDescriptionDitionary(self,context='',sigSep='//',descrSep=':'):
        if(context==''):
            return None
        dict={}
        for s in context.split(sigSep):
            if s!='':
                sigName,sigFormula=s.split(descrSep)
                dict[sigName]=sigFormula
        return dict

    def readFromProtoFile(self,filename=None):
        self.__dictContext={}
        self.__signals=[]       
        if(not filename is None):
            file = open(filename+'.prt', "rb")
            sigs=Signals_pb2.Features()
            sigs.ParseFromString(file.read())
            self.__dictContext=self.__split2SignalDescriptionDitionary(
                    self.__convert2String(sigs.feature['context'].bytes_list))
            for feat in sigs.feature:
                if(feat!='context'):
                    self.__signals.append(Signal(feat,self.__convert2Values(sigs.feature[feat])))
    
    #
    #utility functions
    #

    def __convert2String(self,b_list):
        s=b_list.value
        sRet=''
        for b in s:
            sRet+=b.decode('utf-8')
        return sRet

    def __convert2Values(self,v_list):
        if(len(v_list.int64_list.value)>0):
            iLen=len(v_list.int64_list.value)
            return np.reshape(v_list.int64_list.value,[iLen])
        elif(len(v_list.float_list.value)>0):
            iLen=len(v_list.float_list.value)
            return np.reshape(v_list.float_list.value,[iLen])
        elif(len(v_list.bytes_list.value)>0):
            iLen=len(v_list.bytes_list.value)
            return np.reshape(v_list.bytes_list.value,[iLen])
        return None

#endregion

# converter for transforming .prot format to tfrecord
def convertProtoFile2TFRecord(filename):
    if(not filename is None):
        sigs=Signals()
        sigs.readFromProtoFile(filename)
        sigs.writeToTFRecordFile(filename)

def convertProto2TFRecord(folder):
    for f in os.listdir(folder):
        if(f.endswith('.prt')):
            convertProtoFile2TFRecord(folder+f.replace('.prt',''))




#region examples

# sigs=Signals()
# folder='c:/Projekte/NextCloud/DeepTestITPS/WorkingFolder/HWW/CSharp/TestdatenGenerierung/SignalBeispiele/'
# file='c:\\Projekte\\NextCloud\\DeepTestITPS\\WorkingFolder\\HWW\\CSharp\\TestdatenGenerierung\\SignalBeispiele\\TDGenerierungNeu'
#
# # name='HLDS_A'
# sigs.readFromTFRecordFile(file)
# a=0
# sigs.copySignals2Features(['HLDS_A','HLDS_B'])
# file='c:/Temp/tupleTest'
# sigs.writeToTFRecordFile(file,liSigsAsTuples=['HLDS_A','HLDS_B','HLDS_A_IMPLIES','HLDS_B_IMPLIES'])
#
# file='c:/Projekte/NextCloud/DeepTestITPS/WorkingFolder/HWW/CSharp/TestdatenGenerierung/SignalBeispiele/test.tfrecords'

# sigsNew=Signals()
# sigsNew.readFromTFRecordFile(file)
# l=sigsNew.length()
# a=0
# pls.plotSignalDictionary(signalDictionary={'HLDS_A':sigsNew['HLDS_A']},signalShifts={},show=True)
# feats=sigs['HLDS_A'].copySignal2TupleFeatures()
# sequEx=sigs.copySignals2TupleSequenceExample()
# with tf.io.TFRecordWriter('c:/Temp/neuerTest'+'.tfrecord') as sw:
#             sw.write(sequEx.SerializeToString())

# a=0

#endregion