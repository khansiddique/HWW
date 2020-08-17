# from __future__ import division, print_function, unicode_literals
# import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
# import pandas as pd
import numpy as np

import sys,os

# # %%Code modified start here%%
# sys.path.insert(0,os.getcwd())
# td_btuGit_path = os.path.abspath(os.path.join(__file__, '..', '..', ))
# folders=['\\ProtoBuffer_Signale\\Code\\', 
#     '\\SignalManagementPython\\','\\SignalPreparation\\']

# for f in folders:
#      sys.path.insert(0,td_btuGit_path+f)
# # %%Code modified end here%%

def plotsignalstest():
  print('Plot signals function test.')
# import SignalManagementPython as sm
# Here relative import path has been change to absolute import path
import TestDataProtocolExchange.SignalManagementPython.SignalManagementPython as sm

SAMPLING_TIME = 0.01


#
# basic functions for plotting signals
#
# region plotSignalsSeparately
def plotSignalsSeparately(signals=None, sigNames=[], title='', start=0, end=100, nrows=2, ncols=2):
    if (signals == None):
        return None
    if (len(sigNames) == 0):
        for sig in signals.getSignals():
            sigNames.append(sig['name'])
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.suptitle(title)
    iNdex, iRow, iCol = [0, -1, 0]
    linewidth = max([0.2, 10 / (start - end)])
    # parameter
    xTicks = 5
    for sigName in sigNames:
        sig = signals[sigName]
        iCol = iNdex % ncols
        if (iCol == 0):
            iRow += 1
        if (iRow < nrows):
            ax = None
            if (ncols > 1 and nrows > 1):
                ax = axes[iRow, iCol]
            elif (ncols == 1):
                if(nrows == 1):
                    ax=axes
                else:
                    ax = axes[iRow]
            elif (nrows == 1):
                if (ncols == 1):
                    ax = axes
                else:
                    ax = axes[iCol]


            ax.set_title(sig['name'])
            ax.plot(range(start, end, 1), sig['values'][start:end], linewidth=linewidth)
            ax.use_sticky_edges = False
            # if(iCol>0):
            #     ax.set_yticks([])
            if (iRow < nrows - 1):
                ax.set_xticks([])
            else:
                xar = [int(n * (end - start) / xTicks) for n in range(start, start + xTicks + 1, 1)]
                ax.set_xticks(xar)
        iNdex += 1
    plt.tight_layout()
    plt.show()


# given a dictionary of signal names and their values, compose them to Signals and plot them
def plotSignalDictionary(signalDictionary=None,signalShifts={},**kwargs):

    plt.clf()
    if (signalDictionary == None):
        return None
    for sigName in signalDictionary:
        if (not sigName in signalShifts):
            signalShifts[sigName]=0

    default_args = {'title': '', 'start': 0, 'end': -1,'maxNo': 6,'offset':0,'step_size':1,'figurepath':'','show':False}
    # override defaults
    for key in kwargs.keys():
        default_args[key] = kwargs[key]

    start=default_args['start']
    end=default_args['end']
    step_size=default_args['step_size']
    blnShow=default_args['show']

    labels =[]
    for sigName in signalDictionary:
        labels.append(sigName)

    iLength=-1
    dicValues={}
    for sigName in signalDictionary:
        if (type(signalDictionary[sigName]) == sm.Signal):
            dicValues[sigName]=signalDictionary[sigName].getValues()
        else:
            if isinstance(signalDictionary[sigName],(list, np.ndarray)):
                dicValues[sigName] = signalDictionary[sigName]
                if isinstance(dicValues[sigName][0],tuple):
                    iLength=max(dicValues[sigName].time_steps())
                else:
                    iLength=len(dicValues[sigName])
            else:
                dicValues[sigName] = signalDictionary[sigName]
    if end<0:
        
        end = start+np.max([(iLength+signalShifts[sigN]) for sigN in signalDictionary])


    plt.title(default_args['title'])
    iNdex=0
    for sigName in signalDictionary:
        if iNdex>=default_args['maxNo']:
            break
        if isinstance(dicValues[sigName],(list, np.ndarray,float, int)):
            plt.plot((np.add(range(0,len(dicValues[sigName])),signalShifts[sigName])/step_size)[start:end - signalShifts[sigName]],
                 np.add(dicValues[sigName][start:end - signalShifts[sigName]], iNdex * default_args['offset']), '.-',
                 label=labels[iNdex])            
        else:
            plt.plot([i[1] for i in dicValues[sigName]],[i[0] for i in dicValues[sigName]], '.-',label=labels[iNdex])
        # plt.plot(np.add(range(0, len(dicValues[sigName])), signalShifts[sigName]) / step_size,
        #          np.add(dicValues[sigName][start:end - signalShifts[sigName]], iNdex * default_args['offset']), '.-',
        #          label=labels[iNdex])
        iNdex+=1
    plt.legend()
    # plt.xlim([start, end-start])
    plt.xlabel('Time-Step')
    if default_args['figurepath']!='':
        plt.savefig(default_args['figurepath'])
    plt.show(block=blnShow)

# endregion

# region plot images
#TODO: There is some bug in it!
def BlowUpImage(image):
    scale = (image.shape[1]*3) // (image.shape[0]*4)
    scale=max([scale,1])
    return np.repeat(image[:, :], scale,axis=0)

def plotImageDictionary(imageDictionary=None, **kwargs):
    if (imageDictionary == None):
        return None
    # defaults
    #cmap \in magma,viridis,plasma,... Greys,Blues,Reds,...,binary,hot,gray,...,Set1,Pastel2,Paired,...
    default_args = {'title': 'Images', 'nrows': 2, 'ncols': 2,
                    'cmap': 'Greys', 'interpolation': 'nearest', 'labeltype': 'normal','figurepath':'','block':False}
    # override defaults
    for key in kwargs.keys():
        default_args[key] = kwargs[key]

    ncols = default_args['ncols']
    nrows = default_args['nrows']

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    fig.suptitle(default_args['title'])

    iNdex, iRow, iCol = [0, -1, 0]

    # parameter
    for image in imageDictionary:
        if iNdex >= ncols*nrows:
            break
        imTitle = image
        imData = imageDictionary[image]
        scale = imData.shape[1] // imData.shape[0]
        imData=BlowUpImage(imData)
        iCol = iNdex % ncols
        if (iCol == 0):
            iRow += 1
        if iRow < nrows:
            ax = None
            if (ncols > 1 and nrows > 1):
                ax = axes[iRow, iCol]
            elif (ncols == 1):
                ax = axes[iRow,0]
            elif (nrows == 1):
                ax = axes[0, iCol]
            ax.set_title(imTitle)

            ax.imshow(imData, cmap=default_args['cmap'], interpolation=default_args['interpolation'])
            ax.use_sticky_edges = False

            if iRow < nrows - 1:
                ax.set_xticks([])
            else:
                if default_args['labeltype'] == 'symmetrical':
                    if imData.shape[1] % 2 == 0:
                        xar = range(-imData.shape[1] // 2, imData.shape[1] // 2 + 1, 1)
                    else:
                        xar = range(-imData.shape[1] // 2 + 1, imData.shape[1] // 2 + 1, 1)
                    ax.set_xticks(ticks=xar)
                else:
                    pass
                # ax.set_xticks(ticks=np.arange(0, xscale, step=scale))
                # ax.set_xticklabels(labels=xar)

            if (iCol == 0):
                if default_args['labeltype'] == 'symmetrical':
                    if imData.shape[0] % 2 == 0:
                        yar = range(-imData.shape[0] // 2 +1, imData.shape[0] // 2 + 2, 1)
                    else:
                        yar = range(-imData.shape[0] // 2 + 1, imData.shape[0] // 2 + 1, 1)
                else:
                    ax.set_yticklabels([i for i in range(imData.shape[0])])
                    # scale
                    # ax.set_yticks(ticks=[(n-0.5)*scale for n in range(0,yscale,scale)])
                # ax.set_yticklabels(yar)
            else:
                ax.set_yticks([])
        iNdex += 1
    plt.tight_layout()
    if default_args['figurepath']!='':
        plt.savefig(default_args['figurepath'])
    plt.show(block=default_args['block'])


def generate_and_save_images(fig_path, epoch,imageDictionary):
    
    # tight_layout minimizes the overlap between 2 sub-plots
    figurePath =fig_path +'_image_at_epoch_{:04d}.png'.format(epoch)
    plotImageDictionary(imageDictionary=imageDictionary, figurepath=figurePath)

# endregion



# region examples
# sigs=sm.Signals()
# file='c:\\Projekte\\NextCloud\\DeepTestITPS\\WorkingFolder\\HWW\\CSharp\\TestdatenGenerierung\\SignalBeispiele\\TDGenerierungNeu.tfrecords'
# sigs.readFromTFRecordFile(file)
#
# plotSignalsSeparately(signals=sigs, sigNames=('HLDS_A','HLDS_A_IMPLIES','HLDS_B','HLDS_B_SYNCH'),
#                       title='', start=0, end=512, nrows=2, ncols=2)

# sigs=sm.Signals()
# sigs.readFromTFRecordFile('c:/Projekte/NextCloud/ITPower/CSharpProjects/TestdatenGenerierung/SignalBeispiele/TDGenerierung')
#
# liSigs=['RND_A','RND_A_IMPLIES','RND_C','RND_C_IMPLIES','RND_BOOL1','RND_BOOL1_SYNC']
# signalDictionary={}
# for sigName in sigs.getContext():
#     if sigName in liSigs:
#         signalDictionary[sigName]=sigs[sigName]

# plotSignalDictionary(signalDictionary,start=200,end=600)
#
#
# plotSignalsSeparately(signals=sigs, sigNames=liSigs, title='', start=0, end=100, nrows=3, ncols=2)


# file='c:/Projekte/NextCloud/DeepTestITPS/WorkingFolder/HWW/CSharp/TestdatenGenerierung/SignalBeispiele/test'
#
# sigsNew=sm.Signals()
# sigsNew.readFromTFRecordFile(file)
# plotSignalDictionary(signalDictionary={'HLDS_B':sigsNew['HLDS_B']},signalShifts={},show=True)
# endregion
