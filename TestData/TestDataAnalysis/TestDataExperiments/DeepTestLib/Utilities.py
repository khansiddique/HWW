#%%
import tensorflow as tf
import itertools
import numpy as np
from random import randint
from math import ceil

def utiltest():
  print('Utilitie function test.')

#region Multivariate gaussian distribution
# class for substituting package tensorflow_probability.MultivariateNormalDiag

class MultivariateNormalDiag:
    def __init__(self, mu,sigma):
        self.mu=tf.cast(mu,dtype=tf.float64)
        self.Sigma=tf.linalg.diag(tf.transpose(tf.math.abs(sigma)),k=1)
        self.Sigma_2=tf.linalg.diag(tf.math.square(tf.transpose(sigma)),k=1)
    @tf.function
    def sample(self):
        eps = tf.random.normal(shape=tf.shape(tf.transpose(self.mu)),dtype=tf.float64)
        return tf.transpose((tf.cast(self.Sigma,dtype=tf.float64)@eps[...,None])[...,0])+self.mu
        
    # corresponds to kl_divergence(mvnd,self)
    @tf.function
    def kl_divergence(self,mvnd):
        retVal=-tf.cast(tf.shape(self.mu)[0],dtype=tf.float32)*tf.ones(tf.shape(self.mu)[1])
        # retVal=tf.cast(retVal,dtype=tf.float64)
        mu1=mvnd.mu
        Sigma1_2=mvnd.Sigma_2
        d=tf.linalg.det(self.Sigma)
        d1=tf.linalg.det(mvnd.Sigma)
        retVal+=tf.cast(tf.math.log(d/d1),dtype=tf.float32)
        Sigma_2_inv=tf.linalg.inv(self.Sigma_2)
        retVal+=tf.cast(tf.linalg.trace(tf.linalg.matmul(Sigma1_2,Sigma_2_inv)),dtype=tf.float32)
        mu=tf.cast(self.mu-mu1,dtype=tf.float32)
        retVal+=tf.cast(tf.linalg.diag_part((Sigma_2_inv@mu)[...,0]@mu),dtype=tf.float32)
        return retVal
#endregion

#region combinatorics

def getOuterProduct(li1,li2=None,condition=lambda x,y : True):
    b=[]
    if li2 is None:
        li2=li1
    if isinstance(li1,list) and isinstance(li2,list):
        for i in li1:
            for j in li2:
                if condition(i,j):
                    b.append([i,j])
        return b
    elif isinstance(li1,int) and isinstance(li2,int):
        for i in range(li1):
            for j in range(li2):
                if condition(i,j):
                    b.append([i,j])
        return b

def getOuterProduct2Array(ar,condition=lambda x : True):
    arr=[]
    for i in range(len(ar)):
        arr.append(tuple(range(ar[i])))
    return list(filter(condition,itertools.product(*arr)))

#region n-ary products
def __checkNAry(list,ar):
    for i in range(0,len(ar)):
        for j in range(i+1,len(ar)):
            if not len(set(['_'+str(l[i])+'_'+str(l[j])+'_' for l in list])) \
                ==ar[i]*ar[j]:
                return False
    return True


def __getNAryOuterProduct2Array(ar,n=2):
    liC=np.array(getOuterProduct2Array(ar[0:n]))
    liC=List([List(l) for l in liC])
    dicFunc={}
    dicFunc[0]= lambda l: l[0:n].lappend((l[0]+l[1])%ar[n])
    if n==3:
        dicFunc[1]=( lambda l: l[0:n+1].lappend((l[0]+l[2])%ar[n+1]))
        dicFunc[2]=( lambda l: l[0:n+2].lappend((l[0]+l[2]+l[3])%ar[n+2]))
        dicFunc[3]=( lambda l: l[0:n+3].lappend((3*l[0]+2*l[1]+l[3]+randint(0,ar[n+3]))%ar[n+3]))
        dicFunc[4]=( lambda l: l[0:n+4].lappend((3*l[1]+2*l[2]+l[3]+randint(0,ar[n+4]))%ar[n+3]))
    else:
        dicFunc[1]=( lambda l: l[0:n+1].lappend((3*l[0]+2*l[1]+randint(0,ar[n+1]))%ar[n+1]))
        dicFunc[2]=( lambda l: l[0:n+2].lappend((3*l[1]+2*l[2]+randint(0,ar[n+2]))%ar[n+2]))
        
    for i in range(len(ar)-n):
        liC1=liC.foreach(dicFunc[i])
        iCNT=0
        while not __checkNAry(liC1,ar[0:n+i+1]):
            # liC1=list(map(dicFunc[i],liC))
            liC1=liC.foreach(dicFunc[i])
            iCNT+=1
            if iCNT>30:
                break
        liC=liC1
    return liC


def getNAryOuterProduct2Array(ar,n=2,condition=lambda x : True):
    dic={}
    length=len(ar)
    listCombinations=[]
    for i in range(len(ar)):
        dic[i]=-ar[i]
    liOrderedDim=[-v for (k,v) in sorted(dic.items(), key = lambda kv:(kv[1], kv[0]))]
    liReordering=[k for (k,v) in sorted(dic.items(), key = lambda kv:(kv[1], kv[0]))]
    listCombinations=__getNAryOuterProduct2Array(liOrderedDim,2)
    listCombinations=map(lambda l:[l[i] for i in liReordering], listCombinations )
    return list(filter(condition,listCombinations))

#endregion

def getRandomFractionalCombinatoric(ar,fraction=0.3,condition=lambda x : True):
    listCombinations=getOuterProduct2Array(ar)
    cnt=ceil(len(listCombinations)*fraction)
    liRes=[]
    while len(liRes)<=cnt :
        en=randint(0,len(listCombinations)-1)
        if not en in liRes:
            liRes.append(en)
    return list(filter(condition,[listCombinations[i] for i in liRes]))

# given some dictionary with tuple values (parameter combinatorics) the 
# cartesian product of their tuple values is build
def getParameterCombinations(PARAMS,**kwargs):
    defaultParameter={'combinatoric':'GridSearch','fraction':0.3,'n-ary':2,'condition':lambda x : True}
    defaultParameter.update(kwargs)
    liReturn=[]
    dicComb={}
    iarComb=[]
    combinatorics=[]
    i=0
    for k in PARAMS.keys():
        PARAMS[k]=[ o for o in PARAMS[k] if o is not None]
        iarComb.append(len(PARAMS[k]))
        dicComb[i]=k
        i+=1
    if defaultParameter['combinatoric'].upper().startswith('RANDOM'):
        combinatorics=getRandomFractionalCombinatoric(iarComb,fraction=defaultParameter['fraction'])
    elif defaultParameter['combinatoric'].upper().startswith('NARY'):
        combinatorics=getNAryOuterProduct2Array(iarComb,n=defaultParameter['n-ary'])
    elif defaultParameter['combinatoric'].upper().startswith('GRID') :
        combinatorics=getOuterProduct2Array(iarComb)
    for c in combinatorics:
        i=0
        params={}
        for j in c:
            k=dicComb[i]
            params[k]=PARAMS[k][j]
            i+=1
        liReturn.append(params)
    
    return list(filter(defaultParameter['condition'],liReturn))

# given several parameter dictionaries, the cartesian product of
# their combinatorical cartesian products is build
def getParameterArrayCombinations(PARAMS):
    liComb=[]
    if len(PARAMS)==1:
        return getParameterCombinations(PARAMS[0])
    elif len(PARAMS)==2:
        for l1 in getParameterCombinations(PARAMS[0]):
            for l2 in getParameterCombinations(PARAMS[1]):
                l1.update(l2)
                liComb.append(l1.copy())
        return liComb
    elif len(PARAMS)==3:
        for l1 in getParameterCombinations(PARAMS[0]):
            for l2 in getParameterCombinations(PARAMS[1]):
                for l3 in getParameterCombinations(PARAMS[1]):
                    l1.update(l2.update(l3))
                    liComb.append(l1.copy())
        return liComb
#endregion

#region filtering objects
def filterDictionary(dic,condition=lambda x,y : True):
    return {key: value for (key, value) in dic.items() if condition(key,value)}
#endregion

#region (smooth) minimum
#%%
@tf.custom_gradient
def smoothMinimum(x,gamma=0.1):
    min=np.min(x)
    
    p=tf.cast((-x+min),tf.float64)/gamma
    def grad(dx,gamma=0.1):
        m=tf.minimum(x,10000)[0].numpy()
        minIndex=tf.cast(tf.constant(x==m),tf.float64)
        return tf.multiply(-minIndex,dx/tf.reduce_sum(minIndex)),0
    return tf.Variable(-gamma*np.sum(np.exp(p))+min),grad


def grad_smoothMinimum(x,gamma=0.1):
  with tf.GradientTape() as tape:
    tape.watch(x)
    value = smoothMinimum(x,gamma)
  return tape.gradient(value, x)


@tf.custom_gradient
def Minimum(x):    
    def grad(dx):
        return 1
    return x[tf.argmin(x)],grad


def grad_Minimum(x):
  with tf.GradientTape() as tape:
    tape.watch(x)
    value = Minimum(x)
  return tape.gradient(value, x)
#endregion


#region binary encoding
# The function which converts an integer value to the binary value:
def binaryEncode(i):
    return '{:064b}'.format(i)

# binary to integer
def binaryDecode(bi):
    return sum([2**(63-i) for i in range(64) if bi[i]=='1'])

# combines several digital signal values to one integer value
def encodeSignalValues(sigValues):
    return [sum([2**(len(sigValues)-1-i) for i in range(len(sigValues)) if sigValues[i] ==1 ])]

#decodes integer signal value in n digital signal values
def decodeSignalValues(sigValue,n):
    lb='{:064b}'.format(sigValue)
    return [ord(lb[i])-48 for i in range(64) if i>64-n]

# given 2 integer return the difference when readed as composed digital signal values
def binaryIntegerDifference(val1,val2):
    if isinstance(val1,(list,np.ndarray)):
        val1=val1[0]
    if isinstance(val2,(list,np.ndarray)):
        val2=val2[0]
    return sum([1 for i in '{:064b}'.format(val1^val2) if i=='1'])
#endregion  

#region method extensions

# Method Extension via decorator
def method_extension(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator

# Method Extension via metaclass
def method_extension_class(name, bases, namespace):
    assert len(bases) == 1, "Exactly one base class required"
    base = bases[0]
    for name, value in namespace.iteritems():
        if name != "__metaclass__":
            setattr(base, name, value)
    return base

# class <newclass>(<someclass>):
#     __metaclass__ = monkeypatch_class
#     def <method1>(...): ...
#     def <method2>(...): ...

# This adds <method1>, <method2>, etc. to <someclass>, and makes
# <newclass> a local alias for <someclass>.

#region list/ dictionary extensions
class List(list):
    __metaclass__ = method_extension_class
    def __init__(self, iterable):
        super().__init__(iterable)

    def lappend(self,a):
        self.append(a)
        return self

    def foreach(self,func=lambda a:a):
        return List([func(a) for a in self])

    def __getitem__(self, item):
        lc=self.copy()
        if isinstance(item,slice):
            return List(lc[item])
        else:
            return lc[item]
# class List(list):
#     def __init__(self, iterable):
#         super().__init__(iterable)

 
# @method_extension(List)
# def add(self,a):
#     if not a in self:
#         self.append(a)
#         return self

# @method_extension(List)
# def lappend(self,a):
#         self.append(a)
#         return self

# @method_extension(List)
# def foreach(self,func=lambda a:a):
#         return List([func(a) for a in self])

# @method_extension(List)
# def __getitem__(self,item):
#     lc=self.copy()
#     return List(lc[item])

class Dict(dict):
    __metaclass__ = method_extension_class
    def __init__(self,dic):
        for k,v in dic.items():
            self[k]=[v]
    def __add(self,k,v):
        if k in self.keys():
            self[k].append(v)
        else:
            self[k]=[v]
    def add(self,dic):
        for k,v in dic.items():
            self.__add(k,v)
#endregion
#endregion


#region experiments and tests
# #%%
# li=List([1,2,3,6,7,8,9])
# li1=li.lappend(4)
# isinstance(li1,List)
# print(li1)
# f=lambda l:List([l]).lappend(35)
# l2=li1.foreach(f)
# isinstance(l2,List)

# lis=List(li)

# print(lis.foreach(lambda a:2*a))
# l3=li[2:3]
# print(l3)
# dic={'a':1,'b':2}
# dic1=Dict({})

# dic1.add({'a':3})
# print(dic1)

# print(getNAryOuterProduct2Array([3,4,2,2]))

# # %%
# l=[13,25,2233,43628]
# for i in l:
#     print(i)
#     print(binaryEncode(i))
#     print(binaryDecode(binaryEncode(i)))

# # %%
# l1=[1,1,0,0,0,1]
# l2=[0,1,0,0,1,1]
# d1=encodeSignalValues(l1)
# d2=encodeSignalValues(l2)
# print(d1)
# # print(d2)
# #%%
# dar=[2.0,4.0,2.0]
# p=np.array(dar)
# min=np.min(dar)
# min=(p==min).astype(int)
# print(np.sum(min))
# #%%    # return -gamma * (log(tmp) + max_val)
# x=np.array(dar)
# print(smoothMinimum(x))
# print(grad_smoothMinimum(tf.Variable(x)))
# print(Minimum(x))
# print(grad_Minimum(tf.Variable(x)))
# #%%
# ind=np.array([1, 0, 0])
# p=np.array(dar)
# print(p*ind)
#endregion

# # %%
# import random
# ar=[]
# for i in range(1000):
#     ar.append(random.randint(1,10000)*1.0)
# min1=min(ar)
# print(ar[tf.argmin(ar)])
# print(abs(min1))
# #%%
# print(Minimum(tf.Variable(ar)))
# print(grad_Minimum(tf.Variable(ar)))
# # print(grad_smoothMinimum(tf.Variable(ar)))

# #%%
