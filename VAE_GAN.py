#region import ...
# ## VAE-GAN ([article](https://arxiv.org/abs/1512.09300))
# VAE-GAN combines the VAE and GAN to autoencode over a latent representation of data in the generator to improve over the pixelwise error function used in autoencoders.
#
# pip install --upgrade grpcio --user
# pip install setuptools --upgrade --user
# pip install tensorflow --user
# pip install -U tensorflow-probability==0.6.0 --user
# pip install -U dm-sonnet==1.23 --user
# pip install --upgrade tfp-nightly --user
# pip list or
# pip list | grep tensor
# pip install --upgrade tensorflow
# https: // stackoverflow.com/questions/56935876/no-module-named-tensorflow-probability
# https://stackoverflow.com/questions/59104396/error-tensorboard-2-0-2-has-requirement-setuptools-41-0-0-but-youll-have-set
# https: // askubuntu.com/questions/1212304/tensorboard-refuses-to-open-versionconflict-grpcio-1-24-3
# https://colab.research.google.com/github/as-ideas/headliner/blob/master/notebooks/BERT_Translation_Example.ipynb#scrollTo=UjkRZ2isplkC
# https://www.tensorflow.org/install/pip
import sys, subprocess

# #### Note: to get this working on fashion MNIST without using any sort of batch normalization I added two parameters: `latent_loss_div` and `recon_loss_div`. Their purpose is just to scale the loss of the KL Divergence and the reconstruction error in order to balance the three losses in the generator (reconstruction, KL divergence, GAN). Similar to $\beta$-VAE. In addition I balance the generator and discriminator loss by squashing the discriminator loss with a sigmoid based on how much it is beating the generator by.
#
# #### In addition - the quality of samples continues to improve well past 50 epochs - so I reccomend training for longer than I did here! I'm sure you could also find better hyperparameters to improve training speeds.
# ### load packages

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

import tensorflow_probability as tfp
ds = tfp.distributions

from os import path
import sys,math,io
import datetime
from contextlib import redirect_stdout

modelName='VAE_GAN'
pythonMachine = 'Host'
print("I am in Here")
if pythonMachine == 'Docker':
    # print("I am in docker")
    sys.path.insert(0,'/tf/TestData/TestDataAnalysis/')
    sys.path.insert(0,'/tf/TestDataProtocolExchange/SignalManagementPython')
    signalFile = '/tf/Docker/SignalBeispiele/test'
    modPath = '/tf/TestData/TestDataAnalysis/TestDataExperiments/checkPoints/chkPts'+modelName+'Imgs'
    figPath='/tf/figure_plots/'
    tbPath = '/tf/TensorBoard/tb'+modelName
    from PlotSignals import generate_and_save_images
    import PrepareData as prep
    import PrepareDataSet as prepDS

if pythonMachine == 'Host':
    print("I am in Host")
    # Did a change here
    # sys.path.insert(
    #     0, 'F:/Semester-5th_Internship/ITpower/Task2/Folder/HWW/Python/')
    # sys.path.insert(
    #     0, 'F:/Semester-5th_Internship/ITpower/Task2/Folder/HWW/Python/TestData/TestDataAnalysis/')
    # signalFile = 'c:/Projekte/NextCloud/DeepTestITPS/WorkingFolder/HWW/CSharp/TestdatenGenerierung/SignalBeispiele/test'
    signalFile = '/content/drive/My Drive/Colab Notebooks/HWW/CSharp/TestdatenGenerierung/SignalBeispiele/test'
    print(signalFile)
    # modPath = 'c:/Projekte/NextCloud/DeepTestITPS/WorkingFolder/HWW/Python/TestData/TestDataAnalysis/TestDataExperiments/checkPoints/chkPts'+modelName+'Imgs'
    modPath = '/content/drive/My Drive/Colab Notebooks/HWW/TestData/TestDataAnalysis/TestDataExperiments/checkPoints/chkPts'+modelName+'Imgs'
    print(modPath)
    # figPath='c:/Projekte/NextCloud/DeepTestITPS/WorkingFolder/HWW/Python/figure_plots/'
    figPath = '/content/drive/My Drive/Colab Notebooks/HWW/Python/figure_plots/'
    print(figPath)
    # tbPath = 'c:/Projekte/NextCloud/DeepTestITPS/WorkingFolder/HWW/Python/TensorBoard/tb'+modelName
    tbPath = '/content/drive/My Drive/Colab Notebooks/HWW/Python/TensorBoard/tb' + modelName
    print(tbPath)
    # Did a change till here path of the variable with before.
    from TestDataProtocolExchange.SignalManagementPython.PlotSignals import generate_and_save_images
    import TestDataProtocolExchange.SignalPreparation.PrepareData as prep
    import TestDataProtocolExchange.SignalPreparation.PrepareDataSet as prepDS
    import TestDataProtocolExchange.LearningUtils.CrossValidation as cv
#endregion

# '''

TEST_SIZE=10
TRAIN_SIZE=120
WINDOW_SIZE=256

TRAIN_BUF = 1000
TEST_BUF = 100
MAX=8000
BATCH_SIZE = 32
SIGNAL_CNT,SIGNAL_LENGTH=prepDS.countSignalsLength(signalFile+'.tfrecords')
MAX=8000
train_dataset=prepDS.buildSignalImageDS(signalFile+'.tfrecords',WINDOW_SIZE,0,MAX,False).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset=prepDS.buildSignalImageDS(signalFile+'.tfrecords',WINDOW_SIZE,MAX,SIGNAL_LENGTH-WINDOW_SIZE-1,False).shuffle(TEST_BUF).batch(BATCH_SIZE)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = tbPath +'/'+ current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

EPOCHS=2 #200
SAMPLE_INTERVAL=1 #20
blnSaveModel=False

# SIGNAL_CNT=3#defaultParams['SIGNAL_CNT']
# WINDOW_SIZE=7#defaultParams['WINDOW_SIZE']


#region Hyperparameter...
def setHyperParameter(BATCH_CNT):
    #region set Hyperparameter
    if BATCH_CNT==0:
        enc_optimizer= tf.keras.optimizers.Adam(1e-3, beta_1=0.5)
        dec_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.5)
        disc_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
        ENC_FEATURES=[16,32]
        DEC_FEATURES=[32,16]
        DISC_FEATURES=[16,32,256]
        KERNEL_SIZE=[3,3,3,3]
        N_LATENT_VARIABLES=128
        ALPHA=0.2
        DROPOUT=0.0
        MOMENTUM=0.8
        GRADIENT_PENALTY_WEIGHT=10.0
        LATENT_LOSS_DIV=1
        SIG_MULT=10
        RECON_LOSS_DIV=0.001
        LR_DISC=1e-4


    dicModel={
        'SIGNAL_CNT':SIGNAL_CNT,
        'WINDOW_SIZE':WINDOW_SIZE,
        'N_LATENT_VARIABLES':N_LATENT_VARIABLES,
        'ALPHA':ALPHA,
        'DROPOUT':DROPOUT,
        'MOMENTUM':MOMENTUM,
        'enc_optimizer':enc_optimizer,
        'dec_optimizer':dec_optimizer,
        'disc_optimizer':disc_optimizer,
        'gradient_penalty_weight':GRADIENT_PENALTY_WEIGHT,
        'ENC_FEATURES':ENC_FEATURES,
        'DEC_FEATURES':DEC_FEATURES,
        'DISC_FEATURES':DISC_FEATURES,
        'KERNEL_SIZE':KERNEL_SIZE,
        'LATENT_LOSS_DIV':LATENT_LOSS_DIV,
        'SIG_MULT':SIG_MULT,
        'RECON_LOSS_DIV':RECON_LOSS_DIV,
        'LR_DISC':LR_DISC}

    sModelParams=''
    for it in dicModel:
        sModelParams+=(it +' = '+ str(dicModel[it])+'   \n\n')

    with train_summary_writer.as_default():
        tf.summary.text('model', sModelParams,step=0)
    return dicModel



# lr_base_gen = 1e-3, #
# lr_base_disc = 1e-4, # the discriminator's job is easier than the generators so make the learning rate lower
# latent_loss_div=1, # this variable will depend on your dataset - choose a number that will bring your latent loss to ~1-10
# sig_mult = 10, # how binary the discriminator's learning rate is shifted (we squash it with a sigmoid)
# recon_loss_div = .001, # this variable will depend on your dataset - choose a number that will bring your latent loss to ~1-10



#endregion


# ### Define the network as tf.keras.model object

class VAE_GAN(tf.keras.Model):
    """a VAE_AD class for tensorflow

    Extends:
        tf.keras.Model
    """

    def __init__(self, **kwargs):
        super(VAE_GAN, self).__init__()
        #region default settings
        self.__dict__={}
        self.__dict__['enc_optimizer']= tf.keras.optimizers.Adam(1e-3, beta_1=0.5)
        self.__dict__['dec_optimizer ']= tf.keras.optimizers.Adam(1e-3, beta_1=0.5)
        self.__dict__['disc_optimizer ']= tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
        self.__dict__['ENC_FEATURES']=[16,32]
        self.__dict__['DEC_FEATURES']=[32,16]
        self.__dict__['DISC_FEATURES']=[16,32,256]
        self.__dict__['KERNEL_SIZE']=[3,3,3,3]
        self.__dict__['N_LATENT_VARIABLES']=128
        self.__dict__['ALPHA']=0.2
        self.__dict__['DROPOUT']=0.0
        self.__dict__['MOMENTUM']=0.8
        self.__dict__['GRADIENT_PENALTY_WEIGHT']=10.0
        self.__dict__['LATENT_LOSS_DIV']=1
        self.__dict__['SIG_MULT']=10
        self.__dict__['RECON_LOSS_DIV']=0.001
        self.__dict__['LR_DISC']=1e-4
        self.__dict__['SAVE_MODEL']=True
        self.__dict__.update(kwargs)
        #endregion

        super(VAE_GAN, self).__init__()
        self.enc = self.vae_EncoderModel(self.__dict__)
        self.dec = self.vae_DecoderModel(self.__dict__)
        self.disc = self.vae_GAN_DiscModel(self.__dict__)

        self.latent_loss_div=self.__dict__['LATENT_LOSS_DIV']
        self.recon_loss_div=self.__dict__['RECON_LOSS_DIV']
        self.sig_mult=self.__dict__['SIG_MULT']

        self.enc_optimizer = self.__dict__['enc_optimizer']
        self.dec_optimizer = self.__dict__['dec_optimizer']
        self.disc_optimizer = self.__dict__['disc_optimizer']
        # self.lr_base_disc1=kwargs['dicParams']['disc_optimizer'].learning_rate

        self.disc_optimizer.learning_rate=self.get_lr_d

        if self.__dict__['SAVE_MODEL']:
            with train_summary_writer.as_default():
                    with io.StringIO() as buf, redirect_stdout(buf):
                        self.enc.summary()
                        summary = buf.getvalue()
                        tf.summary.text('encoder', summary,step=0)
                        self.dec.summary()
                        summary = buf.getvalue()
                        tf.summary.text('decoder', summary,step=0)
                        self.disc.summary()
                        summary = buf.getvalue()
                        tf.summary.text('discriminator', summary,step=0)

    #region Construct encoder, decoder discriminant
    def vae_EncoderModel(self,defaultParams):
        SIGNAL_CNT=defaultParams['SIGNAL_CNT']
        WINDOW_SIZE=defaultParams['WINDOW_SIZE']
        ENC_FEATURES=defaultParams['ENC_FEATURES']
        KERNEL_SIZE=defaultParams['KERNEL_SIZE']
        N_LATENT_VARIABLES=defaultParams['N_LATENT_VARIABLES']
        ALPHA=defaultParams['ALPHA']
        MOMENTUM=defaultParams['MOMENTUM']
        DROPOUT=defaultParams['DROPOUT']
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(SIGNAL_CNT ,WINDOW_SIZE,1)),
            tf.keras.layers.Conv2D(
                filters=ENC_FEATURES[0], kernel_size=(SIGNAL_CNT,KERNEL_SIZE[1]), strides=1, activation="relu",padding="SAME"),
            tf.keras.layers.Conv2D(
                filters=ENC_FEATURES[1], kernel_size=(SIGNAL_CNT,KERNEL_SIZE[2]), strides=1, activation="relu",padding="SAME"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=N_LATENT_VARIABLES*2),
            ])
    def vae_DecoderModel(self,defaultParams):
        SIGNAL_CNT=defaultParams['SIGNAL_CNT']
        WINDOW_SIZE=defaultParams['WINDOW_SIZE']
        DEC_FEATURES=defaultParams['ENC_FEATURES']
        DEC_FEATURES.reverse()
        KERNEL_SIZE=defaultParams['KERNEL_SIZE']
        N_LATENT_VARIABLES=defaultParams['N_LATENT_VARIABLES']
        ALPHA=defaultParams['ALPHA']
        MOMENTUM=defaultParams['MOMENTUM']
        DROPOUT=defaultParams['DROPOUT']
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(N_LATENT_VARIABLES)),
            tf.keras.layers.Dense(units=SIGNAL_CNT * WINDOW_SIZE* BATCH_SIZE, activation="relu"),
            tf.keras.layers.Reshape(target_shape=(SIGNAL_CNT ,WINDOW_SIZE, BATCH_SIZE)),
            tf.keras.layers.Conv2DTranspose(
                filters=DEC_FEATURES[0], kernel_size=(SIGNAL_CNT,KERNEL_SIZE[2]), strides=1, padding="SAME", activation="relu"),
            tf.keras.layers.Conv2DTranspose(
                filters=DEC_FEATURES[1], kernel_size=(SIGNAL_CNT,KERNEL_SIZE[1]), strides=1, padding="SAME", activation="relu"),
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=(SIGNAL_CNT,KERNEL_SIZE[0]), strides=1, padding="SAME", activation="sigmoid"),
            ])
    def vae_GAN_DiscModel(self,defaultParams):
        SIGNAL_CNT=defaultParams['SIGNAL_CNT']
        WINDOW_SIZE=defaultParams['WINDOW_SIZE']
        DISC_FEATURES=defaultParams['DISC_FEATURES']
        KERNEL_SIZE=defaultParams['KERNEL_SIZE']
        N_LATENT_VARIABLES=defaultParams['N_LATENT_VARIABLES']
        ALPHA=defaultParams['ALPHA']
        MOMENTUM=defaultParams['MOMENTUM']
        DROPOUT=defaultParams['DROPOUT']

        inputs = tf.keras.layers.Input(shape=(SIGNAL_CNT , WINDOW_SIZE, 1))
        conv1 = tf.keras.layers.Conv2D(
                    filters=DISC_FEATURES[0], kernel_size=(SIGNAL_CNT,KERNEL_SIZE[0]), strides=1, padding="SAME",
                    activation="relu")(inputs)
        conv2 = tf.keras.layers.Conv2D(
                    filters=DISC_FEATURES[1], kernel_size=(SIGNAL_CNT,KERNEL_SIZE[1]), strides=1, padding="SAME",
                    activation="relu" )(conv1)
        flatten = tf.keras.layers.Flatten()(conv2)
        lastlayer = tf.keras.layers.Dense(units=DISC_FEATURES[2], activation="relu")(flatten)
        outputs = tf.keras.layers.Dense(units=1, activation = None)(lastlayer)
        return tf.keras.Model(inputs=[inputs], outputs=[outputs,lastlayer])
    #endregion

    def encode(self, x):
        mu, sigma = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return mu, sigma

    def dist_encode(self, x):
        mu, sigma = self.encode(x)
        # MultivariateNormalDiag(mu,sigma)
        return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

    def get_lr_d(self):
        return self.__dict__['LR_DISC'] * self.D_prop

    def decode(self, z):
        if tf.size(tf.shape(z))>2:
            bz,lv=[tf.shape(z)[0],tf.shape(z)[3]]
            return self.dec(tf.reshape(z,(bz,lv)))
        else:
            return self.dec(z)

    def discriminate(self, x):
        return self.disc(x)

    def reconstruct(self, x):
        mean, _ = self.encode(x)
        return self.decode(mean)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean



    # @tf.function
    def compute_loss(self, x):
        # pass through network
        q_z = self.dist_encode(x)
        z = q_z.sample()
        p_z = ds.MultivariateNormalDiag(
            loc=[0.0] * z.shape[-1], scale_diag=[1.0] * z.shape[-1]
        )
        # p_z = MultivariateNormalDiag(tf.zeros(z.shape,dtype=tf.float32),
        #      tf.ones(z.shape,dtype=tf.float32))
        xg = self.decode(z)
        z_samp = tf.random.normal([x.shape[0], 1, 1, z.shape[-1]])
        xg_samp = self.decode(z_samp)
        d_xg, ld_xg = self.discriminate(xg)
        d_x, ld_x = self.discriminate(x)
        d_xg_samp, ld_xg_samp = self.discriminate(xg_samp)

        # GAN losses
        disc_real_loss = gan_loss(logits=d_x, is_real=True)
        disc_fake_loss = gan_loss(logits=d_xg_samp, is_real=False)
        gen_fake_loss = gan_loss(logits=d_xg_samp, is_real=True)

        discrim_layer_recon_loss = (
            tf.reduce_mean(tf.reduce_mean(tf.math.square(ld_x - ld_xg), axis=0))
            / self.recon_loss_div
        )

        self.D_prop = sigmoid(
            disc_fake_loss - gen_fake_loss, shift=0.0, mult=self.sig_mult
        )

        kl_div = ds.kl_divergence(q_z, p_z)
        latent_loss = tf.reduce_mean(tf.maximum(kl_div, 0)) / self.latent_loss_div

        return (
            self.D_prop,
            latent_loss,
            discrim_layer_recon_loss,
            gen_fake_loss,
            disc_fake_loss,
            disc_real_loss,
        )

    # @tf.function
    def compute_gradients(self, x):
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape:
            (
                _,
                latent_loss,
                discrim_layer_recon_loss,
                gen_fake_loss,
                disc_fake_loss,
                disc_real_loss,
            ) = self.compute_loss(x)

            enc_loss = latent_loss + discrim_layer_recon_loss
            dec_loss = gen_fake_loss + discrim_layer_recon_loss
            disc_loss = disc_fake_loss + disc_real_loss

        enc_gradients = enc_tape.gradient(enc_loss, self.enc.trainable_variables)
        dec_gradients = dec_tape.gradient(dec_loss, self.dec.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        return enc_gradients, dec_gradients, disc_gradients

    # @tf.function
    def apply_gradients(self, enc_gradients, dec_gradients, disc_gradients):
        self.enc_optimizer.apply_gradients(
            zip(enc_gradients, self.enc.trainable_variables)
        )
        self.dec_optimizer.apply_gradients(
            zip(dec_gradients, self.dec.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.disc.trainable_variables)
        )

    def train(self, x):
        enc_gradients, dec_gradients, disc_gradients = self.compute_gradients(x)
        self.apply_gradients(enc_gradients, dec_gradients, disc_gradients)


def gan_loss(logits, is_real=True):

    if is_real:
        labels = tf.ones_like(logits)
    else:
        labels = tf.zeros_like(logits)

    return tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits
    )


def sigmoid(x, shift=0.0, mult=20):

    return tf.constant(1.0) / (
        tf.constant(1.0) + tf.exp(-tf.constant(1.0) * (x * mult))
    )


def buildModel(dicParams):
    dicParams['SIGNAL_CNT']=4
    dicParams['WINDOW_SIZE']=256
    return VAE_GAN(**dicParams)


def learnProcedure(vae_gan_model,train_dataset,test_dataset,**kwargs):
    lossArray=[]
    defaultParams={'EPOCHS':5,'TRAINING_RUNS':250,'TEST_RUNS':10,'CV_LOGPATH':modPath+'\\cv_da.pkl',
        'SAMPLE_INTERVAL':1,'SAVE_MODEL':False,'MODEL_PATH':modPath,
        'PARAM_INDEX':0}
    defaultParams.update(kwargs)

    if defaultParams['SAVE_MODEL']:
        # save and restore learning processes
        checkpoint = tf.train.Checkpoint(model=vae_gan_model)
        manager = tf.train.CheckpointManager(checkpoint, defaultParams['MODEL_PATH'], max_to_keep=3)
        checkpoint.restore(manager.latest_checkpoint)

    for epoch in range(defaultParams['EPOCHS']):
        for image_batch in tqdm(train_dataset.take(defaultParams['TRAINING_RUNS'])):
                vae_gan_model.train(image_batch)

        for image_batch in tqdm(train_dataset.take(defaultParams['TEST_RUNS'])):
            d_prop,latent_loss,discrim_layer_recon_loss,\
                gen_fake_loss,disc_fake_loss,disc_real_loss=vae_gan_model.compute_loss(image_batch)
            if defaultParams['SAVE_MODEL']:
                with train_summary_writer.as_default():
                        tf.summary.scalar('d_prop', d_prop, step=epoch)
                        tf.summary.scalar('latent_loss', latent_loss, step=epoch)
                        tf.summary.scalar('discrim_layer_recon_loss', discrim_layer_recon_loss, step=epoch)
                        tf.summary.scalar('gen_fake_loss', gen_fake_loss, step=epoch)
                        tf.summary.scalar('disc_fake_loss', disc_fake_loss, step=epoch)
                        tf.summary.scalar('disc_real_loss', disc_real_loss, step=epoch)
            dic={'d_prop':d_prop,'latent_loss':latent_loss,
                'discrim_layer_recon_loss':discrim_layer_recon_loss,
                'gen_fake_loss':gen_fake_loss,'disc_fake_loss':disc_fake_loss,'disc_real_loss':disc_real_loss}
            cv.cvLogLosses(defaultParams['CV_LOGPATH'],defaultParams['PARAM_INDEX'],dic)


        if epoch % defaultParams['SAMPLE_INTERVAL'] == 0:
            example_data = next(iter(train_dataset))
            example_data_reconstructed=vae_gan_model.reconstruct(example_data[11:13,:,:,:])
            example_data_reconstructed=np.reshape(example_data_reconstructed,(2,defaultParams['SIGNAL_CNT'],defaultParams['WINDOW_SIZE']))
            sample = vae_gan_model.decode(tf.random.normal(shape=(2, defaultParams['N_LATENT_VARIABLES'])))
            sample=np.reshape(sample,(2,defaultParams['SIGNAL_CNT'],defaultParams['WINDOW_SIZE']))
            generate_and_save_images(figPath+'VAE_GAN_', epoch, {'reconstr_example 1':example_data_reconstructed[0,:,:],'sample 1':sample[0,:,:],
                'reconstr_example 2':example_data_reconstructed[1,:,:],'sample 2':sample[1,:,:]})
            if defaultParams['SAVE_MODEL']:
                manager.save()

    if defaultParams['SAVE_MODEL']:
        manager.save()
        train_summary_writer.close()


PARAMS={ 'N_LATENT_VARIABLES' : (50,30,20),
    'enc_optimizer': (tf.keras.optimizers.Adam(1e-3, beta_1=0.5),None),
    'dec_optimizer': (tf.keras.optimizers.Adam(1e-3, beta_1=0.5),None),
    'disc_optimizer': (tf.keras.optimizers.Adam(1e-3, beta_1=0.5),None),
    'ENC_FEATURES':([16,32],[16,8]),
    'DISC_FEATURES':([16,32,256],[8,16,64]),
    'KERNEL_SIZES':([3,3,3,3],[2,3,4,5],[5,4,3,2]),
    'ALPHA':(0.2,None),
    'MOMENTUM':(0.8,None),
    'GRADIENT_PENALTY_WEIGHT':(10.0,None),
    'LATENT_LOSS_DIV':(1,None),
    'SIG_MULT':(10,None),
    'RECON_LOSS_DIV':(0.001,None),
    'LR_DISC':(1e-4,None),
    'DROPOUT':(0.0,0.1)}


defaultParameter={}
defaultParameter['enc_optimizer']= tf.keras.optimizers.Adam(1e-3, beta_1=0.5)
defaultParameter['dec_optimizer']= tf.keras.optimizers.Adam(1e-3, beta_1=0.5)
defaultParameter['disc_optimizer']= tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
defaultParameter['ENC_FEATURES']=[16,32]


defaultParameter['KERNEL_SIZE']=[3,3,3,3]
defaultParameter['N_LATENT_VARIABLES']=128
defaultParameter['ALPHA']=0.2
defaultParameter['DROPOUT']=0.0
defaultParameter['MOMENTUM']=0.8
defaultParameter['GRADIENT_PENALTY_WEIGHT']=10.0
defaultParameter['LATENT_LOSS_DIV']=1
defaultParameter['SIG_MULT']=10
defaultParameter['RECON_LOSS_DIV']=0.001
defaultParameter['LR_DISC']=1e-4

defaultParameter['DISC_FEATURES']=[16,32,256]
defaultParameter['SAVE_MODEL']=False

# Did a change here
cvParams={'EPOCHS':10,'TRAINING_RUNS':250,'TEST_RUNS':10,'CV_LOGPATH':modPath+'/cv_da.pkl'}
# cvParams={'EPOCHS':10,'TRAINING_RUNS':250,'TEST_RUNS':10,'CV_LOGPATH':modPath+'\\cv_da.pkl'}

# cv.cvRandomSearch(buildModel,learnProcedure,train_dataset,test_dataset,PARAMS,**cvParams)
print("I am in function call")
model=buildModel(defaultParameter)
cvParams.update(defaultParameter)
learnProcedure(model,train_dataset,test_dataset,**cvParams)



# '''
