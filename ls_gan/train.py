
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Sequential, Model

import os
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from init_project import init_project
from batch_iterator import get_data_iterator
import visualize

class trainer:
    
    def __init__(self,
                 project_name,
                 generator, 
                 discriminator,
                 loss_gen  = 'binary_crossentropy',
                 loss_disc = 'binary_crossentropy',
                 learning_rate = 0.0001,
                 adam_beta_1   = 0.5,
                 latent_size   = 100,
                 batch_size    = 128,
                 save_model_step = 25):
        
        # allow groth
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        # init project
        project_name = os.path.join("projects", project_name)
        init_project(project_name)
        init_project(os.path.join(project_name, "generated_img"))
        init_project(os.path.join(project_name, "model"))
        
        # set class variables
        self.generator       = generator
        self.discriminator   = discriminator
        self.learning_rate   = learning_rate
        self.adam_beta_1     = adam_beta_1
        self.latent_size     = latent_size
        self.batch_size      = batch_size
        self.project_name    = project_name
        self.save_model_step = save_model_step


        # compile
        self.discriminator.compile(
            optimizer=Adam(lr=learning_rate, beta_1=adam_beta_1),
            loss=loss_disc
        )

        self.generator.compile(
            optimizer=Adam(lr=learning_rate,beta_1=adam_beta_1),
            loss=loss_gen)

        self.latent = Input(shape=(latent_size, ))
        self.image_class = Input(shape=(1,), dtype='int32')

        self.fake = self.generator(self.latent)

        discriminator.trainable = False
        
        self.fake = discriminator(self.fake)
        self.combined = Model(inputs=self.latent, outputs=self.fake)
        
        self.combined.compile(
            optimizer=Adam(lr=learning_rate, beta_1=adam_beta_1),
            loss=loss_gen
        )


    def train_single_batch(self, imgs_real):
        
        # train discriminator
        noise = np.random.uniform(-1, 1, (self.batch_size, 
                                          self.latent_size))
        
        imgs_gen = self.generator.predict(noise)
        
        X = np.concatenate((imgs_real, imgs_gen))
        y = np.array([1] * self.batch_size + [0] * self.batch_size)
        
        disc_loss = self.discriminator.train_on_batch(X, y)
        
        # train generator
        X = np.random.uniform(-1, 1, (2 * self.batch_size, self.latent_size))
        y = np.ones(2 * self.batch_size)
        gen_loss = self.combined.train_on_batch(X, y)
        
        return disc_loss, gen_loss
    
    
    def save_model(self, epoch_str):
        
        print "save model..."
        
        path = os.path.join(self.project_name, "model")
        
        fname = os.path.join(path, "g_" + epoch_str + ".h5")
        self.generator.save_weights('my_model.h5')
        
        fname = os.path.join(path, "d_" + epoch_str + ".h5")
        self.discriminator.save_weights(fname)
        
        
    
    def train(self, X_train, X_test, n_epochs = 100):
        
        iterator = get_data_iterator(X_train, self.batch_size)
        iters_per_epoch = len(X_train) / self.batch_size
        
        for epoch in range(n_epochs):
            
            e_str = str(epoch).zfill(4)
            
            d_loss_list = []
            g_loss_list = []
            for iteration in tqdm(range(iters_per_epoch)):
                
                disc_loss, gen_loss = self.train_single_batch(iterator.next())
                
                d_loss_list.append(disc_loss)
                g_loss_list.append(gen_loss)
        
            # get mean train loss
            d_loss = np.mean(np.array(d_loss_list))
            g_loss  = np.mean(np.array(g_loss_list))
            
            # get discriminator test loss
            noise = np.random.uniform(-1, 1, (len(X_test), self.latent_size))
            g_imgs = self.generator.predict(noise)
            X = np.concatenate((X_test, g_imgs))
            y = np.array([1] * len(X_test) + [0] * len(X_test))
            d_test_loss = self.discriminator.evaluate(X, y, verbose = False)
            
            # save generated imgs
            noise = np.random.uniform(-1, 1, (self.batch_size, self.latent_size))
            g_imgs = self.generator.predict(noise)
            fname = os.path.join(self.project_name, "generated_img", e_str + ".png")
            visualize.save_g_imgs(fname, g_imgs)
            
            # show / save training status
            fname = os.path.join(self.project_name, "history.npy")
            visualize.show_training_status(fname, epoch, disc_loss, 
                                           gen_loss, d_test_loss)
            
            # save model
            if epoch % self.save_model_step == 0:
                self.save_model(e_str)
        
        
        
    
    