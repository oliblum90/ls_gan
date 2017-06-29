import os
import keras
from utils.init_project import init_project
from keras.preprocessing.image import ImageDataGenerator

class Trainer:
    
    def __init__(self,
                 project_name,
                 model,
                 batch_size      = 512,
                 n_epochs        = 100,
                 save_model_step = 1,
                 only_test       = False):

        # init project
        project_name = os.path.join("projects", project_name)
        if not only_test:
            init_project(project_name)
            init_project(os.path.join(project_name, "log"))
            init_project(os.path.join(project_name, "model"))
        else:
            model.load_weights(os.path.join(project_name, "model"))
        
        # init class variables
        self._model = model
        self._path_log = os.path.join(project_name, "log")
        self._path_model = os.path.join(project_name, "model")
        self.bs = batch_size
        self.n_epochs = n_epochs
        
        # init save checkpoints callback
        model_fname = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        model_fname = os.path.join(self._path_model, model_fname)
        self._cp = keras.callbacks.ModelCheckpoint(model_fname,
                                                   period = save_model_step)
        
        # init tensorboard logging callback
        self._tl = keras.callbacks.TensorBoard(log_dir=self._path_log)
        
        
    def fit(self, X, y, Xt, yt):
        
        self._model.fit(x = X, y = y, 
                        batch_size=self.bs,
                        epochs = self.n_epochs,
                        validation_data = (Xt, yt), 
                        callbacks = [self._cp, self._tl])
        
    def fit_occ(self, X, y, Xt, yt):
        
        # occlussion
        data_train = ImageDataGenerator(shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)

        data_train.fit(X)
        data_train = data_train.flow(X, y, batch_size=self.bs)

        data_test = ImageDataGenerator()
        data_test = data_test.flow(Xt, yt, batch_size=self.bs)

        train_steps = len(y) / self.bs
        test_steps = len(yt) / self.bs
    
        # fit
        self._model.fit_generator(data_train, 
                                  validation_data=data_test,
                                  steps_per_epoch= train_steps, 
                                  validation_steps= test_steps,
                                  epochs=self.n_epochs, 
                                  callbacks = [self._cp, self._tl])
        
        
        
        