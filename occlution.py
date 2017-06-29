from keras.preprocessing.image import ImageDataGenerator


def get_occ_gen(X, y, bs):

    data = ImageDataGenerator(rescale=1. / 255,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True)
    
    data.fit(X)
    data = data.flow(X, y, batch_size=bs)
    
    steps = len(X) / bs
    
    return data, steps
    
    
def norm(X, y, bs):
    
    data = ImageDataGenerator(rescale=1. / 255)
    data = data.flow(X, y, batch_size=bs)
    
    steps = len(X) / bs
    
    return data, steps