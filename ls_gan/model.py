
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout
from keras.models import Sequential, Model
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.advanced_activations import LeakyReLU

def one_class_generator(latent_size = 100):
    
    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))
    
    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
    cnn.add(Dense(128 * 8 * 8, activation='relu'))
    cnn.add(Reshape((8, 8, 128)))
    
    # upsample to (..., 16, 16)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(256, (5, 5), padding='same',
                          activation='relu', kernel_initializer='glorot_normal'))
    
    # upsample to (..., 32, 32)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(128, (5, 5), padding='same',
                          activation='relu', kernel_initializer='glorot_normal'))
    
    # upsample to (..., 64, 64)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(128, (5, 5), padding='same',
                          activation='relu', kernel_initializer='glorot_normal'))

    # take a channel axis reduction
    cnn.add(Convolution2D(3, (2, 2), padding='same',
                          activation='tanh', kernel_initializer='glorot_normal'))

    fake_image = cnn(latent)
    
    return Model(inputs=latent, outputs=fake_image)


def one_class_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()

    # downsample: (...,64, 64) (..., 16, 16)
    cnn.add(Convolution2D(32, (3, 3), padding='same', strides=(2, 2),
                          input_shape=(64, 64, 3)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    # downsample: (...,32, 32) (..., 16, 16)
    cnn.add(Convolution2D(64, (3, 3), padding='same', strides=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    # downsample: (...,16, 16) (..., 16, 16)
    cnn.add(Convolution2D(128, (3, 3), padding='same', strides=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    # downsample: (...,8, 8) (..., 16, 16)
    cnn.add(Convolution2D(256, (3, 3), padding='same', strides=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(64, 64, 3))

    features = cnn(image)

    fake = Dense(1, activation='sigmoid', name='generation')(features)

    return Model(inputs=image, outputs=fake)





def one_class_generator_2(latent_size = 100):
    
    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))
    
    cnn = Sequential()

    cnn.add(Dense(4 * 4 * 64 * 8, input_dim=latent_size, activation='linear'))

    cnn.add(Reshape((4, 4, 64 * 8)))

    # upsample to (..., 8, 8)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(64 * 4, (5, 5), padding='same',
                          activation='relu', kernel_initializer='glorot_normal'))

    # upsample to (..., 16, 16)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(64 * 2, (5, 5), padding='same',
                          activation='relu', kernel_initializer='glorot_normal'))

    # upsample to (..., 32, 32)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(64 * 1, (5, 5), padding='same',
                          activation='relu', kernel_initializer='glorot_normal'))

    # upsample to (..., 64, 64)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(3, (5, 5), padding='same',
                          activation='tanh', kernel_initializer='glorot_normal'))

    fake_image = cnn(latent)
    
    return Model(inputs=latent, outputs=fake_image)


def one_class_discriminator_2():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()

    # downsample: (...,64, 64) (..., 16, 16)
    cnn.add(Convolution2D(64, (3, 3), padding='same', strides=(2, 2),
                          input_shape=(64, 64, 3)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    # downsample: (...,32, 32) (..., 16, 16)
    cnn.add(Convolution2D(64 * 2, (3, 3), padding='same', strides=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    # downsample: (...,16, 16) (..., 16, 16)
    cnn.add(Convolution2D(64 * 4, (3, 3), padding='same', strides=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    # downsample: (...,8, 8) (..., 16, 16)
    cnn.add(Convolution2D(64 * 8, (3, 3), padding='same', strides=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(64, 64, 3))

    features = cnn(image)

    fake = Dense(1, activation='sigmoid', name='generation')(features)

    return Model(inputs=image, outputs=fake)


def ac_generator(latent_size = 100):
    
    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))
    
    cnn = Sequential()

    cnn.add(Dense(4 * 4 * 64 * 8, input_dim=latent_size, activation='linear'))

    cnn.add(Reshape((4, 4, 64 * 8)))

    # upsample to (..., 8, 8)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(64 * 4, (5, 5), padding='same',
                          activation='relu', kernel_initializer='glorot_normal'))

    # upsample to (..., 16, 16)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(64 * 2, (5, 5), padding='same',
                          activation='relu', kernel_initializer='glorot_normal'))

    # upsample to (..., 32, 32)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(64 * 1, (5, 5), padding='same',
                          activation='relu', kernel_initializer='glorot_normal'))

    # upsample to (..., 64, 64)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(3, (5, 5), padding='same',
                          activation='tanh', kernel_initializer='glorot_normal'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(10, latent_size,
                              init='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = merge([latent, cls], mode='mul')

    fake_image = cnn(h)

    return Model(inputs=[latent, image_class], outputs=fake_image)


def ac_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()

    # downsample: (...,64, 64) (..., 16, 16)
    cnn.add(Convolution2D(64, (3, 3), padding='same', strides=(2, 2),
                          input_shape=(64, 64, 3)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    # downsample: (...,32, 32) (..., 16, 16)
    cnn.add(Convolution2D(64 * 2, (3, 3), padding='same', strides=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    # downsample: (...,16, 16) (..., 16, 16)
    cnn.add(Convolution2D(64 * 4, (3, 3), padding='same', strides=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    # downsample: (...,8, 8) (..., 16, 16)
    cnn.add(Convolution2D(64 * 8, (3, 3), padding='same', strides=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(64, 64, 3))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(10, activation='softmax', name='auxiliary')(features)

    return Model(inputs=image, outputs=[fake, aux])
