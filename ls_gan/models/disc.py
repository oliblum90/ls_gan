
def one_class_disc(n_downsampling_layers = 3,
                   filter_multiplier = 64):
    """
    Discriminator for GAN model
    
    Parameters
    ----------
    
    n_downsampling_layers : int
        number of downsampling layers. The input size can be computed by
        4 * 2 ** (n_downsampling_layers)
        
    filter_multiplier : int
        number which determines model complexity
    """
    
    cnn = Sequential()

    # first downsampling layer
    ipt_size = 4 * 2 ** (n_downsampling_layers)
    cnn.add(Convolution2D(filter_multiplier, (3, 3), padding='same', strides=(2, 2),
                          input_shape=(ipt_size, ipt_size, 3)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    # downsampling layers
    mult = 1
    for us_layer in range(n_downsampling_layers - 1):
        
        mult = mult * 2
        cnn.add(Convolution2D(filter_multiplier * mult, (3, 3), 
                              padding='same', 
                              strides=(2, 2)))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

    # dense layers
    cnn.add(Flatten())
    fake = Dense(1, activation='sigmoid', name='generation')(features)

    return Model(inputs=image, outputs=fake)