
def one_class_gen(latent_size = 256, 
                  n_upsampling_layers = 3,
                  filter_multiplier = 64):
    """
    Generator for GAN model
    
    Parameters
    ----------
    
    n_upsampling_layers : int
        number of upsampling layers. The output size can be computed by
        4 * 2 ** (n_upsampling_layers)
        
    filter_multiplier : int
        number which determines model complexity
    """
        
    latent = Input(shape=(latent_size, ))
    
    cnn = Sequential()
    
    mult = 2 ** (n_upsampling_layers - 1)
    cnn.add(Dense(4 * 4 * filter_multiplier * mult, 
                  input_dim=latent_size, 
                  activation='linear'))
    cnn.add(Reshape((4, 4, filter_multiplier * mult)))

    # upsampling layers
    for us_layer in range(n_upsampling_layers - 1):
        
        mult = mult / 2
        cnn.add(UpSampling2D(size=(2, 2)))
        cnn.add(Convolution2D(filter_multiplier * mult, 
                              (5, 5), 
                              padding='same',
                              activation='relu', 
                              kernel_initializer='glorot_normal'))    

    # last upsampling layer with 3 filters (for the three color channels)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(3, (5, 5), padding='same',
                          activation='tanh', kernel_initializer='glorot_normal'))

    fake_image = cnn(latent)
    
    return Model(inputs=latent, outputs=fake_image)