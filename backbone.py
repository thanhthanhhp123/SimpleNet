from lib import *

def identity_block(X, f, filters, training = True, initializer = random_uniform):
    F1, F2, F3 = filters
    X_shortcut = X


    X = Conv2D(
        filters = F1,
        kernel_size = (1, 1),
        strides = (1, 1),
        padding = 'valid',
        kernel_initializer = initializer(seed = 0)
    )(X)
    X = BatchNormalization(axis = 3)(X, training = training)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(
        filters = F2,
        kernel_size = (f, f),
        strides = (1, 1),
        padding = 'same',
        kernel_initializer = initializer(seed = 0)
    )(X)
    X = BatchNormalization(axis = 3)(X, training = training)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(
        filters = F3,
        kernel_size = (1, 1),
        strides = (1, 1),
        padding = 'valid',
        kernel_initializer = initializer(seed = 0)
    )(X)
    X = BatchNormalization(axis = 3)(X, training = training)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def convolutional_block(X, f, filters, s = 2, training = True, initializer = glorot_uniform):
    F1, F2, F3 = filters
    X_shortcut = X

    # First component of main path
    X = Conv2D(
        filters = F1,
        kernel_size = (1, 1),
        strides = (s, s),
        padding = 'valid',
        kernel_initializer = initializer(seed = 0)
    )(X)
    X = BatchNormalization(axis = 3)(X, training = training)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(
        filters = F2,
        kernel_size = (f, f),
        strides = (1, 1),
        padding = 'same',
        kernel_initializer = initializer(seed = 0)
    )(X)
    X = BatchNormalization(axis = 3)(X, training = training)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(
        filters = F3,
        kernel_size = (1, 1),
        strides = (1, 1),
        padding = 'valid',
        kernel_initializer = initializer(seed = 0)
    )(X)
    X = BatchNormalization(axis = 3)(X, training = training)

    # Shortcut path
    X_shortcut = Conv2D(
        filters = F3,
        kernel_size = (1, 1),
        strides = (s, s),
        padding = 'valid',
        kernel_initializer = initializer(seed = 0)
    )(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut, training = training)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def ResNet50(input_shape = (64, 64, 3)):
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    ## Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2) 
    X = identity_block(X, 3, [128, 128, 512]) 
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    
    ## Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024],s=2) 
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    ## Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[512, 512, 2048],s=2) 
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

if __name__ == '__main__':
    model = ResNet50()
    model.summary()