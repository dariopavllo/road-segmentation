# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
from keras.layers import LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from helpers import *

class CnnModel:
    
    def __init__(self):
        """ Construct a CNN classifier. """
        
        self.patch_size = 16
        self.window_size = 72
        self.padding = (self.window_size - self.patch_size) // 2
        self.initialize()
        
    def initialize(self):
        """ Initialize or reset this model. """
        patch_size = self.patch_size
        window_size = self.window_size
        padding = self.padding
        nb_classes = 2
        
        # Size of pooling area for max pooling
        pool_size = (2, 2)

        # Compatibility with Theano and Tensorflow ordering
        if K.image_dim_ordering() == 'th':
            input_shape = (3, window_size, window_size)
        else:
            input_shape = (window_size, window_size, 3)

        reg = 1e-6 # L2 regularization factor (used on weights, but not biases)

        self.model = Sequential()

        self.model.add(Convolution2D(64, 5, 5, # 64 5x5 filters
                                border_mode='same',
                                input_shape=input_shape
                               ))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=pool_size, border_mode='same'))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(128, 3, 3, # 128 3x3 filters
                                border_mode='same'
                               ))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=pool_size, border_mode='same'))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(256, 3, 3, # 256 3x3 filters
                                border_mode='same'
                               ))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=pool_size, border_mode='same'))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(256, 3, 3, # 256 3x3 filters
                                border_mode='same'
                               ))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=pool_size, border_mode='same'))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(128, W_regularizer=l2(reg)
                            )) # Fully connected layer (128 neurons)
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(nb_classes, W_regularizer=l2(reg)
                            ))
        #self.model.add(Activation('softmax')) # Not needed since we use logits
        
    
    def train(self, Y, X):
        """
        Train this model with the given dataset.
        """
        
        patch_size = self.patch_size
        window_size = self.window_size
        padding = self.padding
        
        print('Training set shape: ', X.shape)
        samples_per_epoch = X.shape[0]*X.shape[1]*X.shape[2]//256 # Arbitrary value
        
        # Pad training set images (by appling mirror boundary conditions)
        X_new = np.empty((X.shape[0],
                         X.shape[1] + 2*padding, X.shape[2] + 2*padding,
                         X.shape[3]))
        Y_new = np.empty((Y.shape[0],
                         Y.shape[1] + 2*padding, Y.shape[2] + 2*padding))
        for i in range(X.shape[0]):
            X_new[i] = pad_image(X[i], padding)
            Y_new[i] = pad_image(Y[i], padding)
        X = X_new
        Y = Y_new
            
        batch_size = 125
        nb_classes = 2
        nb_epoch = 200

        def softmax_categorical_crossentropy(y_true, y_pred):
            """
            Uses categorical cross-entropy from logits in order to improve numerical stability.
            This is especially useful for TensorFlow (less useful for Theano).
            """
            return K.categorical_crossentropy(y_pred, y_true, from_logits=True)

        opt = Adam(lr=0.001) # Adam optimizer with default initial learning rate
        self.model.compile(loss=softmax_categorical_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])

        np.random.seed(3) # Ensure determinism
        
        def generate_minibatch():
            """
            Procedure for real-time minibatch creation and image augmentation.
            This runs in a parallel thread while the model is being trained.
            """
            while 1:
                # Generate one minibatch
                X_batch = np.empty((batch_size, window_size, window_size, 3))
                Y_batch = np.empty((batch_size, 2))
                for i in range(batch_size):
                    # Select a random image
                    idx = np.random.choice(X.shape[0])
                    shape = X[idx].shape
                    
                    # Sample a random window from the image
                    center = np.random.randint(window_size//2, shape[0] - window_size//2, 2)
                    sub_image = X[idx][center[0]-window_size//2:center[0]+window_size//2,
                                       center[1]-window_size//2:center[1]+window_size//2]
                    gt_sub_image = Y[idx][center[0]-patch_size//2:center[0]+patch_size//2,
                                          center[1]-patch_size//2:center[1]+patch_size//2]
                    
                    # The label does not depend on the image rotation/flip (provided that the rotation is in steps of 90°)
                    threshold = 0.25
                    label = (np.array([np.mean(gt_sub_image)]) > threshold) * 1
                    
                    # Image augmentation
                    # Random flip
                    if np.random.choice(2) == 0:
                        # Flip vertically
                        sub_image = np.flipud(sub_image)
                    if np.random.choice(2) == 0:
                        # Flip horizontally
                        sub_image = np.fliplr(sub_image)
                    
                    # Random rotation in steps of 90°
                    num_rot = np.random.choice(4)
                    sub_image = np.rot90(sub_image, num_rot)

                    label = np_utils.to_categorical(label, nb_classes)
                    X_batch[i] = sub_image
                    Y_batch[i] = label
                
                if K.image_dim_ordering() == 'th':
                    X_batch = np.rollaxis(X_batch, 3, 1)
                    
                yield (X_batch, Y_batch)

        # This callback reduces the learning rate when the training accuracy does not improve any more
        lr_callback = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5,
                                        verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        
        # Stops the training process upon convergence
        stop_callback = EarlyStopping(monitor='acc', min_delta=0.0001, patience=11, verbose=1, mode='auto')
        
        try:
            self.model.fit_generator(generate_minibatch(),
                            samples_per_epoch=samples_per_epoch,
                            nb_epoch=nb_epoch,
                            verbose=1,
                            callbacks=[lr_callback, stop_callback])
        except KeyboardInterrupt:
            # Do not throw away the model in case the user stops the training process
            pass

        print('Training completed')
        
    def save(self, filename):
        """ Save the weights of this model. """
        self.model.save_weights(filename)
        
    def load(self, filename):
        """ Load the weights for this model from a file. """
        self.model.load_weights(filename)
        
    def classify(self, X):
        """
        Classify an unseen set of samples.
        This method must be called after "train".
        Returns a list of predictions.
        """
        # Subdivide the images into blocks
        img_patches = create_patches(X, self.patch_size, 16, self.padding)
        
        if K.image_dim_ordering() == 'th':
            img_patches = np.rollaxis(img_patches, 3, 1)
        
        # Run prediction
        Z = self.model.predict(img_patches)
        Z = (Z[:,0] < Z[:,1]) * 1
        
        # Regroup patches into images
        return group_patches(Z, X.shape[0])
        