# IMPORTANT:
#  - Tested on Keras 1.2.2 (not guaranteed to work on v2)
#  - Must use Theano backend (change this in ~/.keras/keras.json)
#  - Must use "th" image dimension ordering (change this in ~/.keras/keras.json)
#
# See main() for example of use.
#
# Credit for AlexNet model port to Keras: https://github.com/heuritech/convnets-keras/
#
# LICENSE INFO:
#
# Copyright (c) 2016 Heuritech
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
os.environ["KERAS_BACKEND"] = 'theano'

from keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Lambda
from keras.engine import Layer


class Softmax4D(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        axis_index = self.axis % len(input_shape)
        return tuple([input_shape[i] for i in range(len(input_shape)) \
                      if i != axis_index ])


def crosschannelnormalization(alpha = 1e-4, k=2, beta=0.75, n=5,**kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """
    def f(X):
        b, ch, r, c = X.shape
        # b, ch, r, c = X.get_shape()
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1))
                                              , (0,half))
        extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:,i:i+ch,:,:]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape:input_shape,**kwargs)


def splittensor(axis=1, ratio_split=1, id_split=0,**kwargs):
    def f(X):
        div = X.shape[axis] // ratio_split

        if axis == 0:
            output =  X[id_split*div:(id_split+1)*div,:,:,:]
        elif axis == 1:
            output =  X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:,:,id_split*div:(id_split+1)*div,:]
        elif axis == 3:
            output = X[:,:,:,id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")

        return output

    def g(input_shape):
        output_shape=list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f,output_shape=lambda input_shape:g(input_shape),**kwargs)


def AlexNet(num_outputs):
    inputs = Input(shape=(3,227,227))

    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(num_outputs,name='dense_3')(dense_3)
    prediction = Activation("softmax",name="softmax")(dense_3)

    model = Model(input=inputs, output=prediction)
    return model


def main():
    # NOTE: Must use Keras 1.2.2 with Theano backend, which can be set in ~/.keras/keras.json,
    # or with os.environ['KERAS_BACKEND'] = 'theano', before importing keras
    from keras.applications.imagenet_utils import preprocess_input
    import numpy as np
    from scipy.misc import imread, imresize

    K.set_image_dim_ordering("th")

    # Set num_outputs according to the number of classes for the experiment (should be 2 for Problems A, B, or C):

    # Example using Problem A:
    model = AlexNet(num_outputs=2)

    # Load the provided trained weights
    model.load_weights("/PATH/TO/weights_ProblemA_FOLD_1.hdf5")

    # Load an ultrasound image of some myopathy
    im = imread("/PATH/TO/ULTRASOUND/IMAGE.png")

    # Resize image to to match network input resolution of 227x227
    im = imresize(im, (227, 227))

    # Convert from uint8 to float32
    im = im.astype(np.float32)

    # Switch the dimension containing image color channels to first dimension (matches theano-style image dim ordering)
    im = np.transpose(im, (2, 0, 1))

    # Reshape image from (channels, height, width, channels) to (1, channels, height, width) to represent a batch size of 1
    # (you can classify multiple images in a larger batch if you wish)
    im = np.expand_dims(im, axis=0)

    # Use the standard preprocessing function for ImageNet images (since we use AlexNet)
    im = preprocess_input(im, dim_ordering='th')

    # Perform forward pass through network, and get probabilities for each class
    y_pred_proba = model.predict(im, batch_size=1)[0]

    # Get classification label
    y_pred = np.argmax(y_pred_proba)

    print "Classified image as class", y_pred, "with confidence of", y_pred_proba[y_pred]

if __name__ == '__main__':
    main()
