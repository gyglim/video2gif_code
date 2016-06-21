'''
This module contains the model used in our paper
 Michael Gygli, Yale Song, Liangliang Cao
    "Video2GIF: Automatic Generation of Animated GIFs from Video," IEEE CVPR 2016
'''
__author__ = 'michaelgygli'

from lasagne.layers.shape import PadLayer
from lasagne.layers import InputLayer, DenseLayer
try:
    from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
except ImportError as e:
    print(e)
import pickle
import lasagne
import numpy as np
import theano
import theano.tensor as T
import skimage.transform
from skimage import color
tensor_type = T.TensorType(theano.config.floatX, [False] * 5)



def build_model(input_var=None, batch_size=2, use_cpu_compatible = theano.config.device=='cpu'):
    '''
    Builds Video2GIF model

    @param input_var:
    @param batch_size:
    @param use_cpu_compatible: use CPU compatible layers (i.e. no cuDNN). Default for theano device CPU; otherwise False
    @return: A dictionary containing the network layers, where the output layer is at key 'score'
    '''
    net={}
    net['input'] = InputLayer((batch_size, 3, 16, 112, 112), input_var=input_var)
    if use_cpu_compatible:
        '''
        Slow implementation running on CPU
        Test snip scores: [-0.08948517, -0.01212098]; Time: 11s
        '''
        print('Use slow network implementation (without cuDNN)')
        # ----------- 1st layer group ---------------
        # Pad first, as this layer doesn't support padding
        net['pad']    = PadLayer(net['input'],width=1, batch_ndim=2)
        net['conv1a'] = lasagne.layers.conv.Conv3DLayer(net['pad'], 64, (3,3,3), pad=0,nonlinearity=lasagne.nonlinearities.rectify,flip_filters=True)
        net['pool1']  = lasagne.layers.pool.Pool3Layer(net['conv1a'],pool_size=(1,2,2),stride=(1,2,2))

        # ------------- 2nd layer group --------------
        net['pad2']    = PadLayer(net['pool1'],width=1, batch_ndim=2)
        net['conv2a'] = lasagne.layers.conv.Conv3DLayer(net['pad2'], 128, (3,3,3), pad=0,nonlinearity=lasagne.nonlinearities.rectify)
        net['pool2']  = lasagne.layers.pool.Pool3Layer(net['conv2a'],pool_size=(2,2,2),stride=(2,2,2))

        # ----------------- 3rd layer group --------------
        net['pad3a']    = PadLayer(net['pool2'],width=1, batch_ndim=2)
        net['conv3a'] = lasagne.layers.conv.Conv3DLayer(net['pad3a'], 256, (3,3,3), pad=0,nonlinearity=lasagne.nonlinearities.rectify)
        net['pad3b']    = PadLayer(net['conv3a'],width=1, batch_ndim=2)
        net['conv3b'] = lasagne.layers.conv.Conv3DLayer(net['pad3b'], 256, (3,3,3), pad=0,nonlinearity=lasagne.nonlinearities.rectify)
        net['pool3']  = lasagne.layers.pool.Pool3Layer(net['conv3b'],pool_size=(2,2,2),stride=(2,2,2))

        # ----------------- 4th layer group --------------
        net['pad4a']    = PadLayer(net['pool3'],width=1, batch_ndim=2)
        net['conv4a'] = lasagne.layers.conv.Conv3DLayer(net['pad4a'], 512, (3,3,3), pad=0,nonlinearity=lasagne.nonlinearities.rectify)
        net['pad4b']    = PadLayer(net['conv4a'],width=1, batch_ndim=2)
        net['conv4b'] = lasagne.layers.conv.Conv3DLayer(net['pad4b'], 512, (3,3,3), pad=0,nonlinearity=lasagne.nonlinearities.rectify)
        net['pool4']  = lasagne.layers.pool.Pool3Layer(net['conv4b'],pool_size=(2,2,2),stride=(2,2,2))

        # ----------------- 5th layer group --------------
        net['pad5a']    = PadLayer(net['pool4'],width=1, batch_ndim=2)
        net['conv5a'] = lasagne.layers.conv.Conv3DLayer(net['pad5a'], 512, (3,3,3), pad=0,nonlinearity=lasagne.nonlinearities.rectify)
        net['pad5b']    = PadLayer(net['conv5a'],width=1, batch_ndim=2)
        net['conv5b'] = lasagne.layers.conv.Conv3DLayer(net['pad5b'], 512, (3,3,3), pad=0,nonlinearity=lasagne.nonlinearities.rectify)

        # We need a padding layer, as C3D only pads on the right, which cannot be done with a theano pooling layer
        net['pad']    = PadLayer(net['conv5b'],width=[(0,1),(0,1)], batch_ndim=3)
        net['pool5']  = lasagne.layers.pool.Pool3Layer(net['pad'],pool_size=(2,2,2),pad=(0,0,0),stride=(2,2,2))
        net['fc6-1']  = DenseLayer(net['pool5'], num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)

    else:
        '''
        Fast implementation running on GPU
        Test snip scores:[-0.08948528,-0.01212097]; Time: 0.33s
        '''
        print('Use fast network implementation (cuDNN)')
        # ----------- 1st layer group ---------------
        net['conv1a'] = Conv3DDNNLayer(net['input'], 64, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,flip_filters=False)
        net['pool1']  = MaxPool3DDNNLayer(net['conv1a'],pool_size=(1,2,2),stride=(1,2,2))

        # ------------- 2nd layer group --------------
        net['conv2a'] = Conv3DDNNLayer(net['pool1'], 128, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        net['pool2']  = MaxPool3DDNNLayer(net['conv2a'],pool_size=(2,2,2),stride=(2,2,2))

        # ----------------- 3rd layer group --------------
        net['conv3a'] = Conv3DDNNLayer(net['pool2'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        net['conv3b'] = Conv3DDNNLayer(net['conv3a'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        net['pool3']  = MaxPool3DDNNLayer(net['conv3b'],pool_size=(2,2,2),stride=(2,2,2))

        # ----------------- 4th layer group --------------
        net['conv4a'] = Conv3DDNNLayer(net['pool3'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        net['conv4b'] = Conv3DDNNLayer(net['conv4a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        net['pool4']  = MaxPool3DDNNLayer(net['conv4b'],pool_size=(2,2,2),stride=(2,2,2))

        # ----------------- 5th layer group --------------
        net['conv5a'] = Conv3DDNNLayer(net['pool4'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        net['conv5b'] = Conv3DDNNLayer(net['conv5a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        # We need a padding layer, as C3D only pads on the right, which cannot be done with a theano pooling layer
        net['pad']    = PadLayer(net['conv5b'],width=[(0,1),(0,1)], batch_ndim=3)
        net['pool5']  = MaxPool3DDNNLayer(net['pad'],pool_size=(2,2,2),pad=(0,0,0),stride=(2,2,2))
        net['fc6-1']  = DenseLayer(net['pool5'], num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)

    net['h1']  = DenseLayer(net['fc6-1'], num_units=512,nonlinearity=lasagne.nonlinearities.rectify)
    net['h2']  = DenseLayer(net['h1'], num_units=128,nonlinearity=lasagne.nonlinearities.rectify)
    net['score']  = DenseLayer(net['h2'], num_units=1, nonlinearity=None)

    return net

def set_weights(net,
                c3d_weight_file,
                video2gif_weight_file):
    '''
    set the weights of the given model. We combine C3D and the video2gif weights
    @param net: a lasagne network
    @param c3d_weight_file:
    @param video2gif_weight_file:
    @return:
    '''

    # Get C3D weights
    with open(c3d_weight_file) as f:
        print('Load pretrained weights from %s...' % c3d_weight_file)
        model = pickle.load(f)

    # Get autogif_demo weights and add them to the model weights
    print('Load pretrained autogif_demo weights from %s...' % video2gif_weight_file)
    autogif_weights = np.load(video2gif_weight_file)['arr_0']
    autogif_model=model[0:-4]
    autogif_model.extend(list(autogif_weights))
    print('Set the weights...')
    lasagne.layers.set_all_param_values(net, autogif_model,trainable=True) # Skip fc7 and fc8


######## Below, there are several helper functions to transform (lists of) images into the right format  ######

def get_snips(images,image_mean,start=0, with_mirrored=False):
    '''
    Converts a list of images to a 5d tensor that serves as input to C3D
    Parameters
    ----------
    images: 4d numpy array or list of 3d numpy arrays
        RGB images

    image_mean: 4d numpy array
        snipplet mean (given by C3D)

    start: int
        first frame to use from the list of images

    with_mirrored: bool
        return the snipplet and its mirrored version (horizontal flip)

    Returns
    -------
    caffe format 5D numpy array (serves as input to C3D)

    '''
    assert len(images) >= start+16, "Not enough frames to fill a snipplet of 16 frames"

    # Convert images to caffe format and stack them
    caffe_imgs=map(lambda x: rgb2caffe(x).reshape(1,3,128,171),images[start:start+16])
    snip=np.vstack(caffe_imgs).swapaxes(0,1)

    # Remove the mean
    snip-= image_mean

    # Get the center crop
    snip=snip[:,:,8:120,29:141]
    snip=snip.reshape(1,3,16,112,112)

    if with_mirrored: # Return normal and flipped version
        return np.vstack((snip,snip[:,:,:,:,::-1]))
    else:
        return snip


def rgb2caffe(im, out_size=(128, 171),copy=True):
    '''
    Converts an RGB image to caffe format and downscales it as needed by C3D

    Parameters
    ----------
    im numpy array
        an RGB image
    downscale

    Returns
    -------
    a caffe image (channel,height, width) in BGR format

    '''
    if copy:
        im=np.copy(im)
    if len(im.shape)==2: # Make sure the image has 3 channels
        im = color.gray2rgb(im)

    h, w, _ = im.shape
    im = skimage.transform.resize(im, out_size, preserve_range=True)
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    return np.array(im,theano.config.floatX)