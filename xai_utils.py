import cv2
from time import time
import os
import numpy as np
import json

import tensorflow as tf
from tensorflow import keras
from scipy.ndimage.interpolation import zoom
from keras.preprocessing.image import load_img, img_to_array
import keras.backend as K
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
#############################

def layer_finder(k_model, model_arch, pool_input=True):

  '''
  Returns a list of all of the last layers in each block of the model.

    Parameters:
      k_model (Keras model): Either a VGG or ResNet
      model_arch (str): Either "VGG" or "ResNet"

    Returns:
      last_layers (list): A list of all of the last layers in each block of the
      model.
  '''
  
  if type(model_arch) != str:
    raise TypeError("Input argument \"model_arch\" must be a string that is\
                      either \"VGG\" or \"ResNet\".")

  last_layers = []
  pool_flag=False
  block_end_detected=False
  first_layer=True
  j=0

  if model_arch == "VGG":
 
    for layer in k_model.layers:
      if type(layer) == tf.keras.layers.MaxPool2D:
        last_layers.append(layer.name)

  elif model_arch == "ResNet":


    for i in range(len(k_model.layers)):
      if i<j: continue
      #print(k_model.layers[i])
      if len(k_model.layers[i+1].output.get_shape()) < 4:
        # only save a layer if the block before the end was a convolutional block
            last_layers.append(k_model.layers[i].name)
            break
      
      if k_model.layers[i+1].output.get_shape()[2]<k_model.layers[i].output.get_shape()[2]-4:
          if pool_input==True:
              if type(k_model.layers[i]) == tf.keras.layers.InputLayer: continue
              if 'ZeroPadding2D' in str(type(k_model.layers[i])):
                  if type(k_model.layers[i-1]) == tf.keras.layers.InputLayer: continue
                  last_layers.append(k_model.layers[i-1].name)
              else:
                  last_layers.append(k_model.layers[i].name)
          else:
              if first_layer:
                  j=i+1
                  pool_flag=True
                  while(pool_flag):
                      j += 1
                      #print(str(type(k_model.layers[j])))
                      if  'Conv2D' in str(type(k_model.layers[j])):
                          #print('Here')
                          last_layers.append(k_model.layers[j-1].name)
                          first_layer=False
                          pool_flag=False
              else:
                  j=i
                  pool_flag=True
                  while(pool_flag):
                      j += 1
                      #print(str(type(k_model.layers[j])))
                      if 'merge.Add' in str(type(k_model.layers[j])):
                          block_end_detected=True
                          #print(j)
                      elif block_end_detected==True and 'Conv2D' in str(type(k_model.layers[j])):
                          #print('Here')
                          last_layers.append(k_model.layers[j-1].name)
                          block_end_detected=False
                          pool_flag=False
  else:
    
    print("Input argument \"model_arch\" must be either \"VGG\" or \"ResNet\".")

  return [[lay] for lay in last_layers]
  
def create_random_mask(h=7, w=7, H=224, W=224, p_1=0.5, resample=Image.BILINEAR):
    '''
    Generates one random mask utilized in RISE
    inputs:
        h, w: initial size of binary mask
        H, W: final size of the upsampled mask
        p_1: probability of actiating pixels in the down-sampled masks.
        interp: upsampling technique.
    returns:
        mask: a smooth mask with the values in range [0,1] with size of HxW.
    '''
    assert H>h, 'Masks should be resized to higher dimensions.'
    assert W>w, 'Masks should be resized to higher dimensions.'
    # create random binary hxw mask
    mask=np.random.choice([0, 1], size=(h, w), p=[1-p_1, p_1])

    # upsample mask to (h+H,w+W)
    mask = Image.fromarray(mask*255.)
    mask = mask.resize((H + h, W + w), resample=resample)
    mask = np.array(mask)

    # randomly crop mask to HxW
    w_crop = np.random.randint(0,w+1)
    h_crop = np.random.randint(0,h+1)
    mask = mask[h_crop:H + h_crop, w_crop:W + w_crop]

    # normalize between 0 and 1
    mask /= np.max(mask)

    return mask

def create_attribution_masks(img, model, layers, class_index, max_mask_num, interp='bilinear'):
    '''
    Derives feature maps from one, or a couple of layers, and post-processes them
    to convert them to attribution masks.

    inputs:
        img: a 4-D tensor image.
        model: the classification model
        layers: list of layers to be visualized either individually or mutually.
        class_index: the output class according to whom the layer(s) are visualized.
        max_mask_num: the threshold "normalized gradient" value for sampling attribution masks (\mu in our paper)
        interp: upsampling technique.
        For now, 'bilinear' and 'nearest' are supported.
    returns:
        masks: a set of attribution masks normalized between 0 and 1.
    '''
    assert interp in ['bilinear', 'nearest'], 'Selected upsampling type undefined or unsupported.'
    # Forward pass to get attribution masks.
    conv_outputs=[]
    for layer in model.layers:
        if np.isin(layer.name,layers):
            conv_outputs.append(layer.output)
    conv_outputs.append(model.output)
    feedforward1=keras.models.Model([model.input], [conv_outputs])
    with tf.GradientTape() as tape:
        ff_results=feedforward1([img])[0]
        all_fmap_masks, predictions = ff_results[:-1], ff_results[-1]
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, all_fmap_masks)
    ###
    
    # upsample and normalize masks.
    num_masks=0
    masks=[]
    for i in range(len(layers)):
        tmp_mask = all_fmap_masks[i][0].numpy()
        if len(img.shape)==3:
            axis=0
            size=img.shape[1:]
            tmp_mask = np.expand_dims(tmp_mask, axis=1)
        elif len(img.shape)==4:
            axis=(0,1)
            size=img.shape[1:-1]
        significance = np.mean(grads[i][0], axis=axis)
        #idxs = np.argpartition(significance, -1*max_mask_num)[-1*max_mask_num:]
        idxs = np.where(significance>max_mask_num*np.max(significance))[0]
        if interp == 'bilinear':
            fmap = tf.image.resize(tmp_mask[...,idxs], size, method='bilinear').numpy()
        elif interp == 'nearest':
            fmap = tf.image.resize(tmp_mask[...,idxs], size, method='nearest').numpy()
        else: raise ValueError('You have selected an unsupported interpolation type.')
        
        num_masks+=fmap.shape[2]
        fmap -= np.min(fmap, axis=(0,1))
        fmap /= (np.max(fmap, axis=(0,1))+10e-7)
        masks.append(fmap) 
    return masks

def visualize_layers(img, model, class_index, masks, H=224, W=224, C=3, batch_size = 128):
    '''
    Combines attribution masks using the RISE-based framework mentioned in
    SISE white paper.
    inputs:
        img: a 3-D tensor image.
        model: the classification model
        class_index: the output class according to whom the layer(s) are visualized.
        masks: a set of attribution masks normalized between 0 and 1.
    returns:
        sum_masks: visualization map of the selected layer(s).
    This function follows 'create_attribution_masks()'.
    '''
    # creates perturbed images to probe model.
    img = img if len(img.shape)==3 else np.expand_dims(img, axis=1)
    X = np.einsum('hwc,hwn->nhwc', img, masks)
    # second forward pass to valuate attribution maps
    preds_masked = np.empty([0])
    if masks.shape[2] <= batch_size :
      preds_masked=np.append(preds_masked, model(X, training=False)[:,class_index],axis=0)
    else :
      for i in range (0, masks.shape[2]-batch_size, batch_size) :
        preds_masked=np.append(preds_masked, model(X[i:i+batch_size], training=False)[:,class_index],axis=0)
      preds_masked=np.append(preds_masked, model(X[i+batch_size:], training=False)[:,class_index],axis=0)
    
    # Linear combination of attribution masks.
    masks /= (masks.sum(axis=(0,1))+10e-7)
    sum_mask = np.einsum('hwn,n->hw', masks, preds_masked)

    sum_mask -= np.min(sum_mask)
    sum_mask /= np.max(sum_mask)
    return sum_mask
    
def otsu(I, nbins=256, tau=1.5):
    '''
    Finds the optimum adaptive threshold value for a 2-D image.
    inputs:
        I: a 2-D image (visualization map/ heat-map/ etc.)
        nbins: resolution of histogram. Increasing this parameter yields to more
        precise threshold value, achieved in longer time.
        tau: bottleneck amplititude
        returns: Otsu adaptive threshold value
    '''
    I = np.round(I*nbins)
    #histogram of the image
    hist, bins = np.histogram(I.ravel(),nbins,[0,nbins])
    #CDF/ mean/ variance terms for multiple values
    i = np.arange(nbins)
    varsb = np.zeros(nbins)
    for j in range(1, nbins):
        w0 = np.sum(hist[0:j])
        w1 = np.sum(hist[j:nbins])
        u0 = np.sum(np.multiply(hist[0:j], i[0:j])) / w0
        u1 = np.sum(np.multiply(hist[j:nbins], i[j:nbins])) / w1
        varsb[j] = w0 * w1 * (u0-u1) * (u0-u1)
    # the threshold value is the one maximizing the variance term.
    t = np.argmax(varsb)
    #print(t)
    k = round(t*tau)
    if np.sum(hist[int(k):256]) < .1 * np.sum(hist):
        #print('happened')
        return t*tau/nbins
    else:
        return t/nbins

def otsu_sigmoid(I, nbins=256, T=100., tau=1.5):
    '''
        Thresholds the 2-D visualization map softly, combining Otsu's method and
        sigmoid function.
        inputs:
            I: a 2-D image (visualization map/ heat-map/ etc.)
            nbins: resolution of histogram. Increasing this parameter yields to more
            precise threshold value, achieved in longer time.
            T: sigmoid temparature (preferred to be set to high values.)     
        returns:
            the soft-thresholded heat-map according to the input.
    '''
    thr=otsu(I, nbins=256, tau=1.5)
    return 1/(1 + np.exp(-(I-thr)*T)) 

def fuse_visualization_maps(exmaps, fusion_type='otsu', T=100.):
    '''
    Fuses visualization maps to a unique explanation map. Visualization maps should
    be given with the correct order (low-level layer to high-level layer)

    '''
    assert fusion_type in ['simple', 'otsu']
    ex=exmaps[0]
    if fusion_type=='simple':
        for i in range(1, len(exmaps)):
            ex += exmaps[i]
            ex *= exmaps[i]
    elif fusion_type=='otsu':
        for i in range(1, len(exmaps)):
            ex += exmaps[i]
            ex *= otsu_sigmoid(exmaps[i], T=T)
    return ex
    
def SISE(img, model, class_index, layers, grad_thr, interp='bilinear', 
         fusion_type='otsu', T=100.):

    '''
    For now, this function supports VGG16, ResNet50, and ResNet101.
    img: a 4-D image, or a 3-D array.
    model: the classification model
    layers: list of layers to be visualized either individually or mutually.
    interp: upsampling technique.
    Check the supproted upsampling types in function 'create_attribution_masks'.
	grad_thr: Threshold on the average gradient values to select the most appropriate feature maps.
    fusion_type: the fusion technipue for visualization maps:
        simple: Using only addition and multiplication blocks.
        otsu: Using addition, soft otsu threshold, and multiplication blocks.
    auto_layer_finder: if 'True', the layers are automatically selected. Otherwise,
        pre-defined layers for the models experimented are used.
    pool_input_select: If True, the inputs of pooling layers are detected automatically.
        Otherwise,  the outputs of pooling layers are detected automatically.
        If 'auto_layer_finder=False', this parameter is ineffective.
    '''
    masks = create_attribution_masks(img, model, layers, class_index=class_index, max_mask_num = grad_thr, interp=interp)
    exmaps=[]
    for mask_set in masks:
        exmaps.append(visualize_layers(img[0], model, class_index, mask_set))
    return fuse_visualization_maps(exmaps, fusion_type=fusion_type, T=T)
    
    
def weighted_fusion(w,exmaps, T=100.):
    '''
    Objective: weighted fusion using weighted addition, unweighted multiplication, and otsu threshold blocks.
    inputs:
        w: an array of weight factors of length N-1.
        exmaps: a 3-D array of explanation maps of length H x W x N.
    parameters:
        N: number of visualiation maps received
        H x W: size of visualization maps.
    outputs:
        e_out: fused explanation map.
    '''
    #w_post=np.maximum(w,0)
    w_post=np.clip(a=w, a_min=0, a_max=2)
    e23=np.multiply((exmaps[:,:,0]*w_post[0]+exmaps[:,:,1]*(2-w_post[0])),
                    otsu_sigmoid(exmaps[:,:,1], T=T))
    e234=np.multiply((e23*w_post[1]+exmaps[:,:,2]*(2-w_post[1])),
                    otsu_sigmoid(exmaps[:,:,2], T=T))
    e2345=np.multiply((e234*w_post[2]+exmaps[:,:,3]*(2-w_post[2])),
                    otsu_sigmoid(exmaps[:,:,3], T=T))
    e23456=np.multiply((e2345*w_post[3]+exmaps[:,:,4]*(2-w_post[3])),
                    otsu_sigmoid(exmaps[:,:,4], T=T))
    e_out = e23456
    return e_out

def grad_cam(input_model, image, layer_name):
    cls = np.argmax(input_model.predict(image))
    def normalize(x):
        """Utility function to normalize a tensor by its L2 norm"""
        return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output
    conv_output = input_model.get_layer(layer_name).output
    feedforward1 = keras.models.Model([input_model.input], [conv_output, y_c])
    with tf.GradientTape() as tape:
        ff_results=feedforward1([image])
        all_fmap_masks, predictions = ff_results[0], ff_results[-1]
        loss = predictions[:, cls]
    grads_val = tape.gradient(loss, all_fmap_masks)
    if len(image.shape)==3:
        axis=(0, 1)
    elif len(image.shape)==4:
        axis=(0, 1, 2)
    weights = np.mean(grads_val, axis=axis)
    cam = np.dot(all_fmap_masks[0], weights)
    #print (cam)
    H,W= image.shape[1:3]
    cam = np.maximum(cam, 0)
    #cam = resize(cam, (H, W))
    cam = zoom(cam,H/cam.shape[0])
    #cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam
	
def RISE(img, model, class_index, N_MASKS=8000, H=224, W=224, C=3):
    '''
	img: a 3-D input image
	model: a trained model
	class_index; The class of interest
	N_MASKS: The number of random masks to be generated
	H,W,C: The desired dimensions of the random masks
	'''
    X = np.zeros(shape=(N_MASKS, H, W, C), dtype=np.float32)
    masks = np.zeros((N_MASKS,H,W), dtype=np.float32)
    #for i in tqdm(range(N_MASKS)):
    for i in range(N_MASKS):
        m =create_random_mask(H=H, W=W)
        masks[i] = m
        x = img.copy()
        x[:, :, 0] *= m
        x[:, :, 1] *= m
        x[:, :, 2] *= m
        X[i] = x
    preds_masked = model.predict(X, verbose=0)
    sum_mask = np.zeros(masks[0].shape, dtype=np.float32)

    # np.einsum???
    for i, mask in enumerate(masks):
        m = mask * preds_masked[i, class_index]
        sum_mask += m

    sum_mask -= np.min(sum_mask)
    sum_mask /= np.max(sum_mask)
    return sum_mask