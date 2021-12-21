from convolution import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

s, k = 1, 2 #  generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and standard deviation = s
pdf = [np.exp(-x*x/(2*s*s))/np.sqrt(2*np.pi*s*s) for x in range(-k,k+1)] 
kernel = np.outer(pdf, pdf)
kernel = kernel/np.sum(kernel)

def ComputePyr(input_image, num_layers):

    count = 0
    image_list_gaussian = [input_image.astype('float32')]

    while count < num_layers:

        blurred_image = conv2(input_image, kernel, 0)
        blurred_image = np.float32(blurred_image)

        height = input_image.shape[0]//2
        width = input_image.shape[1]//2

        output_gaussian = cv2.resize(blurred_image, (width, height), interpolation=cv2.INTER_NEAREST)

        image_list_gaussian.append(output_gaussian)

        count += 1

        input_image = output_gaussian

    for i in range(len(image_list_gaussian)):

        image_list_gaussian[i] = np.float32(image_list_gaussian[i])

    image_list_laplacian = []

    for i in range(len(image_list_gaussian)):
    
        image_list_laplacian.append(image_list_gaussian[i].copy()) 

    for i in range(len(image_list_gaussian)-1,0,-1):

        width = image_list_gaussian[i-1].shape[1]
        height = image_list_gaussian[i-1].shape[0]
        
        resized_image = cv2.resize(image_list_gaussian[i], (width,height), interpolation=cv2.INTER_NEAREST)
        
        blurred_image_laplacian = conv2(resized_image, kernel, 0)
        blurred_image_laplacian = np.float32(blurred_image_laplacian)
        
        difference_image = image_list_gaussian[i-1] - blurred_image_laplacian
        
        image_list_laplacian[i-1] = difference_image.astype('float32')

    return image_list_gaussian, image_list_laplacian