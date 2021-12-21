import cv2
import numpy as np

#pad is value of padding and is initialized to zero
#kernel_rows
#kernel_cols

def padding(input_image, pad = 0, kernel_rows = 0, kernel_cols = 0):
    
    #pad = 0 for zero padding
    #pad = 1 for wrap around
    #pad = 2 for copy edge
    #pad = 3 for reflect across edge
    
    input_dims = (input_image.shape[0], input_image.shape[1])
    
    #individual padding values for rows
    top_row_pad_value = int(np.ceil((kernel_rows-1)/2))
    bottom_row_pad_value = int(np.floor((kernel_rows-1)/2))
    
    #individual padding values for cols
    left_col_pad_value = int(np.ceil((kernel_cols-1)/2))
    right_col_pad_value = int(np.floor((kernel_cols-1)/2))
    
    if pad == 0:
        #zero padding
        padded_dims = (input_image.shape[0] + top_row_pad_value + bottom_row_pad_value, 
                       input_image.shape[1] + left_col_pad_value + right_col_pad_value)
        
        padded_image = np.zeros(padded_dims)
        
        #inserting the original image to the zeros image
        padded_image[top_row_pad_value : top_row_pad_value + input_image.shape[0], 
                     left_col_pad_value : left_col_pad_value + input_image.shape[1]] = input_image
        
    if pad == 1:
        
        #wrap around
        padded_dims = (input_image.shape[0] + top_row_pad_value + bottom_row_pad_value, 
                       input_image.shape[1] + left_col_pad_value + right_col_pad_value)
        
        padded_image = np.zeros(padded_dims)
        
        #inserting the original image to the zeros image
        padded_image[top_row_pad_value : top_row_pad_value + input_image.shape[0], 
                     left_col_pad_value : left_col_pad_value + input_image.shape[1]] = input_image
        
        
        #upper most row padding
        if top_row_pad_value != 0:
            padded_image[0 : top_row_pad_value, : ] = padded_image[-1*(top_row_pad_value+bottom_row_pad_value) : top_row_pad_value + input_image.shape[0], : ]
        #lowest row padding
        if bottom_row_pad_value != 0:
            padded_image[-1*(bottom_row_pad_value) : , : ] = padded_image[top_row_pad_value : top_row_pad_value + bottom_row_pad_value, :]
        #right most column padding
        if right_col_pad_value != 0:
            padded_image[ : ,-1*(right_col_pad_value) : ] = padded_image[ : ,left_col_pad_value : left_col_pad_value + right_col_pad_value]
        #left most column padding
        if left_col_pad_value != 0:
            padded_image[ : ,0 : left_col_pad_value] = padded_image[ : ,-1*(left_col_pad_value+right_col_pad_value) : left_col_pad_value + input_image.shape[1]]
    
    if pad == 2:
        
        #copy edge
        padded_dims = (input_image.shape[0] + top_row_pad_value + bottom_row_pad_value, 
                       input_image.shape[1] + left_col_pad_value + right_col_pad_value)
        
        #print(padded_dims)
        padded_image = np.zeros(padded_dims)#.astype(int)
        #print("Dimensions of the padded image:", padded_image.shape)
        
        #inserting the original image to the zeros image
        padded_image[top_row_pad_value : top_row_pad_value + input_image.shape[0], 
                     left_col_pad_value : left_col_pad_value + input_image.shape[1]] = input_image
        
        
        #upper most row padding
        if top_row_pad_value != 0:
            padded_image[0 : top_row_pad_value, : ] = padded_image[[top_row_pad_value], : ]
        #lowest row padding
        if bottom_row_pad_value != 0:
            padded_image[-1*(bottom_row_pad_value) : , : ] = padded_image[[-1*bottom_row_pad_value-1], :]
        #right most column padding
        if right_col_pad_value != 0:
            padded_image[ : ,-1*(right_col_pad_value) : ] = padded_image[ : ,[-1*(right_col_pad_value)-1]]
        #left most column padding
        if left_col_pad_value != 0:
            padded_image[ : ,0 : left_col_pad_value] = padded_image[ : ,[left_col_pad_value]]
            
    if pad == 3:

        #reflect across edge
        padded_dims = (input_image.shape[0] + top_row_pad_value + bottom_row_pad_value, 
                       input_image.shape[1] + left_col_pad_value + right_col_pad_value)

        #print(padded_dims)
        padded_image = np.zeros(padded_dims)#.astype(int)
        #print("Dimensions of the padded image:", padded_image.shape)

        #inserting the original image to the zeros image
        padded_image[top_row_pad_value : top_row_pad_value + input_image.shape[0], 
                     left_col_pad_value : left_col_pad_value + input_image.shape[1]] = input_image


        #upper most row padding
        if top_row_pad_value != 0:
            padded_image[0 : top_row_pad_value, : ] = np.flip(padded_image[top_row_pad_value : 2*top_row_pad_value, :],axis = 0)
        #lowest row padding
        if bottom_row_pad_value != 0:
            padded_image[-1*(bottom_row_pad_value) : , : ] = np.flip(padded_image[-2*(bottom_row_pad_value) : -1*(bottom_row_pad_value), : ], axis = 0)
        #right most column padding
        if right_col_pad_value != 0:
            padded_image[ : ,-1*(right_col_pad_value) : ] = np.flip(padded_image[ : ,-2*(right_col_pad_value) : -1*(right_col_pad_value)], axis = 1)
        #left most column padding
        if left_col_pad_value != 0:
            padded_image[ : ,0 : left_col_pad_value] = np.flip(padded_image[ : ,left_col_pad_value : 2* left_col_pad_value], axis = 1)

    return padded_image


def convolution(f,w):
    
    return np.sum(f*w)

def conv2(f,w,pad = 0):
    
    if len(f.shape)< 3:
        
        image_padded = padding(f, pad, w.shape[0], w.shape[1])
        
        conv_image = np.zeros((f.shape[0], f.shape[1]))
        
        for row in range(conv_image.shape[0]):
            
            for col in range(conv_image.shape[1]):
                
                conv_image[row][col] = convolution(image_padded[row:row+w.shape[0],col:col+w.shape[1]],w)
                
        return conv_image
                
        
    elif len(f.shape) == 3:
        
        b,g,r = cv2.split(f)
        
        image_padded_b = padding(b, pad, w.shape[0], w.shape[1]) 
        image_padded_g = padding(g, pad, w.shape[0], w.shape[1]) 
        image_padded_r = padding(r, pad, w.shape[0], w.shape[1])
        
        conv_image_b = np.zeros((b.shape[0],b.shape[1]))
        conv_image_g = np.zeros((g.shape[0],g.shape[1]))
        conv_image_r = np.zeros((r.shape[0],r.shape[1]))
        
        for row in range(conv_image_b.shape[0]):
            
            for col in range(conv_image_b.shape[1]):
                
                conv_image_b[row][col] = convolution(image_padded_b[row:row + w.shape[0], col:col + w.shape[1]],w)
                conv_image_g[row][col] = convolution(image_padded_g[row:row + w.shape[0], col:col + w.shape[1]],w)
                conv_image_r[row][col] = convolution(image_padded_r[row:row + w.shape[0], col:col + w.shape[1]],w)
                
        conv_image = cv2.merge((conv_image_b, conv_image_g, conv_image_r))
        #print(conv_image.dtype)
        
        return conv_image