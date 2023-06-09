def from_original_to_resized(current_im_shape:tuple,target_im_shape:tuple,BB_original:list):
    """
        BB_original -> list of original bounding box values (to be converted)
        origanal_im_shape  -> shape of the input image
        target_im_shape -> shape of the target image
    """
    y_o,x_o = current_im_shape
    y_t,x_t = target_im_shape
    x_scale = x_t/x_o
    y_scale = y_t/y_o

    BB_target = []
    for bb in BB_original:
        x1, y1, x2, y2 = bb
        x1 = int(x1 * x_scale) #np.round()
        y1 = int(y1 * y_scale)
        x2 = int(x2 * x_scale)
        y2 = int(y2 * y_scale)
        BB_target.append([x1, y1, x2, y2])
    
    return BB_target



#we can also use:
#import albumentations
#OR
#import torchvision.transforms.functional as TF
#image = TF.resize(image, (target_shape))
#bonding_box_coordinate = TF.resize(bonding_box_coordinate, (target_shape))







