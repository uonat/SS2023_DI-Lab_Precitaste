import os
import numpy as np

def select_random(img_names, n):
    selected_img_names = np.random.choice(img_names, size=n, replace=False)
    return selected_img_names.tolist()

def select_uniform(img_names, n):
    num_images = len(img_names)
    img_names.sort()
    step_size = num_images // n
    return [img_names[i*step_size] for i in range(n)]

def select_per_cam(img_names, n):
    selected_img_names = []
    per_cam_img = {'camera0':[], 'camera1':[], 'camera2':[], 'camera3':[]}
    for img_name in img_names:
        for camera_key in per_cam_img:
            if camera_key in img_name:
                per_cam_img[camera_key].append(img_name)
                break
    
    for camera_key in per_cam_img:
        camera_imgs = select_random(per_cam_img[camera_key], n)
        selected_img_names += camera_imgs
    return selected_img_names 


def load_npy_files(parent_dir, npy_file_names):
    features = []
    for feature_file_name in npy_file_names:
        feature_file_path = os.path.join(parent_dir, feature_file_name)
        with open(feature_file_path, 'rb') as npfile:
            np_feat = np.load(npfile, allow_pickle=True)
            features.append(np_feat)
    return features


def make_split(split_method):
    if split_method == 'vanilla':
        train_classes, val_classes = train_test_split(class_names, test_size=VAL_SIZE, random_state=RANDOM_SEED)
    elif split_method == 'base':
        base_class_names = list(set([''.join(cname.split('_')[1:]) for cname in class_names]))
        train_base_classes, val_base_classes = train_test_split(base_class_names, test_size=VAL_SIZE, random_state=RANDOM_SEED)
        train_classes = [cname for cname in class_names if ''.join(cname.split('_')[1:]) in train_base_classes]
        val_classes = [cname for cname in class_names if ''.join(cname.split('_')[1:]) in val_base_classes]
    return train_classes, val_classes