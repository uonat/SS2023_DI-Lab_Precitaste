import os
import cv2
import json

def get_annots(json_file_path):
  with open(json_file_path, 'r') as jfile:
    annot = json.load(jfile)
  return annot

class RPCDataset:
  def __init__(self, parent_dir, split):
    if split == 'train':
      self.imgs_dir = os.path.join(parent_dir, 'train2019')
      annot_path = os.path.join(parent_dir, 'instances_train2019.json')
    elif split == 'val':
      self.imgs_dir = os.path.join(parent_dir, 'val2019')
      annot_path = os.path.join(parent_dir, 'instances_val2019.json')
    elif split == 'test':
      self.imgs_dir = os.path.join(parent_dir, 'test2019')
      annot_path = os.path.join(parent_dir, 'instances_test2019.json')
    else:
      raise ValueError("split argument must be either train, val or test")

    assert os.path.exists(self.imgs_dir), "Image directory does not exists on {}".format(self.imgs_dir)
    assert os.path.exists(annot_path), "Annotation json does not exists on {}".format(annot_path)
    
    self.annots = get_annots(annot_path)
    self.num_images = len(self.annots['images'])
    self.num_classes = len(self.annots['categories'])

    # Create placeholders
    self.img_info = []
    self.annot_info = [[] for _ in range(len(self.annots['images']))]
    self.category_info = [{}] * self.num_classes
    # Ids in the validation images are not consecutive and incremented by 1
    # To keep the order in images and annotations this dictionary use mapping
    img_id_to_idx = {}

    # Fill image, annotation and category dictionaries
    for annot in self.annots['images']:
      idx = annot['id']
      self.img_info.append(annot)
      img_id_to_idx[idx] = len(self.img_info)-1

    for annot in self.annots['annotations']:
      img_idx = annot['image_id']
      img_info_inserted_idx = img_id_to_idx[img_idx]
      self.annot_info[img_info_inserted_idx].append(annot)

    for cat in self.annots['__raw_Chinese_name_df']:
      cur_cat_info = {
          'id': cat['category_id'],
          'sku_class': cat['sku_class'],
          'sku_name': cat['sku_name'], 
          'name': cat['name'],
          'class': cat['clas']
          }
      # Category ids start from 1          
      self.category_info[cat['category_id']-1] = cur_cat_info


  def get_num_imgs(self):
    return self.num_images
  
  def get_img_by_id(self, img_idx):
    img_path = self.get_img_path_by_id(img_idx)
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  def get_img_path_by_id(self, img_idx):
    cur_img_info = self.img_info[img_idx]
    img_path = os.path.join(self.imgs_dir, cur_img_info['file_name'])
    return img_path

  def get_annots_by_img_id(self, img_idx, key_for_category='sku_name'):
    """
    Returns the bounding box annotations of the image and their category labels
    @img_idx: Index of the image to return its annotations
    @key_for_category: Type of the class name to return. Pass 'name' for chinese
    category, 'class' for chinese base category, 'sku_class' for english base
    category, 'sku_name' for english category. Default is sku_name
    Return type of the function is a list of tuples. Each element contains
    the bounding box and corresponding class name of that bbox. 
    The bounding boxes are in the form of [x, y, w, h] 
    """
    cur_annot_info = self.annot_info[img_idx]
    cur_annots = []

    for annot in cur_annot_info:
      cat_id = annot['category_id']
      class_name = self.category_info[cat_id-1][key_for_category]
      
      cur_annots.append((annot['bbox'], class_name))
    return cur_annots

  def get_element_by_id(self, img_idx, key_for_category='sku_name'):
    """
    Returns the element at the given index. Element contains both image and
    annotations along with the width and height of the image 
    """
    annots = self.get_annots_by_img_id(img_idx, key_for_category)
    cur_img_info = self.img_info[img_idx]
    img_path = os.path.join(self.imgs_dir, cur_img_info['file_name'])
    return {
        'img': cv2.imread(img_path),
        'annots': annots,
        'width': cur_img_info['width'],
        'height':cur_img_info['height']
    }