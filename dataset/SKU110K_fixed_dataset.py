import csv
import cv2
import os
import numpy as np


def download_dataset():
    # download SKU110K_fixed dataset
    #!gdown 1iq93lCdhaPUN0fWbLieMtzfB1850pKwd
    #!tar -xzvf "/content/SKU110K_fixed.tar.gz" -C "/content/" &> /dev/null
    #!rm "/content/SKU110K_fixed.tar.gz"
    #!mv "/content/SKU110K_fixed" "/content/SKU110K" 
    
    
    import gdown
    url = "https://drive.google.com/u/0/uc?id=1iq93lCdhaPUN0fWbLieMtzfB1850pKwd&export=download"
    #output = "test.png"
    gdown.download(url, output)

    import tarfile
    with tarfile.open('SKU110K_fixed.tar.gz') as compressed_folder: 
        compressed_folder.extractall()

    path_to_ds = "/content/SKU110K"
    return path_to_ds, "{}/annotations".format(path_to_ds)


#Modified code from github:   SKU110K_CVPR19/object_detector_retinanet/keras_retinanet/preprocessing/csv_generator.py
##########################
def _parse(value, function, fmt):
  return function(value)

def _read_annotations(csv_reader, classes):
    """ Read annotations from the csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        img_file, x1, y1, x2, y2, class_name, width, height = row[:]
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        width = int(width)
        height = int(height)

        if x1 >= width:
            x1 = width -1
        if x2 >= width:
            x2 = width -1

        if y1 > height:
            y1 = height -1
        if y2 >= height:
            y2 = height -1
        # x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0
        if x1 < 0 or y1 < 0 or x2 <= 0 or y2 <= 0:
            print("Warning: Image file {} has some bad boxes annotations".format(img_file))
            continue

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue

        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})
    return result
##########################


def _get_annotations(annotations_path): 
    # classes = []
    # with open(annotations_path, mode='r') as csv_file:
    #   annotations_train_csv = csv.reader(csv_file, delimiter=',')
    #   for row in annotations_train_csv:
    #     if row[5] not in classes:
    #       classes.append(row[5])  #Only object!
    classes = ["object"]      
    with open(annotations_path, mode='r') as csv_file:
        annotations_train_csv = csv.reader(csv_file, delimiter=',')
        result  = _read_annotations(annotations_train_csv, classes)
    return result


def get_image(img_path):
    return cv2.imread(img_path)


def print_img_with_GT_BB(img_path,annotations_path,which_set):

    #annotations_path ="{}/annotations/annotations_{}.csv".format(path_to_ds,which_set)
    specific_ann_path = os.join(annotations_path,"annotations_{}.csv".format(which_set))

    # read image
    img = get_image(img_path)

    result = _get_annotations(specific_ann_path)

    for row in result[img_path.split('/')[-1]]:
        x1,x2,y1,y2 = row['x1'],row['x2'],row['y1'],row['y2']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.imshow("bounding_box", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

