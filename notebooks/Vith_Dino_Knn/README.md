# ViT-H + DINOv2 + KNN

This directory contains the notebook to implement our second method: Detecting objects with general object proposer model(ViT-H) and extracting feature vectors for each of the bounding box regions with a pretrained image classification model(DINOv2) and finally classifying each region with KNN model which is trained before with the features of the ground truth training objects. 



## Pipeline

This method is implemented in three different notebooks to fit the execution time into allowed hours of Kaggle resources. After each step, the output is saved so that each module can be modified without requiring to modify previous steps. 

1. [1-vith-dinov2-knn-rpc](1-vith-dinov2-knn-rpc.ipynb): This notebook runs ViTH on RPC validation dataset. Fills `self.img_name, self.img_path, self.pred_bbox, self.pred_score_bbox, self.pred_features, self.is_train` fields of the objects. 

Note that some objects after this notebook might contain `self.pred_bbox and self.pred_score_bbox` equal to `None`. This means that these objects are missed by the object proposer and must be considered as false negatives of the whole system.

Likewise some objects might contain `self.gt_bbox and self.gt_label` equal to `None`. This objects mean that they are found extra by the object proposer and did not match with any ground truth bounding box. They can be false positives for the whole system or eliminated by the later stages. 

        
2. [2-vith-dino-knn](2-vith-dino-knn.ipynb): For each region detected on the first notebook, extracts DINOv2 features. Fills `self.pred_features`
3. [3-vith-dino-knn](3-vith-dino-knn.ipynb): Evaluates method using KNN classifier. Requires extracted train object features. The features extracted by DINOv2 is downloadable from this [link](https://drive.google.com/file/d/149CjK5Rj5t6XnvXFwMKayuvNNsybo0iL/view?usp=sharing).
Fills `self.pred_label, self.class_score`

All notebook share a common data structure: 

```
class Prediction:
    def __init__(self, img_name, img_path, pred_bbox, pred_score_bbox):
        self.img_name = img_name
        self.img_path = img_path
        
        if pred_bbox is not None:
            self.pred_bbox = pred_bbox.tolist()
        else:
            self.pred_bbox = None
            
        self.pred_score_bbox = pred_score_bbox

        # Obtained when prediction matches with a gt bounding box
        self.gt_bbox = None
        self.gt_label = None

        # Extracted vision features. E.g. dinov2 outputs
        self.pred_features = None
        # Is gt_label belong to training subset
        self.is_train = None

        # Obtained from knn, class label
        self.pred_label = None
        # Obtained from knn by measuring mean distance to its predicted label
        self.class_score = None
```

A `Prediction` object is cretead for each detection or undetected ground truth objects in the dataset. They are initially created by the first notebook and saved as json files. Then each other notebook first loads the json files and extends them with new fields. 

You can find the generated files in this [link](https://drive.google.com/file/d/1UAvnxhYeUkO9cRyvgFqDHXnYTN0GU0Fa/view?usp=sharing). This file contains below files:

* VitH
    * train_config: Configuration file that keeps training, validation classes and train and validation image indices
    * vith_res_train_pred_objects: Prediction objects generated from training subset of RPC validation set.
    * vith_res_train_processed_pred_objects: Same as above item, bounding box post processing applied.
    * vith_res_val_pred_objects: Prediction objects generated from validation subset of RPC validation set.
    * vith_res_val_processed_pred_objects: Same as above item, bounding box post processing applied.
* Dino:
    All files under this path are same with the first step. The only difference is that items in this directory has `pred_features` field. 