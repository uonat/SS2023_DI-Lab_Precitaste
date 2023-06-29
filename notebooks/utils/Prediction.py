import json

class Prediction:
    def __init__(self, img_name, img_path, pred_bbox, pred_score_bbox):
        self.img_name = img_name
        self.img_path = img_path
        
        if (pred_bbox is not None) and (not isinstance(pred_bbox, list)):
            self.pred_bbox = pred_bbox.tolist()
        elif pred_bbox is not None:
            self.pred_bbox = pred_bbox
        else:
            self.pred_bbox = None
            
        self.pred_score_bbox = pred_score_bbox

        # Obtained when prediction matches with a gt bounding box
        self.gt_bbox = None
        self.gt_label = None

        # Obtained from knn
        self.pred_label = None
        # Obtained from knn by measuring mean distance to its predicted label
        self.class_score = None
        self.pred_features = None
        self.is_train = None
        
    def add_gt_bbox(self, gt_bbox, gt_label, train_class_flag):
        self.gt_bbox = gt_bbox
        self.gt_label = gt_label
        self.is_train = train_class_flag
    
    def add_feature_vector(self, feature_vector):
        self.pred_features = feature_vector.tolist()
    
    def add_classification_res(self, pred_label, mean_dist):
        self.pred_label = pred_label
        self.class_score = mean_dist
        
    def to_dict(self):
        return {
            'img_name': self.img_name,
            'img_path': self.img_path,
            'pred_bbox': self.pred_bbox,
            'pred_score_bbox': self.pred_score_bbox,
            'gt_bbox': self.gt_bbox,
            'gt_label': self.gt_label,
            'pred_label': self.pred_label,
            'class_score': self.class_score,
            'pred_features': self.pred_features,
            'is_train': self.is_train
        }
    
    def read_dict(self, content):
        self.img_name = content['img_name']
        self.img_path = content['img_path']
        self.pred_bbox = content['pred_bbox']
        
        self.pred_score_bbox = content['pred_score_bbox']
        self.gt_bbox = content['gt_bbox']
        self.gt_label = content['gt_label']
        self.pred_label = content['pred_label']
        self.class_score = content['class_score']
        self.pred_features = content['pred_features']
        self.is_train = content['is_train']


def dump_pred_objects(prediction_objects, jpath):
    json_content = [pred_object.to_dict() for pred_object in prediction_objects]
    with open(jpath, "w") as jfile:
        json.dump(json_content, jfile)


def get_pred_objects_per_image(pred_objects):
    img_names = list(set([pobject.img_name for pobject in pred_objects]))
    objects_per_img = {}
    for img_name in img_names:
        img_objects = [pobject for pobject in pred_objects if pobject.img_name == img_name]
        objects_per_img[img_name] = img_objects
    return objects_per_img                


def read_pred_objects_json(json_path):
    pred_objects = []
    with open(json_path, "r") as jfile:
        json_objects = json.load(jfile)    
    
    for json_object in json_objects:
        cur_object = Prediction("", "", [], 0.0)
        cur_object.read_dict(json_object)
        pred_objects.append(cur_object)
    return pred_objects