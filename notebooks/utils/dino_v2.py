def crop_object_with_bbox(np_img, bbox):
    x, y, x2, y2 = [int(i) for i in bbox]
    return np_img[y:y2, x:x2].astype('float32')

def calculate_iou(bbox1, bbox2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    Bounding boxes are represented as (x_min, y_min, x_max, y_max).
    Returns:
        IoU value between 0 and 1.
    """
    # Calculate the intersection area
    x_min_inter = max(bbox1[0], bbox2[0])
    y_min_inter = max(bbox1[1], bbox2[1])
    x_max_inter = min(bbox1[2], bbox2[2])
    y_max_inter = min(bbox1[3], bbox2[3])
    
    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(0, y_max_inter - y_min_inter + 1)
    
    # Calculate the union area
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Calculate the IoU
    iou = intersection_area / union_area
    
    return iou

def find_gt_bboxes_of_pred(pred_bboxes, pred_scores, gt_bboxes, iou_thres=0.5):
    """
    For a given prediction boundig boxes and their confidence, returns the index of the 
    gt bounding boxes that matches with the predictions.
    First sorts all prediction bounding boxes and measures the iou with all gt boxes. 
    Two boxes will be considered as matched when their iou is above iou_threshold. 
    Then marks that matched gt as used and for all remaining predictions does the same 
    with remaining unused gt boxes. 
    Returns:
        Two lists, first one is for matched gt bbox index and -1 for unmatched gt bboxes
        secon list is for unfound gt bbox indices.
    """
    # Sort prediction bounding boxes by their scores in descending order
    sorted_indices = sorted(range(len(pred_scores)), key=lambda k: pred_scores[k], reverse=True)
    
    # Initialize an array to store the matched gt bbox indices (-1 for unmatched gt bboxes)
    matched_gt_indices = [-1] * len(pred_bboxes)
    
    # Iterate over the sorted prediction bounding boxes
    for pred_index in sorted_indices:
        pred_bbox = pred_bboxes[pred_index]
        
        # Initialize variables to track the maximum IoU and the corresponding gt bbox index
        max_iou = 0
        max_iou_index = -1
        
        # Iterate over the gt bounding boxes
        for gt_index, gt_bbox in enumerate(gt_bboxes):
            # Calculate the IoU between the prediction bbox and the gt bbox
            iou = calculate_iou(pred_bbox, gt_bbox)
            
            # If the IoU is above the threshold and higher than the previous maximum IoU,
            # update the maximum IoU and the corresponding gt bbox index
            # Also the gt bounding box didn't matched with another prediction before
            if iou > iou_thres and iou > max_iou and gt_index not in matched_gt_indices:
                max_iou = iou
                max_iou_index = gt_index
        
        # If a matching gt bbox is found, mark it as used and store the index in the results array
        if max_iou_index != -1:
            matched_gt_indices[pred_index] = max_iou_index
    
    unmatched_gt_indices = [i for i in range(len(gt_bboxes)) if i not in matched_gt_indices]
    # Return the matched gt bbox indices array
    return matched_gt_indices, unmatched_gt_indices