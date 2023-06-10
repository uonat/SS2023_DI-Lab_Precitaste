def is_bigger_eps(x1, x2, eps):
    """
    Returns true if <second argument> is bigger than <first argument - eps>
    Used to check if second point is sligthly smaller or bigger than first point
    """
    return (x2 > (x1-eps))


def is_smaller_eps(x1, x2, eps):
    """
    Returns true if <second argument> is smaller than <first argument + eps>
    Used to check if second point is sligthly bigger or smaller than first point
    """
    return (x2 < (x1+eps))


def is_box_inside_other(bbox1, bbox2, eps):
    """
    Checks if a bounding box is completely inside other bounding box
    Returns 1 if first box is inside the second
    Returns 2 if second box is inside the first
    Returns 0 if no box is inside other 
    Also allows if the bounding box is slighlty off but mostly inside other
    Control this with eps parameter
    """
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2
    
    bbox1_area = (x12-x11) * (y12-y11)
    bbox2_area = (x22-x21) * (y22-y21)
    
    retVal = 0
    
    if bbox1_area > bbox2_area:
        bigger_bbox = bbox1
        smaller_bbox = bbox2
        # Second argument is inside the first
        retVal = 2
    else:
        bigger_bbox = bbox2
        smaller_bbox = bbox1
        # First argument is inside the second
        retVal = 1
    
    xb1, yb1, xb2, yb2 = bigger_bbox
    xs1, ys1, xs2, ys2 = smaller_bbox
    
    if is_bigger_eps(xb1, xs1, eps) and is_bigger_eps(yb1, ys1, eps) and is_smaller_eps(xb2, xs2, eps) and is_smaller_eps(yb2, ys2, eps):
        return retVal
    
    return 0


def eliminate_boxes(bboxes, img_h, img_w, area_thres=0.25, eps=10, return_bbox_indices=False):
    """
    Eliminates bounding boxes if its area is bigger than the @area_thres of the image area
    or is small and inside another bbox
    @area_thres is the threshold for eliminating big bounding boxes, bbox whose relative 
    area bigger than this will be elminated
    @eps is the small tolerance to consider both points same
    If @return_bbox_indices is true then returns the index of bounding boxes from the @bboxes parameter
    that this function kept
    """    
    img_area = img_h * img_w
    
    final_boxes = []
    # Remove too big bounding boxes 
    for bbox in bboxes:
        bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])    
        if (bbox_area / img_area) > area_thres:
            continue

        final_boxes.append(bbox)

    # Remove small bounding boxes inside other bounding box
    remove_indices = []
    for i in range(len(final_boxes)):
        for j in range(i, len(final_boxes)):

            if i == j:
                continue

            bbox1 = final_boxes[i]
            bbox2 = final_boxes[j]

            ret_val = is_box_inside_other(bbox1, bbox2, eps)
            if ret_val == 1:
                remove_indices.append(i)
            if ret_val == 2:
                remove_indices.append(j)    
    
    if return_bbox_indices:
        return [final_boxes[i] for i in range(len(final_boxes)) if i not in remove_indices], [i for i in range(len(final_boxes)) if i not in remove_indices]
    if return_bbox_indices:
        return [final_boxes[i] for i in range(len(final_boxes)) if i not in remove_indices]
