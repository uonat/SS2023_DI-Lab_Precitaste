import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import *
import glob
import os
import pandas as pd
import argparse

# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append('argument %s: invalid value. It must be either \'xywh\' or \'xyrb\'' %
                      argName)
        
def getBoundingBoxes(gt_annots_path, pred_annots_path, gt_bbox_format, pred_bbox_format):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    # Read ground truths
    os.chdir(gt_annots_path)
    files = glob.glob("*.txt")
    files.sort()
    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    annotations_per_file = {}
    for f in files:
        nameOfImage = f.replace(".txt", "")
        annotations_per_file[nameOfImage] = {}
        file_bboxes = BoundingBoxes()
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            x = float(splitLine[1])  # confidence
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (200, 200),
                BBType.GroundTruth,
                format=gt_bbox_format)
            allBoundingBoxes.addBoundingBox(bb)
            file_bboxes.addBoundingBox(bb)
        fh1.close()
        annotations_per_file[nameOfImage] = file_bboxes
    
    # Read detections
    os.chdir(pred_annots_path)
    files = glob.glob("*.txt")
    files.sort()
    # Read detections from txt file
    # Each line of the files in the detections folder represents a detected bounding box.
    # Each value of each line is  "class_id, confidence, x, y, width, height" respectively
    # Class_id represents the class of the detected bounding box
    # Confidence represents confidence (from 0 to 1) that this detection belongs to the class_id.
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        # nameOfImage = f.replace("_det.txt","")
        nameOfImage = f.replace(".txt", "")
        if nameOfImage not in annotations_per_file:
            annotations_per_file[nameOfImage] = {}
        file_bboxes = BoundingBoxes()
        
        # Read detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])  # confidence
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (200, 200),
                BBType.Detected,
                confidence,
                format=pred_bbox_format)
            allBoundingBoxes.addBoundingBox(bb)
            annotations_per_file[nameOfImage].addBoundingBox(bb)
        fh1.close()
    return allBoundingBoxes, annotations_per_file

parser = argparse.ArgumentParser(description="Evaluates object detection results using gt and prediction annotation text files.")
parser.add_argument('-gt', '--gtfolder', dest='gtFolder', metavar='', help='folder containing your ground truth bounding boxes')
parser.add_argument('-det', '--detfolder', dest='detFolder', metavar='',  help='folder containing your detected bounding boxes')
parser.add_argument('-sp', '--savepath', dest='savePath', metavar='', help='folder where the outputs are saved')

parser.add_argument('-gtformat',
                    dest='gtFormat',
                    metavar='',
                    default='xywh',
                    help='format of the coordinates of the ground truth bounding boxes: '
                    '(\'xywh\': <left> <top> <width> <height>)'
                    ' or (\'xyrb\': <left> <top> <right> <bottom>)')

parser.add_argument('-detformat',
                    dest='detFormat',
                    metavar='',
                    default='xywh',
                    help='format of the coordinates of the detected bounding boxes '
                    '(\'xywh\': <left> <top> <width> <height>) '
                    'or (\'xyrb\': <left> <top> <right> <bottom>)')


if __name__ == "__main__":

    args = parser.parse_args()
    # Arguments validation
    errors = []
    # Validate formats
    gtFormat = ValidateFormats(args.gtFormat, '-gtformat', errors)
    detFormat = ValidateFormats(args.detFormat, '-detformat', errors)

    if len(errors) > 0:
        raise ValueError("Wrong value passed for bbox formats: {}".format(errors[0]))
    
    gt_annotations_path = args.gtFolder
    pred_annotations_path = args.detFolder

    if not os.path.exists(gt_annotations_path) or not os.path.isdir(gt_annotations_path):
        raise ValueError("GT annotations path doesn't exists or not a folder")
    
    if not os.path.exists(pred_annotations_path) or not os.path.isdir(pred_annotations_path):
        raise ValueError("Prediction annotations path doesn't exists or not a folder")    

    save_path = os.path.join('evaluation_results')
    
    if args.savePath is None:
        print("No value passed for savepath argument. All outputs will be saved under: {}".format(save_path))
    else:
        save_path = args.savePath

    os.makedirs(save_path, exist_ok=True)

    all_bboxes, bboxes_per_img = getBoundingBoxes(gt_annotations_path, pred_annotations_path, gtFormat, detFormat)

    # Uncomment the line below to generate images based on the bounding boxes
    # createImages(dictGroundTruth, dictDetected)
    # Create an evaluator object in order to obtain the metrics
    evaluator = Evaluator()

    ##############################################################
    # VOC PASCAL Metrics
    ##############################################################
    # Plot Precision x Recall curve
    evaluator.PlotPrecisionRecallCurve(
        all_bboxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=0.3,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,  # As the official matlab code
        showAP=True,  # Show Average Precision in the title of the plot
        savePath=save_path,
        showInterpolatedPrecision=True,
        showGraphic=False)  # Plot the interpolated precision curve

    # Get metrics with PASCAL VOC metrics
    metricsPerClass = evaluator.GetPascalVOCMetrics(
        all_bboxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=0.3,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
    
    print("Average precision values per class for the whole images:\n")
    # Loop through classes to obtain their metrics
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        # Print AP per class
        print('Class: %s: AP: %f' % (c, average_precision))

    results = []

    print("Evaluating each image...")
    for img_name in bboxes_per_img:

        img_bboxes = bboxes_per_img[img_name]

        per_img_evaluator = Evaluator()
        # Get metrics with PASCAL VOC metrics
        metricsPerClass = per_img_evaluator.GetPascalVOCMetrics(
            img_bboxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=0.3,  # IOU threshold
            method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
                
        for mc in metricsPerClass:
            if len(mc['precision']) == 0:
                precision = 0.0
            # Precision will be equal to the final element of the cumulative sum
            else:
                precision = mc['precision'][-1]

            if len(mc['recall']) == 0:
                recall = 0.0
            # Recall will be equal to the final element of the cumulative sum
            else:
                recall = mc['recall'][-1]
            
            results.append({
                'img_name': img_name,
                'class': mc['class'],
                'AP': mc['AP'],
                'TP': mc['total TP'],
                'FP': mc['total FP'],
                'Recall': recall,
                'Precision': precision,
                'GT Objects': mc['total positives']
                })

    results_df = pd.DataFrame(results)
    results_save_path = os.path.join(save_path, 'eval_results_per_image.csv')
    results_df.to_csv(results_save_path, index=False)
