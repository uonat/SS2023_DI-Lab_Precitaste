# Evaluating with Object-Detection-Metrics

This directory includes code for evaluating an object detection results. The code is extended from [rafaelpadilla/Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics/tree/master) with image level statistics.

## How to use?

1. Create two different folder for ground truth and prediction bounding boxes as text file per image. Each text file must named as image_name.txt where image_name is the name of the predicted image without the extension. 

Even there is no objects for gt or prediction, the text file must exists. Create an empty text file for such cases.  

The text files must include a single bounding box per line. The exact format is:

- For GT: `<class_name> <x1> <y1> <w> <h>`
- For Prediction: `<class_name> <confidence> <x1> <y1> <w> <h>`

Each part must be splitted with a space `" "` character and coordinate values must be given absolute. In other words they should not be divided by image width and height. 

x1 and y1 represents the top-left values of the bounding boxes. It is also possible to provide bounding box part as:

`<x1> <y1> <x2> <y2>`

For this option please check command line arguments. 


2. Run the [evaluate.py](evaluate.py) with the given arguments:

```
python evaluate.py -gt <path of the folder that includes gt texts> -det <path of the folder that includes pred texts> -sp <path to save outputs of the script> -gtformat <format of the gt bounding boxes. Can be xywh or xyrb. Default is xywh> -detformat <format of the prediction bounding boxes. Can be xywh or xyrb. Default is xywh>
```

The script will print the AP value per class for the whole dataset and plot precision x recall curve for each class.

It also generates a csv file that contains below metrics per each class in each image:

* img_name: Name of the image
* class: Name of the class
* AP: Average precision
* TP: Number of true positives for prediction
* TP: Number of false positives
* Recall: Recall value for that class for that image
* Precision: Precision value for that class for that image
* GT Objects: Number of objects in the GT annotations for that class for that image
