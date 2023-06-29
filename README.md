# Open-vocabulary Object Detections of Inventory Items

This repository is the code base for the project of ‘Open-vocabulary Object Detections of Inventory Items’ at TUM Data Innovation Lab of Munich Data Science Institute (MDSI) at the Technical University of Munich sponsored by PreciBake GmbH.


## Problem Setting
Computer vision based inventory tracking allows our restaurants and retail clients to measure changes in their food inventory in real-time. This can be used to dynamically schedule and optimize stock-ups of new inventory to ensure that we always have an optimal amount of fresh inventory available, without creating excessive inventory that could result in food waste.

Tracking inventory can be challenging since a typical inventory might contain hundreds or even thousands of unique items. As the inventory size grows, object detection becomes increasingly challenging and maintaining
high quality datasets becomes increasingly difficult. What makes it even more challenging is that new items could be introduced or their packaging could be changed which would occasionally require additional relabeling and retraining.

At the same time humans are able to easily recognize hundreds of thousands of different object categories without needing hundreds of training examples per category. In order to recognize new objects we only need a description of what the object looks like.

In this project we will study the challenging task of detecting an open set of objects merely based on the object description. Such a model could be used to detect a large number of different object categories including objects categories that it was not specifically trained for.

## Usage

## Pipeline

## Results

## Future Work








## Contributors
- Burak Bekci
- Umut Onat
- Florian Schraitle
- Yushan Zheng




## Acknowledgements

First, we would like to express our utmost gratitude for Mathias, Max and Sebastian at PreciBake GmbH for their significant expertise and support that enable this project. 

As this project is rooted in the integration and utilization of [ViTDet](https://github.com/ViTAE-Transformer/ViTDet), [CLIP](https://github.com/openai/CLIP), [DINOv2](https://github.com/facebookresearch/dinov2) with a substantial reliance on [Detectron2](https://github.com/facebookresearch/detectron2/tree/main), we also acknowledge their substantial impact. Their research forms the cornerstone and provides the necessary foundation for our project. Thanks for their great works!

Additionally, we extend our sincere appreciation to [GroundingDINO], [CORA], [RegionCLIP] and many other works in the area of open-vocabulary object detecion, from which we draw inspiration and insights.




## References