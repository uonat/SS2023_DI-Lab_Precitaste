import sys
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import f1_score,precision_score,recall_score,average_precision_score,accuracy_score
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt

current_directory = os.getcwd()
sys.path.insert(0, current_directory)
from notebooks.utils.dino_v2 import read_pred_objects_json, get_pred_objects_per_image, crop_object_with_bbox

def visualize_dino_v2_features(input_dir, output_dir, query_number, image_number=-1, threshold=0.5, save_pred=True, save_gt=False):
    output_dir = Path(output_dir)
    input_dir = Path(input_dir)
    path_to_pred_json_file = Path("dataset") / "dinov2_processed_data" / "vith_res_val_processed_pred_objects_2.json"

    # max_cosine_similarity = -1

    if save_gt:
        gt_output_dir = output_dir / "ground_truth"         # f"ground_truth_{threshold}"
        gt_output_dir.mkdir(parents=True, exist_ok=True)
        # print(f"Saving ground truth to {str(gt_output_dir)}")

    # if save_pred:
    #     for threshold_value in threshold:
    #         pred_output_dir = output_dir / f"predictions_{threshold_value}"       # output_dir / "predictions" + threshold 
    #         pred_output_dir.mkdir(parents=True, exist_ok=True)
    #         print(f"Saving predictions to {pred_output_dir}")

    # [type: pred_objects]
    val_pred_objects = read_pred_objects_json(str(path_to_pred_json_file))

    query_list = val_pred_objects[:query_number]
    query_features = np.array([query.pred_features[0] for query in query_list])
    query_labels = [query.gt_label for query in query_list]

    query_input_images = [crop_object_with_bbox(cv2.imread(str(input_dir / query.img_name)), query.pred_bbox) for query in query_list]

    per_image_val_pred_objects = get_pred_objects_per_image(val_pred_objects)

    img_names = list(per_image_val_pred_objects.keys())
    if image_number > 0:
        img_names = img_names[:image_number]

    img_paths = [str(input_dir / img_name) for img_name in img_names]

    for threshold_value in tqdm(threshold):
        print(threshold_value)
        for i, (img_name, img_path) in tqdm(enumerate(zip(img_names, img_paths)), total=min(len(img_names), len(img_paths))):
            # 
            input_img_pred_objects = per_image_val_pred_objects[img_name]
            img_feature = np.array([bounding_box.pred_features[0] for bounding_box in input_img_pred_objects])

            if query_features.ndim == 1:
                query_features = query_features[None, ...]

            cosine_similarity_scores = cosine_similarity(query_features, img_feature)

            # if cosine_similarity_scores.max() > max_cosine_similarity:
            #     max_cosine_similarity = cosine_similarity_scores.max()

            bboxes_dict_pred = {}
            for row_ind, row_of_cosine_similarity_scores in enumerate(cosine_similarity_scores):
                column_indices_over_threshold = np.where(row_of_cosine_similarity_scores > threshold_value)[0]
                y_score = row_of_cosine_similarity_scores 

                if column_indices_over_threshold.size == 0:
                    bboxes_dict_pred[row_ind] = []
                else:
                    bboxes_dict_pred[row_ind] = np.array(
                        [input_img_pred_objects[j].pred_bbox + [j] + [query_labels[row_ind]] for j in column_indices_over_threshold]
                        )

            if save_pred:
                fig, axs = plt.subplots(max(len(query_input_images), len(bboxes_dict_pred)), 2, figsize=(20, 30))

                # Showing query images
                for j, img in enumerate(query_input_images):
                    axs[j, 0].imshow(img)
                    axs[j, 0].axis('off')

                # Showing bounding boxes with cosine similarity scores
                for j, boxes in enumerate(bboxes_dict_pred.values()):
                    img = cv2.imread(img_path)
                    for box in boxes:
                        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                        img = cv2.putText(img, f'{cosine_similarity_scores[j, int(box[4])]:.2f}', (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
                        img = cv2.putText(img, box[5], (int(box[0]), int(box[1])-60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
                    axs[j, 1].imshow(img)
                    axs[j, 1].axis('off')

                plt.tight_layout()
                pred_output_dir = output_dir / f"predictions_{threshold_value}"       # output_dir / "predictions" + threshold 
                pred_output_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(pred_output_dir / f'pred_{i}.png')
                print(f"Saving predictions to {pred_output_dir}")
                plt.close()

            if save_gt:

                ###  only show obj that share same label with queries
                if os.path.exists(str(gt_output_dir / f"gt_{i}.jpg")):
                    continue
                img_gt_objects = [obj for obj in input_img_pred_objects if obj.gt_label in query_labels]
                img_gt_bboxes = [np.array(pred_object.gt_bbox).astype(np.int32) for pred_object in img_gt_objects if pred_object.gt_bbox is not None]
                img_gt_labels = [pred_object.gt_label for pred_object in img_gt_objects if pred_object.gt_label is not None]
                img_gt_bboxes = np.array(img_gt_bboxes)

                if len(img_gt_bboxes) > 0:
                    print(f"In image number {i} there are {len(img_gt_bboxes)} ground truth bounding boxes")

                img_cv2_gt = cv2.imread(img_path)

                for _i, (x, y, w, h) in enumerate(img_gt_bboxes):
                    img_cv2_gt = cv2.rectangle(img_cv2_gt, [x, y, w, h], (0, 0, 255), 3)
                    img_cv2_gt = cv2.putText(img_cv2_gt, img_gt_labels[_i], [x, y - 10], cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                
                cv2.imwrite(gt_output_file_path:=str(gt_output_dir / f"gt_{i}.jpg"), img_cv2_gt)
                print(f"Saved a ground truth image at {gt_output_file_path}")
            
    # print(f"Max Cosine Smilarity Score was: {max_cosine_similarity}")

    
def main():
    config_for_visualization = {
        "input_dir" : "RPCdataset/val2019",
        "output_dir" : "tmp/visualization",
        "query_number" : 5,
        "image_number" : 200,
        "threshold" : [0.2, 0.4, 0.6],       #   [0.2, 0.3, 0.4, 0.5, 0.6],
        "save_pred" : True,
        "save_gt" : True
    }

    print(f"Visualizing bounding boxes with following config: {config_for_visualization}")

    visualize_dino_v2_features(**config_for_visualization)

if __name__=='__main__':
    main()
