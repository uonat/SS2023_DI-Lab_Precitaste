from models.VitDet import load_model,Draw_pred_BB,Draw_original_seg
from dataset.SKU110K_fixed_dataset import get_image,download_dataset,print_img_with_GT_BB
import torch, cv2



## Data ##
path_to_ds,annotations_path = download_dataset() #TODO or if downloaded, use directly to the directory path

which_set = "train" #train #test #val
random_img_num = 4
img_path="{}/images/{}_{}.jpg".format(path_to_ds,which_set,random_img_num) #Check if exists

#TODO get annotations here:  (& update print_img_with_GT_BB)
#all_annotations = {"train":_get_annotations(train_ann_path),"test":_get_annotations(test_ann_path),"val":_get_annotations(val_ann_path)} 
#print_img_with_GT_BB(img_path,annotations_path,which_set=which_set)  



## Model ##
model_path ="/content/drive/MyDrive/ApplicationProject/Models/model2.pkl"
model = load_model(model_path) 
model.to('cpu')
model.eval()



## Output ##
img = get_image(img_path)
batch = [{'image':torch.from_numpy(img).movedim(-1,0)}]


with torch.no_grad():
  model_result=model(batch)



Draw_pred_BB(img,model_result)
#Draw_original_seg(img,model_result)


