import clip
import pickle
import cv2
from tqdm import tqdm
import torch
from PIL import Image



def available_clip_models():
    return clip.available_models()

def load_model(model_name,device):
    return clip.load(model_name, device=device)

def tokenize_text(txt,device):
    return clip.tokenize(txt).to(device)

def get_total_num_obj(dataset):
    total_num_obj = 0
    for i in range(dataset.get_num_imgs()):  
        annots  = dataset.get_annots_by_img_id(i, key_for_category='sku_name')
        total_num_obj += len(annots)
    return total_num_obj

def load_results(output_dir,total_num_obj):
    with open(output_dir, "rb") as fp:
        Results = [pickle.load(fp) for i in range(total_num_obj)]
    return Results

def Calculate_Scores(clip_model,preprocess,dataset,target_features,all_labels,output_dir,device,fine_grained):
    Results = []
    total_num_obj = 0

    for i in tqdm(range(dataset.get_num_imgs())):
        img_path = dataset.get_img_path_by_id(i)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        annots = dataset.get_annots_by_img_id(i, key_for_category='sku_name')
        total_num_obj += len(annots)

        for annot in annots:
            x1,y1,w,h = list(map(int, annot[0]))
            x2,y2 = x1+w,y1+h    
            img_p = preprocess(Image.fromarray(img[y1:y2,x1:x2].copy())).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(img_p)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            lbl = all_labels.index(annot[1]) if fine_grained else all_labels.index(' '.join(annot[1].split('_')[1:]))
            Results.append(((image_features @ target_features.T).tolist()[0],lbl))

        if (i+1) % 500 == 0: # save in every 500 images
            with open(output_dir, "ab") as fp:
                for item in Results: pickle.dump(item, fp)
            Results = []

    with open(output_dir, "ab") as fp:    # save the remaining
        for item in Results: pickle.dump(item, fp)

    return load_results(output_dir,total_num_obj)

def Calculate_Embeddings(clip_model,preprocess,dataset,output_dir,device): ####TODO: CHECK tolist()[0]!!!!!!!!!
    Results = []

    for i in tqdm(range(dataset.get_num_imgs())):  #Every annots is a list of 1
        img_path = dataset.get_img_path_by_id(i)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        annot = dataset.get_annots_by_img_id(i, key_for_category='sku_name')[0] 

        x1,y1,w,h = list(map(int, annot[0]))
        x2,y2 = x1+w,y1+h

        img_p = preprocess(Image.fromarray(img[y1:y2,x1:x2].copy())).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(img_p)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        Results.append((image_features.tolist()[0],annot[1]))  
        
        if (i+1) % 500 == 0: # save in every 500 images
            with open(output_dir, "ab") as fp:
                for item in Results: pickle.dump(item, fp)
            Results = []

    with open(output_dir, "ab") as fp:    # save the remaining
        for item in Results: pickle.dump(item, fp)

    return load_results(output_dir,dataset.get_num_imgs())

def Calculate_Embeddings_general(clip_model,preprocess,dataset,output_dir,device): ####TODO: CHECK tolist()[0]!!!!!!!!!
    Results = []
    total_num_obj = 0

    for i in tqdm(range(dataset.get_num_imgs())): 
        img_path = dataset.get_img_path_by_id(i)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        annots = dataset.get_annots_by_img_id(i, key_for_category='sku_name')
        total_num_obj += len(annots)
        
        for annot in annots:
            x1,y1,w,h = list(map(int, annot[0]))
            x2,y2 = x1+w,y1+h
            img_p = preprocess(Image.fromarray(img[y1:y2,x1:x2].copy())).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(img_p)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            Results.append((image_features.tolist()[0],annot[1]))  
        
        if (i+1) % 500 == 0: # save in every 500 images
            with open(output_dir, "ab") as fp:
                for item in Results: pickle.dump(item, fp)
            Results = []

    with open(output_dir, "ab") as fp:    # save the remaining
        for item in Results: pickle.dump(item, fp)

    return load_results(output_dir,total_num_obj)