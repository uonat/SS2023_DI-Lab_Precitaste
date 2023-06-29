import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA

class DINOFeatureExtractor:
    def __init__(self, dino_version='dinov2_vits14', resize_dim=512, input_dim=448):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        allowed_versions = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14']
        if dino_version not in allowed_versions:
            raise ValueError("Given version of the dino not found! Please provide one of: {}".format(dino_version))
            
        self.model = torch.hub.load('facebookresearch/dinov2', dino_version)
        
        self.model.eval()
        self.model.to(self.device)
        
        self.transforms = T.Compose([
            T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(input_dim),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    
    def predict(self, np_arr):
        inp = self.transforms(Image.fromarray(np_arr.astype('uint8')))
        # Inference object
        model_inp = inp.unsqueeze(0).to(self.device)
        feats = self.model(model_inp) 
        return feats
    
    
    def patch_feature_extraction(self, np_arr):
        inp = self.transforms(Image.fromarray(np_arr.astype('uint8')))
        model_inp = inp.unsqueeze(0).to(self.device)
        feats = self.model.forward_features(model_inp)
        patch_feats = feats['x_norm_patchtokens']
        return patch_feats

    def print_patch_features(self, pil_img, patch_num=52, background_thres=20):

        img = pil_img.convert('RGB')
        # add below to class code 
        
        transform = T.Compose([
            #T.GaussianBlur(9, sigma=(0.1, 2.0)),
            T.Resize((patch_num * self.patch_dim, patch_num * self.patch_dim)),
            T.CenterCrop((patch_num * self.patch_dim, patch_num * self.patch_dim)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        img_tensor = transform(img)[:3]

        img_tensor = torch.unsqueeze(img_tensor, 0)

        with torch.no_grad():

            features_dict = self.model.forward_features(img_tensor)
            features = features_dict['x_norm_patchtokens']

        np_features = features.cpu().numpy()[0]
        pca = PCA(n_components=3)
        pca.fit(np_features)
        pca_features = pca.transform(np_features)

        # Apply a mask to remove background
        pca_features_bg = pca_features[:, 0] < background_thres
        pca_features_fg = ~pca_features_bg    

        pca.fit(np_features[pca_features_fg]) # NOTE: I forgot to add it in my original answer
        pca_features_rem = pca.transform(np_features[pca_features_fg])
        for i in range(3):
            # pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())
            # transform using mean and std, I personally found this transformation gives a better visualization
            pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].mean()) / (pca_features_rem[:, i].std() ** 2) + 0.5

        pca_features_rgb = pca_features.copy()
        pca_features_rgb[pca_features_bg] = 0
        pca_features_rgb[pca_features_fg] = pca_features_rem

        pca_features_rgb = pca_features_rgb.reshape(1, patch_num, patch_num, 3)
        return pca_features_rgb