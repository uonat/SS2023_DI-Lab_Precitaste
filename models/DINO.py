import torch
import torchvision.transforms as T
from PIL import Image

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