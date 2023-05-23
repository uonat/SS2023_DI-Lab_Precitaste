import clip




def available_clip_models():
    return clip.available_models()

def load_model(model_name,device):
    return clip.load(model_name, device=device)

def tokenize_text(txt,device):
    return clip.tokenize(txt).to(device)
