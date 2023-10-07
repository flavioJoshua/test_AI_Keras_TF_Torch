from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device = torch.device("cuda:0")  # Sceglie la prima GPU
else:
    device = torch.device("cpu")  # Sceglie la CPU se CUDA non è disponibile o non ci sono GPU


#Load CLIP model
model = SentenceTransformer('clip-ViT-B-32').to(device=device)

#Encode an image:
img_emb = model.encode(Image.open('img.png'))

labels=['green grass', 'A cat on a table', 'A picture of London at night']
#Encode text descriptions
text_emb = model.encode(labels)

#Compute cosine similarities 
cos_scores = util.cos_sim(img_emb, text_emb).to(device=device)
_max=torch.max(cos_scores)
print( f" valori {cos_scores}  il valore  massimo {_max}   e  la stringa :  { labels[torch.argmax(cos_scores)] } ")
