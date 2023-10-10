from transformers import AutoModel

# Caricare un modello pre-addestrato
model = AutoModel.from_pretrained('/media/flavio/bck-img-6T/Backup/AI/huggingsface.cache.hub/cross-encoder/ms-marco-MiniLM-L-12-v2')
# model.save_pretrained('/media/flavio/bck-img-6T/Backup/AI/huggingsface.cache.hub/cross-encoder/ms-marco-MiniLM-L-12-v2')



# Use a pipeline for text classification
# Here, the model "cross-encoder/ms-marco-MiniLM-L-12-v2" is used for demonstration
from  transformers  import    pipeline
try:
    text_classification_pipeline = pipeline("text-classification", model="/media/flavio/bck-img-6T/Backup/AI/huggingsface.cache.hub/cross-encoder_text-classification/ms-marco-MiniLM-L-12-v2")
    # result = text_classification_pipeline("This is a sample text for classification.")
    # print("Pipeline result:", result)



    text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    result=text_classification_pipeline(text)
    print("Pipeltext_classine result:", result)

except Exception as e:
    print(f"Could not create pipeline. Error: {e}")



from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")

features = tokenizer(['How many people live in Berlin?', 'How many people live in Berlin?'], ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.'],  padding=True, truncation=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    scores = model(**features).logits
    print(scores)



# # Load model directly
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
# model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")