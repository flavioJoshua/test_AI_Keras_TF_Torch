
# Rilevamento del dispositivo
import torch
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device = torch.device("cuda:0")  # Sceglie la prima GPU
else:
    device = torch.device("cpu")  # Sceglie la CPU se CUDA non è disponibile o non ci sono GPU

# INFORM: funziona con la  GPU   meglio usare  export xport CUDA_VISIBLE_DEVICES=0  altrimenti con  due    GPU  va in split di carico 


from datasets import load_dataset

dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")


print(dataset )



example = dataset['train'][0]
print(example)

labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
print(labels)

from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_data(examples):
  # take a batch of texts
  text = examples["Tweet"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()

  return encoding


encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)


example = encoded_dataset['train'][0]
print(example.keys())


tokenizer.decode(example['input_ids'])


print(example['labels'])

_testo=[id2label[idx] for idx, label in enumerate(example['labels']) if label == 1.0]
print(_testo)



encoded_dataset.set_format("torch")




from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                           problem_type="multi_label_classification",
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id).to(device)




batch_size = 8
metric_name = "f1"

from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    f"bert-finetuned-sem_eval-english",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)


from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

print(encoded_dataset['train'][0]['labels'].type())

print(encoded_dataset['train']['input_ids'][0])

#forward pass  senza   GPU ...   sostituito
#outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0)).to(device)

# Assicurati che sia il modello che gli input siano sul dispositivo corretto
model = model.to(device)

# ... (parte di codice per il preprocessing e l'encoding)

# Muovi i tensori sul dispositivo corretto prima del forward pass
input_ids = encoded_dataset['train']['input_ids'][0].unsqueeze(0).to(device)
labels = encoded_dataset['train'][0]['labels'].unsqueeze(0).to(device)

# Esegui il forward pass
outputs = model(input_ids=input_ids, labels=labels)

# ... (resto del codice)



print(outputs)


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()

trainer.evaluate()

text = "I'm happy I can finally train a model for multi-label classification"

encoding = tokenizer(text, return_tensors="pt").to(device)
encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

outputs = trainer.model(**encoding).to(device)

logits = outputs.logits
print   (f"shape  del  logits: {logits.shape  }" )

# apply sigmoid + threshold
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1
# turn predicted id's into actual label names
predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
print(predicted_labels)

