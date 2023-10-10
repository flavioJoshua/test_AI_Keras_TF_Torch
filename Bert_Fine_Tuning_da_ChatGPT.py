import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Scegliere il dispositivo (CPU o GPU)
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# FIXME:   troppo  diverso dall originale non dovrebbe  funzionare almeno per il mancato passaggio delle etichette  al pre trained 
# Caricare il dataset
dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")

# Preprocessing e tokenizzazione
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_data(examples):
    text = examples["Tweet"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    labels_matrix = np.zeros((len(text), len(labels)))

    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()
    return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
encoded_dataset.set_format("torch")

# Definire il modello e spostarlo sul dispositivo
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                           problem_type="multi_label_classification",
                                                           num_labels=len(labels)).to(device)

# Definire le metriche
def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result

# Definire gli argomenti per la formazione
args = TrainingArguments(
    "bert-finetuned-sem_eval",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

# Inizializzare il Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,
)

# Avviare la formazione
trainer.train()

# Eseguire l'inferenza
text = "I'm happy I can finally train a model for multi-label classification"
encoding = tokenizer(text, return_tensors="pt").to(device)
outputs = model(**encoding)
