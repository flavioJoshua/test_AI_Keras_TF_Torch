# %% [markdown]
# <a href="https://colab.research.google.com/github/flavioJoshua/test_AI_Keras_TF_Torch/blob/main/Copia_di_Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Fine-tuning BERT (and friends) for multi-label text classification
# 
# In this notebook, we are going to fine-tune BERT to predict one or more labels for a given piece of text. Note that this notebook illustrates how to fine-tune a bert-base-uncased model, but you can also fine-tune a RoBERTa, DeBERTa, DistilBERT, CANINE, ... checkpoint in the same way.
# 
# All of those work in the same way: they add a linear layer on top of the base model, which is used to produce a tensor of shape (batch_size, num_labels), indicating the unnormalized scores for a number of labels for every example in the batch.
# 
# 
# 
# ## Set-up environment
# 
# First, we install the libraries which we'll use: HuggingFace Transformers and Datasets.

# %%

# %% [markdown]
# ## Load dataset
# 
# Next, let's download a multi-label text classification dataset from the [hub](https://huggingface.co/).
# 
# At the time of writing, I picked a random one as follows:   
# 
# * first, go to the "datasets" tab on huggingface.co
# * next, select the "multi-label-classification" tag on the left as well as the the "1k<10k" tag (fo find a relatively small dataset).
# 
# Note that you can also easily load your local data (i.e. csv files, txt files, Parquet files, JSON, ...) as explained [here](https://huggingface.co/docs/datasets/loading.html#local-and-remote-files).
# 
# 
# Rilevamento del dispositivo
import torch
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device = torch.device("cuda:0")  # Sceglie la prima GPU
else:
    device = torch.device("cpu")  # Sceglie la CPU se CUDA non è disponibile o non ci sono GPU



# %%
from datasets import load_dataset

dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")

# %% [markdown]
# As we can see, the dataset contains 3 splits: one for training, one for validation and one for testing.

# %%

print(dataset )

# %% [markdown]
# Let's check the first example of the training split:

# %%
example = dataset['train'][0]
print(example)

# %% [markdown]
# The dataset consists of tweets, labeled with one or more emotions.
# 
# Let's create a list that contains the labels, as well as 2 dictionaries that map labels to integers and back.

# %%
labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
labels

# %% [markdown]
# ## Preprocess data
# 
# As models like BERT don't expect text as direct input, but rather `input_ids`, etc., we tokenize the text using the tokenizer. Here I'm using the `AutoTokenizer` API, which will automatically load the appropriate tokenizer based on the checkpoint on the hub.
# 
# What's a bit tricky is that we also need to provide labels to the model. For multi-label text classification, this is a matrix of shape (batch_size, num_labels). Also important: this should be a tensor of floats rather than integers, otherwise PyTorch' `BCEWithLogitsLoss` (which the model will use) will complain, as explained [here](https://discuss.pytorch.org/t/multi-label-binary-classification-result-type-float-cant-be-cast-to-the-desired-output-type-long/117915/3).

# %%
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

# %%
encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

# %%
example = encoded_dataset['train'][0]
print(example.keys())

# %%
tokenizer.decode(example['input_ids'])

# %%
example['labels']

# %%
_testo=[id2label[idx] for idx, label in enumerate(example['labels']) if label == 1.0]
print(_testo)
# %% [markdown]
# Finally, we set the format of our data to PyTorch tensors. This will turn the training, validation and test sets into standard PyTorch [datasets](https://pytorch.org/docs/stable/data.html).

# %%
encoded_dataset.set_format("torch")

# %% [markdown]
# ## Define model
# 
# Here we define a model that includes a pre-trained base (i.e. the weights from bert-base-uncased) are loaded, with a random initialized classification head (linear layer) on top. One should fine-tune this head, together with the pre-trained base on a labeled dataset.
# 
# This is also printed by the warning.
# 
# We set the `problem_type` to be "multi_label_classification", as this will make sure the appropriate loss function is used (namely [`BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)). We also make sure the output layer has `len(labels)` output neurons, and we set the id2label and label2id mappings.

# %%
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                           problem_type="multi_label_classification",
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)

# %% [markdown]
# ## Train the model!
# 
# We are going to train the model using HuggingFace's Trainer API. This requires us to define 2 things:
# 
# * `TrainingArguments`, which specify training hyperparameters. All options can be found in the [docs](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments). Below, we for example specify that we want to evaluate after every epoch of training, we would like to save the model every epoch, we set the learning rate, the batch size to use for training/evaluation, how many epochs to train for, and so on.
# * a `Trainer` object (docs can be found [here](https://huggingface.co/transformers/main_classes/trainer.html#id1)).

# %%
batch_size = 8
metric_name = "f1"

# %%
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

# %% [markdown]
# We are also going to compute metrics while training. For this, we need to define a `compute_metrics` function, that returns a dictionary with the desired metric values.

# %%
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

# %% [markdown]
# Let's verify a batch as well as a forward pass:

# %%
encoded_dataset['train'][0]['labels'].type()

# %%
encoded_dataset['train']['input_ids'][0]

# %%
#forward pass
outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))
outputs

# %% [markdown]
# Let's start training!

# %%
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# %%
trainer.train()

# %% [markdown]
# ## Evaluate
# 
# After training, we evaluate our model on the validation set.

# %%
trainer.evaluate()

# %% [markdown]
# ## Inference
# 
# Let's test the model on a new sentence:

# %%
text = "I'm happy I can finally train a model for multi-label classification"

encoding = tokenizer(text, return_tensors="pt")
encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

outputs = trainer.model(**encoding)

# %% [markdown]
# The logits that come out of the model are of shape (batch_size, num_labels). As we are only forwarding a single sentence through the model, the `batch_size` equals 1. The logits is a tensor that contains the (unnormalized) scores for every individual label.

# %%
logits = outputs.logits
logits.shape

# %% [markdown]
# To turn them into actual predicted labels, we first apply a sigmoid function independently to every score, such that every score is turned into a number between 0 and 1, that can be interpreted as a "probability" for how certain the model is that a given class belongs to the input text.
# 
# Next, we use a threshold (typically, 0.5) to turn every probability into either a 1 (which means, we predict the label for the given example) or a 0 (which means, we don't predict the label for the given example).

# %%
# apply sigmoid + threshold
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1
# turn predicted id's into actual label names
predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
print(predicted_labels)

