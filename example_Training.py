import torch
import  numpy as np

# Inizializza un tensore di logits come esempio
# I logits sono solitamente ottenuti come output da un modello di classificazione prima dell'attivazione
logits = torch.tensor([[2.0], [-1.0], [0.5]])

# Inizializza la funzione di attivazione Sigmoid
sigmoid = torch.nn.Sigmoid()

# Rimuove le dimensioni singole (1) dal tensore
# Ad esempio, se logits è di forma (3, 1), diventerà (3,)
squeezed_logits = logits.squeeze()

# Sposta il tensore alla CPU (se non è già lì)
cpu_logits = squeezed_logits.cpu()

# Calcola le probabilità applicando la funzione Sigmoid ai logits
probs = sigmoid(cpu_logits)




_probs_adj=np.copy(probs) 

_probs_adj[ np.where(probs  >=0.5)]=1
# Stampa le probabilità calcolate
print("Calculated Probabilities:  {0} ,  raw  possibilities : {1} ".format(_probs_adj  , probs   ))
