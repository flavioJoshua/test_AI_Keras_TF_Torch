
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

# Carica il dataset MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Crea il modello
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compila il modello
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Crea una callback TensorBoard
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)

# Addestra il modello
model.fit(train_images, train_labels,
          epochs=5,
          validation_data=(test_images, test_labels),
          callbacks=[tensorboard])

# Ora puoi avviare TensorBoard con il comando:
# tensorboard --logdir=./logs


# In questo codice:

# 1. Importiamo le librerie necessarie.
# 2. Carichiamo il dataset MNIST.
# 3. Creiamo un semplice modello di rete neurale.
# 4. Compiliamo il modello specificando l'ottimizzatore, la funzione di perdita e le metriche.
# 5. Creiamo una callback TensorBoard specificando la directory dei log (`./logs`).
# 6. Addestriamo il modello usando il metodo `fit`, passando i dati di training, il numero di epoche, i dati di validazione, e la callback TensorBoard.
# 7. Dopo l'addestramento, avviamo TensorBoard dal terminale usando il comando `tensorboard --logdir=./logs` per visualizzare i log.