import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import  numpy as  np

# Definiamo un modello di rete neurale semplice
# Define a simple neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images
    Dense(128, activation='relu'),  # Dense layer with 128 units and ReLU activation
    Dense(10, activation='softmax')  # Output layer with 10 units (one for each digit)
])


""" 
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images
    Normalization(),  # Normalize the pixel values
    Dense(128, activation='relu'),  # Dense layer with 128 units and ReLU activation
    Dense(10, activation='softmax')  # Output layer with 10 units (one for each digit)
])
 """


# Carichiamo i dati di addestramento e test
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# One-hot encode i dati di etichetta
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)



""" per  test  funziona è equivalente """
test_val=np.reshape(x_train,(x_train.shape[0],28 * 28  ))


# Addestriamo il modello per ogni ottimizzatore
for optimizer in [
    tf.keras.optimizers.SGD(),
    tf.keras.optimizers.SGD(momentum=0.9),
    tf.keras.optimizers.Adadelta(),
    tf.keras.optimizers.RMSprop(),
    tf.keras.optimizers.Adam(),
    tf.keras.optimizers.Adamax(),
    tf.keras.optimizers.Nadam()
]:
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Optimizer: {optimizer.name}, Accuracy: {score[1]}%")







# # Compiliamo il modello
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Addestriamo il modello
# model.fit(x_train, y_train, epochs=10)

# # Valutiamo il modello
# score = model.evaluate(x_test, y_test)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
