import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

# Caricamento e preprocessamento del dataset MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28)).astype('float32') / 255
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Costruzione del modello
# model = tf.keras.Sequential([
#     Flatten(input_shape=(28, 28, 1)),
#     Dense(64, activation='relu'),
#     MultiHeadAttention(num_heads=8, key_dim=64),  # Layer di attenzione con 8 "teste"
#     Dense(64, activation='relu'),
#     Dense(10, activation='softmax')
# ])


# Definizione del modello
input_layer = Input(shape=(28, 28))
x = Flatten()(input_layer)
x = Dense(64, activation='relu')(x)
# Espandi le dimensioni per farle coincidere con le aspettative del layer MultiHeadAttention
x_expanded = tf.expand_dims(x, axis=1)
attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(x_expanded, x_expanded, return_attention_scores=False)
# Appiattisci l'output per i layer successivi
attention_output_flat = Flatten()(attention_output)
x = Dense(64, activation='relu')(attention_output_flat)
output_layer = Dense(10, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Compilazione del modello
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Addestramento del modello
model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_data=(test_images, test_labels))


# Facciamo una previsione sul primo elemento del set di test
test_sample = test_images[0:1]  # Prendiamo il primo elemento e manteniamo la dimensione batch
prediction = model.predict(test_sample)

# La variabile "prediction" conterrà le probabilità predette per ciascuna delle 10 classi
print("Probabilità predette per ciascuna classe:", prediction)

# Per ottenere l'indice della classe con la probabilità più alta:
predicted_class = tf.argmax(prediction, axis=1).numpy()
print("Classe predetta:", predicted_class)
