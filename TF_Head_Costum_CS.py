import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

# Load and preprocess MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28)).astype('float32') / 255
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Custom MultiHeadAttention class
class MyCustomMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model):
        super(MyCustomMultiHeadAttention, self).__init__()
        self.internal_mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

    def call(self, query, key, value):
        attention_output = self.internal_mha(query, key, value)
        return attention_output

# Instantiate an object of your custom class
multi_head_attention_layer = MyCustomMultiHeadAttention(num_heads=8, d_model=64)

# Building the model using the custom class
input_layer = Input(shape=(28, 28))
x = Flatten()(input_layer)
x = Dense(64, activation='relu')(x)
x_expanded = tf.expand_dims(x, axis=1)

# Use the custom attention layer here
attention_output = multi_head_attention_layer(x_expanded, x_expanded, x_expanded)
attention_output_flat = Flatten()(attention_output)

x = Dense(64, activation='relu')(attention_output_flat)
output_layer = Dense(10, activation='softmax')(x)
model = Model(inputs=input_layer, outputs=output_layer)

# Compiling and training the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_data=(test_images, test_labels))

# Making a prediction on the first test sample
test_sample = test_images[0:1]
prediction = model.predict(test_sample)

print("Predicted probabilities for each class:", prediction)
predicted_class = tf.argmax(prediction, axis=1).numpy()
print("Predicted class:", predicted_class)
