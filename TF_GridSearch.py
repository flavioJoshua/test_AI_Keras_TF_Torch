# Importing required libraries for TensorFlow example
import tensorflow as tf
from tensorflow.keras.datasets import mnist


# # Verificare la disponibilità della GPU
# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     device = '/GPU:0'
# else:
#     device = '/CPU:0'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')  # Sceglie la prima GPU


# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocessing
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Building the model
def build_and_train_model(optimizer):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Using SGD optimizer
print("Training model with SGD optimizer")
build_and_train_model(tf.keras.optimizers.SGD(lr=0.01, momentum=0.9))

# Using Adam optimizer
print("Training model with Adam optimizer")
build_and_train_model(tf.keras.optimizers.Adam(lr=0.001))

# Commenting all code for better readability
# The above code defines a function `build_and_train_model` which accepts an optimizer.
# Two optimizers, SGD and Adam, are then used to train a simple neural network on the MNIST dataset.
# The model consists of two layers and is compiled with the given optimizer.
# The function then trains the model using the `fit` method.

# import sys
# sys.exit("Messaggio di terminazione")



#INFORM: 2  OK funziona  

# from kerastuner.tuners import GridSearch
from keras_tuner.tuners  import gridsearch  #  da  usare .... al posto di quello sopra
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model(hp):
    model = Sequential()
    #model.add(Dense(64, activation='relu', input_shape=(28*28,)))
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
                    activation='relu', input_shape=(28 * 28,)))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    # Scelta dell'ottimizzatore
    optimizer = hp.Choice('optimizer', ['SGD', 'Adam'])
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

tuner = gridsearch.GridSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir3',  # WARM: se non cambi la directory e la vede piena  skippa il processo 
    project_name='helloworld'
)


# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocessing
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)



X=train_images
y=train_labels

X_val=test_images

y_val=test_labels


# Supponiamo che X_val e y_val siano i tuoi dati di convalida
# tuner.search(X, y, validation_data=(X_val, y_val))
tuner.search(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels),verbose=2 )  

# #BUG: 3  non funziona  errore di importa  di KerasClassifier  

# from sklearn.model_selection import GridSearchCV
# from keras.wrappers.scikit_learn import KerasClassifier
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# model = KerasClassifier(build_fn=build_model, epochs=10)
# param_grid = dict(optimizer=['SGD', 'Adam'])
# grid = GridSearchCV(estimator=model, param_grid=param_grid)
# grid_result = grid.fit(X, y)


