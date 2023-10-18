
# INFORM: questo funziona  con GPU 

# Importing required libraries
import  tensorflow as  tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from kerastuner.tuners import RandomSearch


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')  # Sceglie la prima GPU


# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Model building function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
                    activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'sgd']),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

# Initialize Keras Tuner's RandomSearch
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='my_dir',
    project_name='mnist_kt')

# Perform hyperparameter search
tuner.search(train_images, train_labels,
             epochs=5,
             validation_data=(test_images, test_labels), verbose=2)

# Using KerasClassifier with sklearn's GridSearchCV
from sklearn.model_selection import GridSearchCV

def build_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# model = KerasClassifier(build_fn=build_model, epochs=5, batch_size=32)
# param_grid = {'optimizer':['sgd', 'adam']}
# grid = GridSearchCV(estimator=model, param_grid=param_grid)
# grid_result = grid.fit(train_images, train_labels)
