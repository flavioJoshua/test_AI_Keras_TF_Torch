import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np

# Dati (30 coppie)
premise_sentences = [
    "Il cane abbaia al luna",
    "Mangio una mela rossa",
    "Vado al parco con gli amici",
    "Il cielo è nuvoloso",
    "Mi alleno in palestra",
    "Mi piace il cioccolato",
    "Guardo una serie tv",
    "Ho comprato un nuovo telefono",
    "Faccio jogging al mattino",
    "Mi piace nuotare in mare",
    "Il gatto dorme sul divano",
    "Prendo il caffè al bar",
    "Vado a fare shopping",
    "Mi piace leggere romanzi",
    "Ascolto musica rock",
    "Studio matematica",
    "Gioco a calcio",
    "Mi piace viaggiare",
    "Faccio yoga al mattino",
    "Mangio sushi",
    "Il sole sorge ad est",
    "Vado al cinema",
    "Sto imparando una nuova lingua",
    "Mi piace dipingere",
    "Faccio fotografie",
    "Faccio volontariato",
    "Mi diverto a fare puzzle",
    "Mi piace andare in bicicletta",
    "Mi piace cucinare",
    "Vado in biblioteca",
    "Ho un appuntamento dal dentista"
]

hypothesis_sentences = [
    "Il cane abbaia agli estranei",
    "Mi piacciono le mele", 
    "Mi piace passeggiare nel parco",
    "Il cielo è sereno",
    "Mi alleno in piscina",
    "Mi piace la cioccolata calda",
    "Guardo un film al cinema",
    "Ho venduto il mio vecchio telefono",
    "Faccio jogging al pomeriggio",
    "Mi piace nuotare in piscina",
    "Il gatto dorme sul letto",
    "Prendo il tè al bar",
    "Vado a fare una passeggiata",
    "Mi piace leggere poesie",
    "Ascolto musica classica",
    "Studio storia",
    "Gioco a tennis",
    "Mi piace viaggiare in Europa",
    "Faccio yoga la sera",
    "Mangio pizza",
    "Il sole tramonta ad ovest",
    "Vado al teatro",
    "Sto imparando a suonare uno strumento",
    "Mi piace disegnare",
    "Faccio sculture",
    "Faccio del volontariato in un rifugio",
    "Mi diverto a giocare a videogiochi",
    "Mi piace fare escursioni in montagna",
    "Mi piace sperimentare nuove ricette",
    "Vado al museo",
    "Ho un appuntamento dal parrucchiere"
]

labels = [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,0]

# Tokenizzazione
tokenizer = Tokenizer()
tokenizer.fit_on_texts(premise_sentences + hypothesis_sentences)
vocab_size = len(tokenizer.word_index) + 1

# Conversione del testo in sequenze di token
premise_sequences = tokenizer.texts_to_sequences(premise_sentences)
hypothesis_sequences = tokenizer.texts_to_sequences(hypothesis_sentences)

# Padding
premise_padded = pad_sequences(premise_sequences, padding='post')
hypothesis_padded = pad_sequences(hypothesis_sequences, padding='post')

# Divisione in set di addestramento e validazione
(premise_train, premise_val, hypothesis_train, hypothesis_val, labels_train, labels_val) = train_test_split(
    premise_padded, hypothesis_padded, labels, test_size=0.2, random_state=42)

# Creazione del modello
input_premise = tf.keras.layers.Input(shape=(None,), dtype='int32')
input_hypothesis = tf.keras.layers.Input(shape=(None,), dtype='int32')

embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=150)
premise_embedding = embedding_layer(input_premise)
hypothesis_embedding = embedding_layer(input_hypothesis)

lstm_layer = tf.keras.layers.LSTM(50)
premise_lstm = lstm_layer(premise_embedding)
hypothesis_lstm = lstm_layer(hypothesis_embedding)


# # Aggiunta di layer di dropout
# premise_lstm = tf.keras.layers.Dropout(0.5)(premise_lstm)
# hypothesis_lstm = tf.keras.layers.Dropout(0.5)(hypothesis_lstm)

merged = tf.keras.layers.concatenate([premise_lstm, hypothesis_lstm])

merged = tf.keras.layers.concatenate([premise_lstm, hypothesis_lstm])
dense = tf.keras.layers.Dense(50, activation='relu')(merged)
output = tf.keras.layers.Dense(3, activation='softmax')(dense)

model = tf.keras.models.Model(inputs=[input_premise, input_hypothesis], outputs=output)

# Compilazione e addestramento
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([premise_train, hypothesis_train], np.array(labels_train), epochs=20, batch_size=3, validation_data=([premise_val, hypothesis_val], np.array(labels_val)))

# Predizione e calcolo del punteggio F1
predictions = model.predict([premise_val, hypothesis_val])
predicted_labels = np.argmax(predictions, axis=1)

f1 = f1_score(labels_val, predicted_labels, average='weighted')
print(f"F1 Score: {f1}")
