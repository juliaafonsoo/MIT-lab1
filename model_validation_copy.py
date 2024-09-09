import comet_ml
import tensorflow as tf
from tqdm import tqdm
import mitdeeplearning as mdl
from IPython import display as ipythondisplay
from IPython.display import Audio
import numpy as np
from scipy.io.wavfile import write


songs = mdl.lab1.load_training_data()
songs_joined = "\n\n".join(songs)
vocab = sorted(set(songs_joined))

char2idx = { ch:i for i,ch in enumerate(vocab) }
idx2char = np.array(vocab)

def LSTM(rnn_units):
  return tf.keras.layers.LSTM(
    rnn_units,
    return_sequences=True,
    recurrent_initializer='glorot_uniform',
    recurrent_activation='sigmoid',
     stateful=True
  )
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, embeddings_initializer='uniform'),
    LSTM(rnn_units),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
  ])
  model.build([batch_size, None])

  return model

model = build_model(len(vocab), 256, 1024, 1)
model.load_weights('/Users/juliaafonso/Documents/MITlab1/severe_dragster_model.weights.h5')
model.build(tf.TensorShape([1, None]))
model.summary()

#Prediction of a generated song
def generate_text(model, start_string, generation_length):

  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  tqdm._instances.clear()
  lstm_layer = model.layers[1] 
  lstm_layer.reset_states()

  for i in tqdm(range(generation_length)):
      predictions = model(input_eval)
      # Removing the batch dimension
      predictions = tf.squeeze(predictions, 0)

      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


generated_text = generate_text(model, start_string="X:", generation_length=10000)
generated_songs = mdl.lab1.extract_song_snippet(generated_text)

for i, song in enumerate(generated_songs):
  waveform = mdl.lab1.play_song(song)

  if waveform:
    print("Generated song", i)
    ipythondisplay.display(waveform)

