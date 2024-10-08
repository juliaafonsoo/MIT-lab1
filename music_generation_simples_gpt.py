
#setup 
import mitdeeplearning as mdl
import numpy as np
import os
import time
import functools
import tensorflow as tf
import keras
from keras import ops
from keras import models
from keras import layers
from keras import ops
import h5py
from tqdm import tqdm
from scipy.io.wavfile import write
import regex as re
import subprocess
import urllib
from IPython import display as ipythondisplay
from IPython.display import Audio

# Download the dataset
songs = mdl.lab1.load_training_data()

#Join our list of song strings into a single string containing all songs
songs_joined = "\n\n".join(songs)

# # Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")
chars = sorted(list(set(songs_joined)))
print(''.join(chars))

# create a mapping from characters to integers
char2idx = { ch:i for i,ch in enumerate(vocab) }
idx2char = np.array(vocab)


def vectorize_string(string):
    vectorized = np.array([char2idx[ch] for ch in string])
    return vectorized

vectorized_songs = vectorize_string(songs_joined)

print ('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))

# Batch definition to create training examples

def get_batch(vectorized_songs, seq_length, batch_size):
  # the length of the vectorized songs string
  n = vectorized_songs.shape[0] - 1
  # randomly choose the starting indices for the examples in the training batch
  idx = np.random.choice(n-seq_length, batch_size)

  input_batch = np.array([vectorized_songs[i:i+seq_length] for i in idx])
  output_batch = np.array([vectorized_songs[i+1:i+1+seq_length] for i in idx])

  # x_batch, y_batch provide the true inputs and targets for network training
  x_batch = np.reshape(input_batch, [batch_size, seq_length])
  y_batch = np.reshape(output_batch, [batch_size, seq_length])
  return x_batch, y_batch

x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)


#MODEL
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

model = build_model(len(vocab), embedding_dim=256, rnn_units=1024,batch_size=32)

#checking the model
model.summary()
x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
pred = model(x)
print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")

sampled_indices = tf.random.categorical(pred[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()


#Training the model: loss and training operations

#loss function
def compute_loss(labels, logits):
  loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
  return loss

example_batch_loss = compute_loss(y, pred)

  
#Hyperparameter setting and optimization
vocab_size = len(vocab)
params = dict(
  num_training_iterations = 4000,  
  batch_size = 8,  
  seq_length = 100,  
  learning_rate = 1e-2, 
  embedding_dim = 256,
  rnn_units = 1024,
)

#Define optimizer and training operation
model = build_model(vocab_size, params["embedding_dim"], params["rnn_units"], params["batch_size"])
optimizer = tf.keras.optimizers.Adam(params["learning_rate"])


#backprop operations
@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    y_hat = model(x)
    loss = compute_loss(y, y_hat)
  
  grads = tape.gradient(loss, model.trainable_variables) # gradiant

  optimizer.apply_gradients(zip(grads, model.trainable_variables)) # Applying the gradients to the optimizer
  return loss


#training 

history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')

if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
for iter in tqdm(range(params["num_training_iterations"])):

  x_batch, y_batch = get_batch(vectorized_songs, params["seq_length"], params["batch_size"])
  loss = train_step(x_batch, y_batch)

  history.append(loss.numpy().mean())
  plotter.plot(history)

  if iter % 100 == 0:
     model.save_weights('/Users/juliaafonso/Documents/MITlab1/model.weights.h5')

# Saving the trained model and the weights
model.save_weights('/Users/juliaafonso/Documents/MITlab1/model.weights.h5', overwrite=True)

# Restoring latest checkpoint
model = build_model(vocab_size, params["embedding_dim"], params["rnn_units"], batch_size=1)
# Restoring weights for the last checkpoint after training
model.load_weights('/Users/juliaafonso/Documents/MITlab1/model.weights.h5')
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

generated_text = generate_text(model, start_string="X", generation_length=1000)
print(generated_text)

#Play back generated songs

generated_songs = mdl.lab1.extract_song_snippet(generated_text)
print(generated_songs)

for i, song in enumerate(generated_songs):
  waveform = mdl.lab1.play_song(song)

  if waveform:
    print("Generated song", i)
    ipythondisplay.display(waveform)

    numeric_data = np.frombuffer(waveform.data, dtype=np.int16)
    wav_file_path = f"output_{i}.wav"
    write(wav_file_path, 88200, numeric_data)