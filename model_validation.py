import comet_ml
from music_generation import build_model
from music_generation import params
from music_generation import vocab_size
from music_generation import char2idx
from music_generation import idx2char
import tensorflow as tf
from tqdm import tqdm
import mitdeeplearning as mdl
from IPython import display as ipythondisplay
from IPython.display import Audio
import numpy as np
from scipy.io.wavfile import write

# Restoring latest checkpoint
model = build_model(vocab_size, params["embedding_dim"], params["rnn_units"], 1)
# Restoring weights for the last checkpoint after training
# download_experiment_asset(
#     experiment_key: 'ceb0eca07e8f49a0add953c0f52590f8', asset_id: str, output_path: str
# ) -> None
model.load_weights('/Users/juliaafonso/Documents/MITlab1/swift_baluster_8593_model.weights.h5')
model.summary()
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

