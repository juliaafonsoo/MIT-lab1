Recurrent Neural Network (RNN) for music generation.
Model trained to learn the patterns in raw sheet music in ABC notation and then use this model to generate new music.

Three layers are used to define the model:

1 - tf.keras.layers.Embedding: This is the input layer, consisting of a trainable lookup table that maps the numbers of each character to a vector with embedding_dim dimensions.
2 - tf.keras.layers.LSTM: Our LSTM network, with size units=rnn_units.
3 - tf.keras.layers.Dense: The output layer, with vocab_size outputs.

credits to MIT Introduction to Deep Learning
http://introtodeeplearning.com
