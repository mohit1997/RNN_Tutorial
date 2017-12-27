import tensorflow as tf

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length
num_layers = 3
X = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length, 1])

def lstm_cell():
	return tf.contrib.rnn.BasicLSTMCell(state_size)
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])

x_ = tf.unstack(X, axis=1, num=15)
output, layers = tf.contrib.rnn.static_rnn(stacked_lstm, x_, dtype=tf.float32)