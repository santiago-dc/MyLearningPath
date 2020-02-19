import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
print('=)')

#load data
df_train = pd.read_csv('data/mnist_train.csv', delimiter = ',')
df_test = pd.read_csv('data/mnist_test.csv', delimiter = ',')

df_train = df_train.dropna(how='any',axis=0)
df_test = df_test.dropna(how='any',axis=0)

def extract(df_train, df_test):
    df_train = pd.read_csv('data/mnist_train.csv', delimiter = ',').to_numpy()
    df_test = pd.read_csv('data/mnist_test.csv', delimiter = ',').to_numpy()
    df = pd.DataFrame(np.concatenate((df_train,df_test)))
    return df

def transform(df): 
    y_raw = df.iloc[:, 0].to_numpy()
    x = df.drop(df.columns[0], axis='columns').to_numpy()
    y = []
    for i in range(0,y_raw.shape[0]):
        row=np.zeros(10)
        row[y_raw[i]]=1
        y.append(row)
    y=np.array(y)
    y.reshape(y.shape[0],10)
    return x, y

df = extract(df_train, df_test)
x_train,y_train = transform(df.iloc[0:50000])
x_val,y_val = transform(df.iloc[50000:60000])
x_test,y_test = transform(df.iloc[60000:70000])
print(y_test.shape)

n_input = 784  # input layer (28x28 pixels)
n_hidden1 = 512  # 1st hidden layer
n_hidden2 = 256  # 2nd hidden layer
n_hidden3 = 128  # 3rd hidden layer
n_output = 10  # output layer (0-9 digits)

learning_rate = 0.01
n_iterations = 1000
batch_size = 128
threshold = 0.5

X = tf.placeholder("float", [None, x_train.shape[1]])#None = any amount of data
Y = tf.placeholder("float", [None, y_train.shape[1]])
keep_prob = tf.placeholder(tf.float32)

weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=Y, logits=output_layer
        ))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train on mini batches
for i in range(n_iterations):
    batch_x = x_train[(i*128):((i+1)*128)]
    batch_y = y_train[(i*128):((i+1)*128)]
    sess.run(train_step, feed_dict={
        X: batch_x, Y: batch_y, keep_prob: 0.5
        })

    # print loss and accuracy (per minibatch)
    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [cross_entropy, accuracy],
            feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
            )
        print(
            "Iteration",
            str(i),
            "\t| Loss =",
            str(minibatch_loss),
            "\t| Accuracy =",
            str(minibatch_accuracy)
            )

test_accuracy = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, keep_prob: 0.5})
print("\nAccuracy on test set:", test_accuracy)