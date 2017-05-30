# imports
import tensorflow as tf
import csv
from random import shuffle

# variables
n_classes = 42
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 177
learning_rate = 0.001
training_epochs = 102400
batch_size = 250
display_step = 1
display_accuracy_step = 100
minimum_cost = 0.025

#tensorflow graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes]) 

# read data from csv
def read_wifidata(filename, train_size = 0.5, validation_size = 0.3, test_size = 0.2):
    # baca isi complete.csv, jadikan list
    data = list()
    # kelompokkan berdasarkan kode ruangan
    group_data = [[] for i in range(n_classes)]
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data = list(reader)
    # abaikan data beris pertama
    # menghasilkan list of lists dengan index sesuai ruangan (kolom terakhir)
    i = 1
    for row in data:
        if (i != 1):
            newdata = [0.5 + float(x)/2 for x in row[:177]]
            newdata.append(int(row[-1]))
            group_data[int(row[-1])].append(newdata)
        i = i + 1
    
    train = []
    test = []
    validation = []
    
    for wifidata in group_data:
        shuffle(wifidata)
        train_count = int(train_size * len(wifidata))
        validation_count = train_count + int(validation_size * len(wifidata))
        i = 0
        for apdata in wifidata:
            if(i < train_count):
                train.append(apdata)
            elif (i < validation_count):
                validation.append(apdata)
            else:
                test.append(apdata)
            i = i + 1
    shuffle(train)
    shuffle(test)
    shuffle(validation)
    
    return {'train': train, 'test' : test, 'validation' : validation};

#split data dan target
def split_data_and_target(data):
    data_x = []
    data_y = []
    for rowdata in data:
        data_x.append(rowdata[:177])
        #one hot encoding
        one_hot = ([0] * n_classes)
        one_hot[rowdata[-1]] = 1
        data_y.append(one_hot)
    return data_x, data_y

def multilayer_perceptron(x, weights, biases):
    # hidden layer dengan relu
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # hidden layer dengan relu
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # output layer dengan fungsi aktivasi linear
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# 0. Read dataset (Dataset.csv)
result = read_wifidata('Dataset.csv')
train = result['train']
test = result['test']
validation = result['validation']

train_data, train_target = split_data_and_target(train)
test_data, test_target = split_data_and_target(test)
validation_data, validation_target = split_data_and_target(validation)

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# multilayer perceptron model
pred = multilayer_perceptron(x, weights, biases)

# loss dan optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize variables
init = tf.global_variables_initializer()

# jalankan graph
with tf.Session() as sess:
    sess.run(init)
    
    # training
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(train_data)/batch_size)
        start = 0
        for i in range(total_batch):
            count = (i + 1) * batch_size
            batch_x = train_data[start:count]
            batch_y = train_target[start:count]
            start = count
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y:batch_y})
            avg_cost += c / total_batch
            
        if (epoch % display_step == 0):
            print("Epoch:", (epoch+1), " cost= {:.9f}".format(avg_cost))
            
        if (epoch % display_accuracy_step == 0):
            current_correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            current_accuracy = tf.reduce_mean(tf.cast(current_correct_prediction, "float"))
            print("Validation Accuracy:", current_accuracy.eval({x: validation_data, y: validation_target}))
        
        if (avg_cost < minimum_cost):
            print("Minimum cost achieved!")
            break
        
    print("Optimization finished!")
    
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test_data, y: test_target}))