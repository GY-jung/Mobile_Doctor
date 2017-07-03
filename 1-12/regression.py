# coding=utf-8 
# 아이의 정보 (체중, 성별, 나이) 와 섭취한 해열제 종류 및 양, 해열제 먹인 순간의 측정 초기 온도, 해열제 먹인 후 경과 시간을 바탕으로, 초기에 비해 온도 감소량이 얼마나 될지 예측할 수 있는 모델을 설계함.
# 모델은 hidden layer가 2개인 neural net이고, 모델에 의해 예측된 체온감소와 실제 체온감소의 차의 제곱을 최소화하도록 설계함
# 대략 100번정도의 iteration이 진행되면 validation set에서 모델에 의해 예측된 체온감소와 실제 체온감소의 차의 제곱의 평균이 0.65 ~ 0.7 정도 나옴. 즉 체온 감소량을 평균적으로 0.8도 범위 정도로 예측함.

# 디렉토리에 "final%d.csv" %datanum 을 필요로 함 

import numpy as np
import csv
import time
import tensorflow as tf
from preprocessing import *
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 모델 hyperparameter
eps = 1e-1
lr = 0.001
decay = 0.9
scalable = [0,1,16,18]
hidden1 = 100
hidden2 = 50
datanum = 50000


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#데이터 로딩

data = tonumpy(csvreader("final%d.csv" %datanum))
data = np.array(data, dtype = "float")

np.random.shuffle(data)
printmax(data)
printmin(data)
data = np.transpose(data)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#dataset 분할

train_set = data[:, 0:int(data.shape[1] * 0.6)]
dev_set = data[:, int(data.shape[1]* 0.6) : int(data.shape[1]* 0.8)]
test_set = data[:, int(data.shape[1]* 0.8):]             # 6 : 2 : 2 로 training / validation / test 을 분할함


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 체온 감소량을 예측할 것이므로 체온 감소량을 제외한 값들을 training에 사용
temp_mask = np.ones(data.shape[0], dtype = "bool")
temp_mask[3] = False                                    

X_train = np.array(train_set[temp_mask], dtype = "float")
temp_train = np.array(train_set[3], dtype = "float")

X_dev = np.array(dev_set[temp_mask], dtype = "float32")
temp_dev = np.array(dev_set[3], dtype = "float32")

X_test = np.array(test_set[temp_mask], dtype = "float32")
temp_test = np.array(test_set[3], dtype = "float32")

X_train = np.transpose(X_train)

X_dev = np.transpose(X_dev)
X_test = np.transpose(X_test)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#some tensorflow...

#placeholder
inputs_placeholder = tf.placeholder(tf.float32, (None, X_train.shape[1]), name = "inputs")
temps_placeholder = tf.placeholder(tf.float32, (None,), name = "temps")

#affine layers
W1 = tf.Variable(tf.random_uniform((X_train.shape[1],hidden1), minval = -np.sqrt(6.0/(X_train.shape[1] + hidden1) ), maxval = np.sqrt(6.0/(X_train.shape[1] + hidden1))), dtype = tf.float32)
b1 = tf.Variable(tf.zeros((1,hidden1)) , dtype = tf.float32)
W2 = tf.Variable(tf.random_uniform((hidden1,hidden2), minval = -np.sqrt(6.0/(hidden1 + hidden2)), maxval = np.sqrt(6.0/(hidden1 + hidden2))), dtype = tf.float32)
b2 = tf.Variable(tf.zeros((1,hidden2)), dtype = tf.float32)
W3 = tf.Variable(tf.random_uniform((hidden2,1), minval = -np.sqrt(6.0/hidden2), maxval = np.sqrt(6.0/hidden2)), dtype = tf.float32)
b3 = tf.Variable(tf.zeros((1,1)), dtype = tf.float32)

#activation : tanh 
z1 = tf.matmul(inputs_placeholder, W1)  + b1
h1 = tf.nn.tanh(z1)
z2 = tf.matmul(h1, W2) + b2
h2 = tf.nn.tanh(z2)
z3 = tf.add(tf.matmul(h2, W3) , b3, name = "op_to_restore")  # 예상 체온



y1 = tf.matmul(X_dev, W1) + b1
g1 = tf.nn.tanh(y1)
y2 = tf.matmul(g1, W2) + b2
g2 = tf.tanh(y2)
y3 = tf.matmul(g2,W3) + b3

#L2 loss
loss = tf.reduce_mean((z3 - temps_placeholder)**2)
dev_loss = tf.reduce_mean((y3 - temp_dev)**2)


#model 재사용을 위한 saver
saver = tf.train.Saver()

#Adamoptimizer
optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Training. 5번의 step마다 loss 확인. devset에서 loss가 늘어난 경우에 learning rate를 0.9배로 decay

best_dev = 10
for step in range(101):
    sess.run(train_op, feed_dict = {inputs_placeholder : X_train, temps_placeholder : temp_train})

    if step%5 == 0:   # 매 스텝마다 예상 감소 체온과 실제 감소 체온 3개, train_loss, dev_loss 출력
        print(step, sess.run(loss,  feed_dict = {inputs_placeholder : X_train, temps_placeholder : temp_train}), sess.run(loss, feed_dict = {inputs_placeholder : X_dev, temps_placeholder : temp_dev}))
        print(sess.run((z3[3],z3[14], z3[159]), feed_dict = {inputs_placeholder : X_train, temps_placeholder : temp_train}))
        print (temp_train[3], temp_train[14], temp_train[159])
        if best_dev < sess.run(loss, feed_dict = {inputs_placeholder : X_dev, temps_placeholder : temp_dev}) : 
            lr *= decay
            print ("decay")
            print (lr)
        else :
            best_dev = sess.run(dev_loss, feed_dict = {inputs_placeholder : X_train, temps_placeholder : temp_train})

print ("test set loss")
print (sess.run(loss, feed_dict = {inputs_placeholder : X_test, temps_placeholder : temp_test}))

#save model. predict에서 사용함
save_path = saver.save(sess, 'C:\\Users\\MobileDoctor\\1-12\\model')





 











