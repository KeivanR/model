import numpy as np
def sigmoid(x):
    return (1/(1+np.exp(-x)))
def update_syn(X,Y,s0,s1,interval):
    l1 = sigmoid(np.dot(X[interval],s0))
    l2 = sigmoid(np.dot(l1,s1))
    error = Y[interval]-l2
    delta_l2 = error*(l2*(1-l2))
    delta_l1 = np.dot(delta_l2,s1.T)*(l1*(1-l1))
    s1 += np.dot(l1.T,delta_l2)
    s0 += np.dot(X[interval].T,delta_l1)
    return{'s0':s0,'s1':s1}
np.random.seed(1)
training_size = 40
input_size = 10
l1_size = 7
batch_size = 1
X = (2*np.random.random((training_size,input_size))).astype(int)
print('input: ',X)
Y = (2*np.random.random((training_size,1))).astype(int)
print('output: ',Y)
s0 = 2*np.random.random((input_size,l1_size))-1
s1 = 2*np.random.random((l1_size,1))-1
epochs = 1000
for i in range(0,epochs):
    for j in range(0,int(training_size/batch_size)):
        interval = np.arange(j*batch_size,(j+1)*batch_size)
        if i==0:
            print(interval[0],':',interval[-1])
        up = update_syn(X,Y,s0,s1,interval)
        s0 = up['s0']
        s1 = up['s1']   
    interval = np.arange(interval[-1]+1,training_size)
    if len(interval)>0:
        print('rest of batch:')
        if i==0:
            print(interval[0],':',interval[-1])
        up = update_syn(X,Y,s0,s1,interval)
        s0 = up['s0']
        s1 = up['s1']
    
l1 = sigmoid(np.dot(X,s0))
l2 = sigmoid(np.dot(l1,s1))
print('output after training:',(10*l2).astype(int)/10)
print('error:')
print(np.mean(np.abs(Y-l2)))