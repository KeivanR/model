import numpy as np
def sigmoid(x):
    return (1/(1+np.exp(-x)))
np.random.seed(1)
X = (2*np.random.random((10,4))).astype(int)
print('input: ',X)
Y = (2*np.random.random(10)).astype(int).T
print('output: ',Y)
s0 = 2*np.random.random((4,7))-1
s1 = 2*np.random.random((7,1))-1
epochs = 1000
for i in range(0,epochs):
    print(i)
    l1 = sigmoid(np.dot(X,s0))
    print('l1:',l1)
    l2 = sigmoid(np.dot(l1,s1))
    print('l2:',l2)
    error = Y-l2
    
    delta_s1 = error*(l2*(1-l2))
    print('delta_s1:',delta_s1)
    delta_s0 = np.dot(delta_s1,s1.T)*(l1*(1-l1))
    print('delta_s0:',delta_s0)
    s1 += np.dot(l1.T,delta_s1)
    s0 += np.dot(X.T,delta_s0)