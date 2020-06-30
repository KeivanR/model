from CNN import *
import NN_database as nd
import model_storage as ms
getter = nd.Datagetter()
X_train,y_train,mndata = getter.get_MNIST('train',printfirst=0)
X_train = X_train.reshape((60000,28,28,1))
#print (y_train)
#print(X_train[1])
model = Model(X_train[0].shape,[1,10])


#model.add_filters(2,[5,5],2,3,'Filter 1',reLU)

model.add_FC(300,'FC 0',tanh,0.01)
model.add_FC(10,'final layer',tanh,0.01)

trainer = Trainer(0,'L2','sgd')

#print(model.synapses[2].W)
size_sample = 60000
trainer.train(X_train[:size_sample],y_train[:size_sample],model,num_epochs = 5,batch_size = 50,l_rate = 0.01,loss_func = softmax_loss)
print(trainer.get_loss())
all_W = model.get_W()
print('W: mean = ',np.mean(all_W),' ; std = ',np.std(all_W))
#print(model.synapses[2].W)
ms.save(model,'model5')
