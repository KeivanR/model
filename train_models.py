from CNN import *
import NN_database as nd
import model_storage as ms
getter = nd.Datagetter()
X_train,y_train,mndata = getter.get_MNIST('train',printfirst=0)
X_train = X_train.reshape((60000,28,28,1))
#print (y_train)
#print(X_train[1])
model = Model(X_train[0].shape,[1,10])


model.add_filters(3,[5,5],2,3,'Filter 1',preLU,0.0001)
#model.add_FC(300,'FC 0',preLU,0.0001)
#model.add_FC(100,'FC 0',preLU,0.0001)
model.add_FC(10,'final layer',preLU,0.0001)

trainer = Trainer(1e-3,'L2','sgd')

#print(model.synapses[2].W)
size_sample = 60000
num_epochs = 50
trainer.train(X_train[:size_sample],y_train[:size_sample],model,num_epochs = num_epochs,batch_size = 50,l_rate = 0.001,loss_func = softmax_loss)
print(trainer.get_loss())
all_W = model.get_W()
print('W: mean = ',np.mean(all_W),' ; std = ',np.std(all_W))
#print(model.synapses[2].W)
#ms.save(model,type(model.synapses[0]).__name__+' 300 100 preLU '+str(size_sample)+'x'+str(num_epochs))
ms.save(model,'model_test')
