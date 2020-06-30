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

model.add_FC(100,'FC 0',reLU)

model.add_FC(10,'FC 1',reLU)

trainer = Trainer(0,'L2','sgd')

#print(model.synapses[2].W)
trainer.train(X_train[:10000],y_train[:10000],model,num_epochs = 3,batch_size = 200,l_rate = 0.01,loss_func = softmax_loss)
#print(model.synapses[2].W)
ms.save(model,'model2')
