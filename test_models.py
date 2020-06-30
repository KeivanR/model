from CNN import *
import NN_database as nd
import model_storage as ms
getter = nd.Datagetter()
X_test,y_test,mndata = getter.get_MNIST('test',printfirst=0)
X_test_square = X_test.reshape((10000,28,28,1))
model = ms.get('model5')
index = 0
acc = 0
print('Press any button to start testing')
while index<10000:
	model.feed_fwd(X_test_square[index])
	if np.argmax(model.layers[-1])==y_test[index]:
		acc += 1
	else:
		print('WROOOOOOOOOOOOOOOOOOOOOOOOOONG!!!')
		input()
		print(mndata.display(X_test[index]))
		print(y_test[index])
	
		for i in range(0,len(model.layers[-1])):
			print('P(X=',i,') = ',int(100*probability(model.layers[-1],i))/100)
	
	index +=1
print('accuracy = ',acc/10000)