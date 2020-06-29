import numpy as np

#activation functions
def sigmoid(x):
	return 1/(1+np.exp(-x))
def tanh(x):
	return np.tanh(x)
def reLU(x):
	return np.maximum(0,x)
def preLU(x):
	return np.maximum(0.01*x,x)
#activation function gradients
def activation_grad(function,x):
	if function == sigmoid:
		return sigmoid(x)*(1-sigmoid(x))
	elif function == tanh:
		return 1-tanh(x)**2
	elif function == reLU or function == preLU:
		res = function(x)/x
		res[np.isnan(res)] = 0
		return res
#loss functions
def multiclass_SVM_loss(output,y):
	margins = np.maximum(0,output-output[y]+1)
	margins[y]=0
	return np.sum(margins)
def softmax_loss(output,y):
	return -np.log(np.exp(output[y])/np.sum(np.exp(output)))
#loss function gradients
def loss_grad(function,output,y):
	if function == softmax_loss:
		all_exp = np.exp(output)
		res = all_exp/np.sum(all_exp)
		res[y] -= 1
		return res
	if function == multiclass_SVM_loss:
		res = 0*output
		resy = 0
		for i in range(0,len(output)):
			if output[i]-output[y]+1>0:
				res[i]=1
				resy -= 1
		res[y] = resy
		return res
class Trainer():
	def __init__(self,reg_coeff,reg_type,update):
		self.reg_coeff = reg_coeff
		self.reg_type = reg_type
		self.update = update
	def get_loss(self):
		return self.loss	
	def gradient(self,model,score_func,y):
		grad = [None]*len(model.synapses)
		last_grad = loss_grad(score_func,model.layers[-1],y)
		for i in range(len(model.synapses)-1,0,-1):
			grad[i] = model.synapses[i].get_grad(model.layers[i-1],last_grad)
	def train(self,X,y,model,num_epochs,batch_size,l_rate,score_func):
		for i in range(0,num_epochs):
			for j in range(0,int(len(X)/batch_size)):
				interval = np.arange(j*batch_size,(j+1)*batch_size)
				if i==0:
					print(interval[0],':',interval[-1])
				self.loss = 0
				for k in range(0,len(interval)):
					model.feed_fwd(X[interval][k])
					loss = score_func(model.layers[-1],y[interval][k])
					grad += model.get_grad(loss)
					self.loss += loss
				
				self.loss = self.loss/len(interval)+self.reg_coeff*model.get_sumW(self.reg_type)
				
				   
			interval = np.arange(interval[-1]+1,len(X))
			if len(interval)>0:
				print('rest of batch:')
				if i==0:
					print(interval[0],':',interval[-1])
				loss = 0
				for k in range(0,len(interval)):
					model.feed_fwd(X[interval][k])
					loss += score_func(model.layers[-1],y[interval][k])
				self.loss = loss/len(interval)+self.reg_coeff*model.get_sumW(self.reg_type)
				model.backprop(self.loss)
		
class Model():
	def __init__(self,input_size,output_size):
		self.input_size = input_size
		self.output_size = output_size
		self.current_output_size = input_size
		self.synapses = []
		self.layers = []
	def add_filters(self,n,size,padding,stride,name,activation):
		self.synapses.append(Filter(n,size,padding,stride,name,self.current_output_size,activation))
		self.layers.append(0)
		self.current_output_size = self.synapses[-1].output_size
	def add_FC(self,size,name,activation):
		self.synapses.append(FC(size,name,self.current_output_size,activation))
		self.layers.append(0)
		self.current_output_size = size
	def feed_fwd(self,input):
		for i in range(0,len(self.synapses)):
			self.layers[i]=self.synapses[i].feed_fwd(input)
			input = self.layers[i]
	def get_sumW(self,type):
		reg = 0
		if type == 'L2':
			for i in range(0,len(self.synapses)):
				reg+=np.sum(np.square(self.synapses[i].W))
			return reg	
		if type == 'L1':
			for i in range(0,len(self.synapses)):
				reg+=np.sum(np.abs(self.synapses[i].W))
			return reg	
	
	
		
		
class Filter():
	def __init__(self,n,size,padding,stride,name,input_size,activation):
		self.number = n
		self.size = size
		self.padding = padding
		self.stride = stride
		self.name = name
		self.activation = activation
		self.W = []
		self.b = []
		for i in range(0,n):
			self.W.append(0.0001*np.random.randn(size[0],size[1],input_size[2]))
			self.b.append(0)
		self.W = np.asarray(self.W)
		self.b = np.asarray(self.b)
		out_sizeX = int((input_size[0]+2*padding-size[0])/stride+1)
		out_sizeY = int((input_size[1]+2*padding-size[1])/stride+1)
		if (int(out_sizeX)!=out_sizeX):
			print("the stride for filter ",name," does not fit X input size")
		if (int(out_sizeY)!=out_sizeY):
			print("the stride for filter ",name," does not fit Y input size")
		self.output_size = [out_sizeX,out_sizeY,n]
	def feed_fwd(self,input):
		layer = np.zeros(self.output_size)
		for k in range(0,self.number):
			for i in range(0,self.output_size[0]):
				for j in range(0,self.output_size[1]):
					inputConv = input[
						max(0,i*self.stride-self.padding):min(input.shape[0],i*self.stride+self.size[0]-self.padding),
						max(0,j*self.stride-self.padding):min(input.shape[1],j*self.stride+self.size[1]-self.padding),
						:
						]
					WConv = self.W[k][
						max(0,self.padding-i*self.stride):min(self.W[k].shape[0],self.W[k].shape[0]-(i*self.stride+self.size[0]-self.padding-input.shape[0])),
						max(0,self.padding-j*self.stride):min(self.W[k].shape[1],self.W[k].shape[0]-(j*self.stride+self.size[1]-self.padding-input.shape[1])),
						:
						]
					layer[i,j,k] = self.activation(np.dot(inputConv.flatten(),WConv.flatten())+self.b[k])
		return layer
	def get_grad(self,input,last_grad):
		for k in range(0,self.number):
			
			bgrad = last_grad*activation_grad(self.activation,np.dot(input.flatten(),self.W)+self.b)
			wgrad = last_grad*activation_grad(self.activation,np.dot(input.flatten(),self.W)+self.b)*input
			xgrad = last_grad*activation_grad(self.activation,np.dot(input.flatten(),self.W)+self.b)*self.W
			next_grad = np.sum(xgrad,axis=0)

class FC():
	def __init__(self,size,name,input_size,activation):
		self.size = size
		self.name = name
		self.activation = activation
		print(input_size)
		if len(input_size)>1:
			streched_input_size = input_size[0]*input_size[1]*input_size[2]
		else:
			streched_input_size = input_size
		print(streched_input_size)
		self.W = 0.0001*np.random.randn(streched_input_size,size)
		self.b = np.zeros(size)
	def feed_fwd(self,input):
		return self.activation(np.dot(input.flatten(),self.W)+self.b)
	def get_grad(self,input,last_grad):
		bgrad = last_grad*activation_grad(self.activation,np.dot(input.flatten(),self.W)+self.b)
		wgrad = np.outer(input.flatten(),last_grad*activation_grad(self.activation,np.dot(input.flatten(),self.W)+self.b))
		xgrad = np.dot(last_grad*activation_grad(self.activation,np.dot(input.flatten(),self.W)+self.b),self.W.T)
		xgrad = xgrad.reshape(input.shape)

model = Model([10,10,3],[1,10])

print(model.output_size)
#print(model.current_output_size)
model.add_filters(1,[5,5],2,1,'Filter 1',reLU)
print(model.synapses[0].W,model.synapses[0].b)
#print(model.current_output_size)
model.add_filters(5,[3,3],0,1,'Filter 2',reLU)
print(model.synapses[1].W,model.synapses[1].b)
#print(model.current_output_size)
model.add_FC(10,'FC 1',reLU)
print(model.synapses[2].W,model.synapses[2].b)
model.feed_fwd(np.ones((10,10,3)))
print(model.layers[2])
print(model.synapses[1].W.flatten())