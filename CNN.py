import numpy as np

class Model():
	def __init__(self,input_size,output_size):
		self.input_size = input_size
		self.output_size = output_size
		self.current_output_size = input_size
		self.layers = []
	def add_filters(self,n,size,padding,stride,name):
		self.layers.append(Filter(n,size,padding,stride,name,self.current_output_size))
		self.current_output_size = self.layers[-1].output_size
	def add_FC(self,size,name):
		self.layers.append(FC(size,name,self.current_output_size))
		self.current_output_size = size
	
		
class Filter():
	def __init__(self,n,size,padding,stride,name,input_size):
		self.number = n
		self.size = size
		self.padding = padding
		self.stride = stride
		self.name = name
		self.W = []
		self.b = []
		for i in range(0,n):
			self.W.append(0.0001*np.random.randn(size[0],size[1],input_size[2]))
			self.b.append(0)
		out_sizeX = int((input_size[0]+2*padding-size[0])/stride+1)
		out_sizeY = int((input_size[1]+2*padding-size[1])/stride+1)
		if (int(out_sizeX)!=out_sizeX):
			print("the stride for filter ",name," does not fit X input size")
		if (int(out_sizeY)!=out_sizeY):
			print("the stride for filter ",name," does not fit Y input size")
		self.output_size = [out_sizeX,out_sizeY,n]

class FC():
	def __init__(self,size,name,input_size):
		self.size = size
		self.name = name
		print(input_size)
		if len(input_size)>1:
			streched_input_size = input_size[0]*input_size[1]*input_size[2]
		else:
			streched_input_size = input_size
		print(streched_input_size)
		self.W = 0.0001*np.random.randn(streched_input_size,size)
		self.b = np.zeros(size)

model = Model([10,10,3],[1,10])
print(model.output_size)
#print(model.current_output_size)
model.add_filters(1,[5,5],2,1,'Filter 1')
print(model.layers[0].W,model.layers[0].b)
#print(model.current_output_size)
model.add_filters(5,[3,3],0,1,'Filter 2')
print(model.layers[1].W,model.layers[1].b)
#print(model.current_output_size)
model.add_FC(10,'FC 1')
print(model.layers[2].W,model.layers[2].b)