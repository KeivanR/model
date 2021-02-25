import common as c
import model_functions as mf

def input(video,x1,x2):
	return mf.light_qty(video,x1,x2)/255
def dyna_input(video,X1,X2):
	return mf.dyna_light_qty(video,X1,X2)/255
X = input(video,x1,x2)
class L2_model():
	def __init__(self,w1,w2):
		self.w1 = w1
		self.w2 = w2
		self.phr = 0
		self.lam = 0
	def feed_fwd(self,X):
		self.batch_size = len(X)
		self.phr_hist = np.zeros(self.batch_size)
		self.lam_hist = np.zeros(self.batch_size)
		for t in range(self.batch_size):
			self.phr = X[t]+self.w2*self.lam
			self.lam = max(0,self.w1*self.phr)
			self.phr_hist[t] = self.phr
			self.lam_hist[t] = self.lam
	def back_prop(self,Y):
		grad_lam = np.zeros((self.batch_size,self.batch_size))
		grad_phr = np.zeros((self.batch_size,self.batch_size))
		self.grad_w1 = np.zeros((self.batch_size,self.batch_size))
		self.grad_w2 = np.zeros((self.batch_size,self.batch_size))
		for i in range(self.batch_size):
			grad_lam[i,i] = self.lam_hist[i]-Y[i]
			grad_phr[i,i] = grad_lam[i,i]*self.w1
			self.grad_w1[i,i] = grad_lam[i,i]*self.phr_hist[t]
			self.grad_w2[i,i] = grad_phr[i,i]*self.lam_hist[i-1]
			for t in range(i-1,t1-1,-1):
				grad_lam[i,t] = grad_phr[i,t+1]*self.w2
				grad_phr[i,t] = grad_lam[i,t]*self.w1
				self.grad_w1[i,t] = grad_lam[i,t]*self.phr_hist[t]
				self.grad_w2[i,t] = grad_phr[i,t]*self.lam_hist[t-1]
	
	def update(self,alpha):
		self.w1 -= alpha*self.grad_w1.sum
		self.w2 -= alpha*self.grad_w2.sum