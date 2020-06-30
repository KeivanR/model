import pickle

# Saving the objects:
def save(obj,name):
	with open('C:/Users/Diana/Documents/Models/'+name+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
		pickle.dump(obj, f)
	# Getting back the objects:
def get(name):
	with open('C:/Users/Diana/Documents/Models/'+name+'.pkl','rb') as f:  # Python 3: open(..., 'rb')
		return pickle.load(f)