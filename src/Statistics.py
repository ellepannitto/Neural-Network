
import numpy as np

class MSELoss:
	
	def __init__(self):
		self.loss = 0.0
		
	def update (self, predicted, gold):
		for i in range(len(predicted)):
			self.loss += 0.5*((predicted[i] - gold[i])**2)
			
		#~ print	

class Accuracy:
	
	def __init__ (self):
		self.correctly_predicted = 0
		self.num_patterns = 0
	
	def update ( self, predicted, gold ):
		#~ print ("predicted {}".format(predicted))
		#~ print ("gold {}".format(gold))
		self.num_patterns += 1
		self.correctly_predicted += 1 if all ( [ (o-0.5)*(y-0.5)>0 for o,y in zip (predicted, gold) ] ) else 0
	
	def get ( self):
		return self.correctly_predicted*1.0 / self.num_patterns

class MulticlassificationAccuracy:
	
	def __init__ (self):
		self.correctly_predicted = 0
		self.num_patterns = 0
	
	def update ( self, predicted, gold ):
		self.num_patterns += 1
		one_of_k_predicted = [0]*len(predicted)
		pos_max = np.argmax( predicted )
		one_of_k_predicted[pos_max] = 1
		#~ print ("[DEBUG] predicted: {}".format(predicted))
		#~ print ("[DEBUG] gold: {}".format(gold))
		#~ print ("[DEBUG] one of k predicted: {}".format(one_of_k_predicted))
		#~ input()
		self.correctly_predicted += 1 if all ( [ (o-0.5)*(y-0.5)>0 for o,y in zip (one_of_k_predicted, gold) ] ) else 0

	def get ( self):
		return self.correctly_predicted*1.0 / self.num_patterns
