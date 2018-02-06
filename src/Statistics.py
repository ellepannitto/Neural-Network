'''
 this module implements some functions of Error and Accuracy.
'''

import numpy as np
import math

class MSELoss:
	'''
	  
	  implements the Mean Squared Error function, given by:
	  
	               ___
	          1 1  \                
	    MSE = - -  /__ || op - yp ||
	          2 N   p               2
	   
	   where p is a pattern, op is the output of the neural network for pattern p and yp is the expected output for pattern p.
	   N is the number of patterns.
	   || op - yp ||   is the norm 2 of the difference vector op - yp
	                2 
	  
	'''
	
	def __init__(self):
		'''
		  creates an instance of MSELoss, for which the loss is zero
		'''
		
		self.loss = 0.0
		self.num_patterns = 0
		
	def update (self, predicted, gold):
		'''
		  updates the loss, summing the contribution of a pattern p
		  
		  :params:
		   predicted: the neural network output for the pattern p
		   gold:      the expected output for the pattern p
		'''
		
		self.num_patterns += 1
		for i in range(len(predicted)):
			self.loss += 0.5*((predicted[i] - gold[i])**2)
	
	def get ( self):
		'''
		  gets the MSE Loss value
		'''
		return self.loss / self.num_patterns

class MEELoss:
	'''
	  
	  implements the Mean EUCLIDEAN Error function, given by:
	  
	             ___    ________________
	          1  \     /                
	    MEE = -  /__ \/  || op - yp ||
	          N   p                   2
	   
	   where p is a pattern, op is the output of the neural network for pattern p and yp is the expected output for pattern p.
	   N is the number of patterns.
	   || op - yp ||   is the norm 2 of the difference vector op - yp
	                2 
	  
	'''
	
	def __init__(self):
		'''
		  creates an instance of MEELoss, for which the loss is zero
		'''
		self.loss = 0.0
		self.num_patterns = 0
		
	def update (self, predicted, gold):
		'''
		  updates the loss, summing the contribution of a pattern p
		  
		  :params:
		   predicted: the neural network output for the pattern p
		   gold:      the expected output for the pattern p
		'''
		
		self.num_patterns += 1
		dist = 0
		for i in range(len(predicted)):
			dist += (predicted[i] - gold[i])**2
		self.loss += math.sqrt ( dist )
	
	def get ( self ):
		'''
		  gets the MEE Loss value
		'''
		return self.loss / self.num_patterns
	
class Accuracy:
	'''
	  
	  implements binary classification mean accuracy measure, defined as:
	  
	             ___ 
	          1  \              
	    ACC = -  /__ I ( op, yp )
	          N   p              
	   
	   where p is a pattern, op is the output of the neural network for pattern p and yp is the expected output for pattern p.
	   N is the number of patterns.
	                                                      _
	                                                     |   1 if op == yp
	   I ( op, yp ) is the indicator variable defined as |
	                                                     |_  0 otherwise
	  
	'''
	
	def __init__ (self):
		'''
		  initialize an instance of class Accuracy, for which the accuracy is zero.
		'''
		
		self.correctly_predicted = 0
		self.num_patterns = 0
	
	def update ( self, predicted, gold ):
		'''
		  updates the accuracy, summing the contribution of a pattern p
		  
		  :params:
		   predicted: the neural network output for the pattern p
		   gold:      the expected output for the pattern p
		'''
		
		#~ print ("predicted {}".format(predicted))
		#~ print ("gold {}".format(gold))
		self.num_patterns += 1
		self.correctly_predicted += 1 if all ( [ (o-0.5)*(y-0.5)>0 for o,y in zip (predicted, gold) ] ) else 0
	
	def get ( self):
		'''
		  gets the Accuracy value
		'''
		return self.correctly_predicted*1.0 / self.num_patterns

class MulticlassificationAccuracy:
	'''
	  
	  implements multiclass classification task mean accuracy measure, defined as:
	  
	             ___ 
	          1  \              
	    MCA = -  /__ I ( argmax (op), argmax(yp) )
	          N   p              
	   
	   where p is a pattern, op is the output of the neural network for pattern p and yp is the expected output for pattern p.
	   N is the number of patterns.
	                                                                      _
	                                                                     |   1 if argmax(op) == argmax(yp)
	   I ( argmax(op), argmax(yp) ) is the indicator variable defined as |
	                                                                     |_  0 otherwise
	  
	'''
	
	def __init__ (self):
		'''
		  Initializes an instance of class MulticlassificationAccuracy, for which the accuracy is 0.
		'''
		self.correctly_predicted = 0
		self.num_patterns = 0
	
	def update ( self, predicted, gold ):
		'''
		  updates the accuracy, summing the contribution of a pattern p
		  
		  :params:
		   predicted: the neural network output for the pattern p
		   gold:      the expected output for the pattern p
		'''
		
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
		'''
		  gets the accuracy value
		'''
		return self.correctly_predicted*1.0 / self.num_patterns
