'''
  this module contains:
   - a container for the model parameters (the ConfigurableParams class)
   - the default parameters for all the models.
'''

import math
import random
import numpy as np

#~ random.seed(0)

def sigmoid_function (x):
	'''
	  :returns: the sigmoid  function applied to the input `x`.
	'''
	#~ try:
		return 1.0 / (1 + math.exp(x*SLOPE) )
	#~ except Exception as e: 
		#~ print ("[ERROR] in sigmoid function x:{}".format(x))
		#~ print (e)
		
	#~ return 0

def derivative_sigmoid_function (x):
	'''
	  :returns: the derivative of the sigmoid  function applied to the input `x`.
	'''
	#~ try:
		return -SLOPE*math.exp(SLOPE*x) / ((1 + math.exp(x*SLOPE) )**2)
	#~ except Exception as e:
		#~ print ("[ERROR] in derivative sigmoid function x:{}".format(x))
		#~ print (e)
	#~ return 0
		
ActivationFunctions = {
	"sigmoid": sigmoid_function,
	"sigmoid_der": derivative_sigmoid_function,
	"id": lambda x: x,
	"id_der": lambda x: 1
}

def unit_weigth_initializer ():
	'''
	  :returns: 1.
	'''
	return 1

def constant_weigth_initializer_initializer ():
	'''
	  :returns: a function that, every time is called, returns the next weight choosen from a fixed list of possible weights.
	  useful for debugging.
	'''
	
	def constant_weigth_initializer ():
		'''
		  :returns: the next weight choosen from a fixed list of possible weights
		'''
		constant_weigth_initializer.nxt += 1
		return constant_weigth_initializer.weight_to_assign[constant_weigth_initializer.nxt]
		
	constant_weigth_initializer.weight_to_assign = [0.15, 0.2, 0.35, 0.25, 0.30, 0.35, 0.4, 0.45, 0.6, 0.5, 0.55, 0.6]
	constant_weigth_initializer.nxt = -1
	return constant_weigth_initializer 

#~ WEIGHTS = [-1, 1]
WEIGHTS = [-700, 700]
	
#~ def random_weigth_initializer ():
	#~ return random.uniform ( WEIGHTS[0], WEIGHTS[1] ) * 1.0 / 1000
def random_weigth_initializer ():
	'''
	  :returns: a random weight choosen from a normal distribution with mean 0 and variance 0.7
	'''
	return np.random.normal ( 0, 0.7 )


SLOPE = -1

ETA = 0.5
ALPHA = 0.1
LAMBDA = 0.02

ETA_DECAY=True
ETA_RANGE=[0.01, 0.5]
ETA_DECREASING_PERIOD=3/4

MINIBATCH=False
#set MINIBATCH_SAMPLE=1 for online version
MINIBATCH_SAMPLE=35

LAYERS_SIZE=(6,4)

NUM_TRIALS_PER_CONFIGURATION = 3
MAX_EPOCH = 200
NUM_FOLDS = 10

class ConfigurableParams:
	'''
	  a container from the model parameters 
	'''
	
	def __init__ ( self, params_dict ):
		'''
		  creates an instance of ConfigurableParameters, given a dictionary of parameters.
		  Each element in the dictionary has the form PARAM_NAME : value
		'''
		self.params_dict = params_dict
	
	def __getattr__ ( self, attr ):
		'''
		  :returns: the value of the parameters with name `attr`.
		'''
		return self.params_dict[attr]
		
	
