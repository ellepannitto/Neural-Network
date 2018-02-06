'''
 This module contains the implementation of the base unit of the artificial neural network (the Neuron class)
'''

import math
import Params



class Neuron:
	'''
	 Represents a single artificial neuron, trained with the backpropagation algorithm.
	  Neurons can be initialized with an activation function and initial weights, and fired with certain inputs. After many firings, the weights can be updated 
	  according to the backpropagation algorithm, applying momentum and regularization at the end of a training epoch.
	'''
	
	def __init__(self, params=Params):
		'''
		 
		 Creates a Neuron, given the params that the neuron uses to update its input weigths.
		  params can be an instance of the ConfigurableParams class, or the global default params, (those defined in the module Params).
		  requested Params are:
		   - ETA:    the learning rate
		   - LAMBDA: the regularizatione coefficient
		   - ALPHA:  the momentum coefficient
		  other params are ignored
		  
		'''
		self.params = params
		
	def initialize (self, fun_name, weights, weights_inizialization):
		'''
		
		 Initialize the Neuron, assigning an activation function and the initial weights to it.
		 :params:
		  fun_name:               can be "sigmoid" (the sigmoid function) or "id" (the identity function). Is the activation function of this Neuron.
		  weights:                the number of inputs of this Neuron.
		  weights_inizialization: function that returns a weight every time is called. It can be random.random, random.normal, random.uniform or something like
		                          these, or one of constant_weigth_initializer, random_weigth_initializer or unit_weight_initializer defined in the module Params.
		  
		'''
		
		self.activation_function = Params.ActivationFunctions[fun_name]
		self.activation_function_derivative = Params.ActivationFunctions[fun_name+"_der"]
		self.weights = [weights_inizialization() for i in range(weights)]
		#~ self.weights = [ for w in self.weights]
		
		self.dw = [0]*weights
		self.prev_dw = [0]*weights
		self.cumulative_prev_dw = [0]*weights
		
	def compute_net (self):
		'''
		  computes the net of this Neuron, using the memorized inputs.
		'''
		
		self.net = 0.0
		for i in range(len(self.inputs)):
			self.net+=self.weights[i]*self.inputs[i]

	def fire (self, inputs):
		'''
		  fires this Neuron on the passed inputs, memorizing the inputs and computing the net.
		  
		  :params:
		   inputs: list of shape (self.weights): the inputs to this neuron
		  
		  :returns:
		   the net of this Neuron
		'''
		
		self.inputs = inputs
		
		self.compute_net()
		
		self.out = self.activation_function(self.net)
		#~ print ("FIRING NEURON.")
		#~ print ("inputs  = {}".format(inputs))
		#~ print ("weights = {}".format(self.weights))
		#~ print ("net = {}".format(self.net))
		#~ print ("out = {}".format(self.out))
		#~ input()
		
		return self.out

	def update_backpropagation_output(self, d):
		'''
		 updates the gradient of the loss function for every weight of this Neuron, after feeding the neural network with a single pattern p.
		 this function assumes this Neuron is an output neuron.
		 
		 :params:
		  d: expected output for the feeded pattern (p).
		
		'''
		
		#~ print ("Updating backprop for an output neuron")
		#~ print ("The one with inputs weights {}".format(self.weights))
		
		#~ print ("-dE/do   =  d - out = {}".format(d-self.out))
		#~ print ("f' (net) = {}".format(self.activation_function_derivative(self.net)))
		self.bp = ( d - self.out ) * self.activation_function_derivative(self.net)
		#~ print ("-dE/dnet = (d - self.out)*f'(net) = {}".format(self.bp))
		
		#~ print ("Computing dw for each weight")
		for i in range(len(self.weights)):
			#~ print ("weight: {}".format(self.weights[i]))
			#~ print ("input : {}".format(self.inputs[i]))
			self.dw[i] += self.bp * self.inputs[i]
			#~ print ("dw = -dE/dnet * input = {}".format(self.dw[i]))
			
			#~ input()
	
	def update_backpropagation_hidden (self, neuroni, pesi):
		'''
		 updates the gradient of the loss function for every weight of this Neuron, after feeding the neural network with a single pattern p.
		 this function assumes this Neuron is an hidden neuron.
		 
		 :params:
		  neuroni: Neurons on the upper layer to which this Neuron is connected.
		  pesi:    weights of the connections between this Neuron and the neurons in the upper layer. 
		           pesi[i] is the weight of the connecton between self and neuroni[i]
		
		'''
		
		s = 0
		
		for i in range(len(neuroni)):
			s+= neuroni[i].getBp () * pesi[i]
		
		self.bp = s * self.activation_function_derivative (self.net)
			
		for i in range(len(self.weights)):
			self.dw[i] += self.bp * self.inputs[i]
			
	def update_weights(self, sum_w, examples_number=1, end_epoch=True):
		'''
		 
		 updates the input weights of this neuron, according to the backpropagation algorithm. 
		  The function considers all the contributes given by all the patterns that were given in input to this Neuron, using the `fire` and 
		  `update_backpropagation_output` and `update_backpropagation_hidden` functions.
		 
		 the weight are updated by a portion of the gradient of the Loss functions, applying momentum and regularization. The momentum and regularization
		 coefficients, as well as the learning rate are taken from the params given in the __init__ function. 
		
		 :params:
		  sum_w:           sum of the weights of all the neural network, not used
		  examples_number: how many patterns were given in input to this neuron since the last weights update (default: 1)
		  end_epoch:       True if this weight update cames after a whole epoch of learning, False otherwise. 
		                   regularization and momentum are applied only if end_epoch is true.
		
		'''
		
		for i in range(len(self.weights)):
			
			self.weights[i] += self.params.ETA * self.dw[i] / examples_number
			
			self.weights[i] += self.params.ALPHA * self.prev_dw[i]	
			self.cumulative_prev_dw[i] += self.params.ETA * self.dw[i] / examples_number
			
			if end_epoch:
				self.weights[i] -= 2*self.params.LAMBDA * self.weights[i]
				self.prev_dw[i] = self.cumulative_prev_dw[i] + self.params.ALPHA * self.prev_dw[i]
				self.cumulative_prev_dw[i] = 0
		
		self.dw = [0]*len(self.dw)

	def sum_weights(self):
		'''
		  :returns: the squared sum of the weights of this Neuron.
		'''
		if len(self.weights) == 1:
			return 0
		else:		
			s = 0
			for w in self.weights:
				s+=w**2
			
			return s
	
	def getValue(self):
		'''
		  :returns: the output value for this Neuron.
		'''
		return self.out

	def getNet(self):
		'''
		  :returns: the net value for this Neuron.
		'''
		return self.net

	def getNthWeight(self,i):
		'''
		  :returns: the value of the i-th weight of this Neuron, where i is the parameter.
		'''
		return self.weights[i]

	def getBp(self):
		'''
		  :returns: the computed derivative of the Loss function with respect to the net of this neuron.
		'''
		return self.bp
