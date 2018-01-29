import math

import Params

class Neuron:
	def __init__(self, params=Params):
		self.params = params
		
	def initialize (self, fun_name, weights, weights_inizialization):
		self.activation_function = Params.ActivationFunctions[fun_name]
		self.activation_function_derivative = Params.ActivationFunctions[fun_name+"_der"]
		self.weights = [weights_inizialization() for i in range(weights)]
		#~ self.weights = [ for w in self.weights]
		
		self.dw = [0]*weights
		self.prev_dw = [0]*weights
		self.cumulative_prev_dw = [0]*weights
		
	def compute_net (self):
		
		self.net = 0.0
		for i in range(len(self.inputs)):
			self.net+=self.weights[i]*self.inputs[i]

	def fire (self, inputs):
		
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
		
		s = 0
		
		for i in range(len(neuroni)):
			s+= neuroni[i].getBp () * pesi[i]
		
		self.bp = s * self.activation_function_derivative (self.net)
			
		for i in range(len(self.weights)):
			self.dw[i] += self.bp * self.inputs[i]
			
	def update_weights(self, sum_w, examples_number=1, end_epoch=True):
		
		for i in range(len(self.weights)):
			
			self.weights[i] -= 2*self.params.LAMBDA * self.weights[i]
			self.weights[i] += self.params.ETA * self.dw[i] / examples_number
			
			self.weights[i] += self.params.ALPHA * self.prev_dw[i]	
			self.cumulative_prev_dw[i] += self.params.ETA * self.dw[i] / examples_number
			
			if end_epoch:
				self.prev_dw[i] = self.cumulative_prev_dw[i] + self.params.ALPHA * self.prev_dw[i]
				self.cumulative_prev_dw[i] = 0
		
		self.dw = [0]*len(self.dw)

	def sum_weights(self):
		#TODO: fix neuron inputs
		if len(self.weights) == 1:
			return 0
		else:		
			s = 0
			for w in self.weights:
				#~ print (w)
				s+=w**2
			
			#~ print s
			#~ raw_input()
			return s
	
	def getValue(self):
		return self.out

	def getNet(self):
		return self.net

	def getNthWeight(self,i):
		return self.weights[i]

	def getBp(self):
		return self.bp
