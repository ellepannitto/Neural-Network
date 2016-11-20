import math

import Params

class Neuron:
	def __init__(self):
		pass
		
	def initialize (self, fun_name, weights, weights_inizialization):
		self.activation_function = Params.ActivationFunctions[fun_name]
		self.activation_function_derivative = Params.ActivationFunctions[fun_name+"_der"]
		self.weights = [weights_inizialization() for i in range(weights)]
		#~ self.weights = [ for w in self.weights]
		
		self.dw = [0]*weights
		self.prev_dw = [0]*weights
		
	def compute_net (self):
		
		self.net = 0.0
		for i in range(len(self.inputs)):
			self.net+=self.weights[i]*self.inputs[i]

	def fire (self, inputs):
		
		self.inputs = inputs
		
		self.compute_net()
		
		self.out = self.activation_function(self.net)
		return self.out

	def update_backpropagation_output(self, d):
		
		self.bp = ( self.out - d ) * self.activation_function_derivative(self.net) 
		
		
		for i in range(len(self.weights)):
			self.dw[i] += self.bp * self.inputs[i]
		
		#~ print self.dw
		#~ raw_input()
	
	def update_backpropagation_hidden(self, neuroni, pesi):
		
		s = 0
		
		for i in range(len(neuroni)):
			s+= neuroni[i].getBp () * pesi[i]
		
		#~ print s
		#~ raw_input()	
		
		self.bp = s * self.activation_function_derivative (self.net)
		
		#~ print self.bp
		#~ raw_input()
		
		for i in range(len(self.weights)):
			self.dw[i] += self.bp * self.inputs[i]
		
		#~ print self.dw
		#~ raw_input()
			
	def update_weights(self, sum_w):
		
		
		for i in range(len(self.weights)):
			
			self.weights[i] -= 2*Params.LAMBDA * self.weights[i]
			self.weights[i] -= Params.ETA * self.dw[i] 
			self.weights[i] -= Params.ALPHA * self.prev_dw[i] 
			
			self.prev_dw[i] = Params.ETA*self.dw[i]
		
		
		self.dw = [0]*len(self.dw)

	def sum_weights(self):
		#TODO: fix neuron inputs
		if len(self.weights) == 1:
			return 0
		else:		
			s = 0
			for w in self.weights:
				#~ print w
				#~ raw_input()
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
