'''

 this module implements the artificial neural network itself (NeuralNetwork class).

'''

import random
import copy
import numpy as np
import Plotting

import Params
import Neuron
import Statistics

#~ import Monk
#~ import Iris
#~ import Machine
import MLCUP2017

from sklearn.utils import shuffle as parallel_shuffle

class NeuralNetwork:
	'''
	  Implements an artificial neural network, trained using the backpropagation algorithm.
	   The network can be built with any number of input and output units, and with any number of hidden layers, each of them can have a variable size.
	   The network train itself using a train set, and can compute the accuracy and Loss on a validation_set (which is not used for training).
	   After training, the network can predict output for (possibly unseen) patterns.  
	'''
	
	def __init__(self, params=Params):
		'''
		 
		 Creates a NeuralNetwork, given the params that the NeuralNetwork uses during the training (see below).
		  params can be an instance of the ConfigurableParams class, or the global default params, (those defined in the module Params).
		  requested Params are:
		   - ETA:                                            the learning rate.
		   - ETA_DECAY, ETA_DECREASING_PERIOD and ETA_RANGE: parameters that control the ETA decreasing during training.
		   - LAMBDA:                                         the regularizatione coefficient.
		   - ALPHA:                                          the momentum coefficient.
		   - MINBATCH and MINIBATCH_SAMPLE:                  parameters that control the use of minibatch mode.
		   - MAX_EPOCH:                                      number of epochs after which to stop training.
		  other params are ignored
		  
		'''
		self.lista_neuroni = []
		
		self.archi_entranti = []
		self.archi_uscenti = []
		
		self.weights = {}
		
		self.inputDim = 0
		self.outputDim = 0
		self.layersDim = []
		
		self.normalization_factor = 0
		
		self.validation_set = None
		
		self.params = params
		
	def fire_network (self, instance):
		'''
		  fires the network, on a certain input.
		  
		  :params:
		   instance: list of values to give in input to the neural network
		'''
		
		bias = 1
		
		for i in range(len(self.lista_neuroni[0])-1):
			self.lista_neuroni[0][i].fire([instance[i]])
		self.lista_neuroni[0][-1].fire([bias])
	
		l=1
		for layer in self.lista_neuroni[1:-1]:
			for i in range(len(layer)-1):
				indici_entranti = self.archi_entranti[l][i]
				inputs = [self.lista_neuroni[l-1][k].getValue() for k in indici_entranti]
				layer[i].fire(inputs)
			layer[-1].fire([bias])
			l+=1
		
		for i in range(len(self.lista_neuroni[-1])):
			indici_entranti = self.archi_entranti[-1][i]
			inputs = [self.lista_neuroni[-2][k].getValue() for k in indici_entranti]
			self.lista_neuroni[-1][i].fire(inputs)
	
	def setInputDim (self, n):
		'''
		 sets the input dimension of the neural network i.e. the number of input units
		 
		 :params:
		  n: the input dimension to set
		'''
		
		self.inputDim = n
		#~ print ("[NeuralNetwork] input dim: {}".format(self.inputDim))
	
	def setOutputDim(self, n):		
		'''
		 sets the output dimension of the neural network i.e. the number of output units
		 
		 :params:
		  n: the output dimension to set
		'''
		
		self.outputDim = n
		#~ print ("[NeuralNetwork] output dim: {}".format(self.outputDim))
		
	def addLayer (self, n):
		'''
		 adds a layer of neurons, of the given dimension.
		 
		 :params:
		  n: the dimension of the new layer.
		'''
		self.layersDim.append(n)
		
	def addBias(self, lista):
		'''
		 adds a bias Neuron to a layer of neurons, in the last position.
		 
		 :params:
		  lista: a list of Neuron, representing a layer of neurons.
		'''
		bias = Neuron.Neuron( self.params )
		bias.initialize ("id", 1, Params.unit_weigth_initializer )
		lista.append( bias )
		
		
	def buildGraph (self):
		'''
		  builds the network graph, inserting the units and biases for each layer, and then connecting all the units at every level to the ones at the upper level.
		  This will result in a fully-connected graph.
		'''
		self.addUnits()
		
		#~ constant_weigth_initializer = Params.constant_weigth_initializer_initializer ()
		
		#archi entranti nell'input
		self.archi_entranti.append([])
		for i in range(len(self.lista_neuroni[0])):
			self.archi_entranti[0].append([-1])
			self.lista_neuroni[0][i].initialize ("id", 1, Params.unit_weigth_initializer )
		
		#archi uscenti ed entranti dai vari layer, compresi quelli uscenti dall'input e quelli entranti nell'output
		for i in range(len(self.lista_neuroni)-1):
			
			self.addBias(self.lista_neuroni[i])
			self.archi_entranti[i].append([-1])
			
			j = i+1
			
			
			self.archi_uscenti.append([])
			for n in range(len(self.lista_neuroni[i])):
				self.archi_uscenti[i].append ([])
				
			self.archi_entranti.append([])
			for m in range(len(self.lista_neuroni[j])):
				self.archi_entranti[j].append ([])
				
			for n in range(len(self.lista_neuroni[i])):	
				for m in range(len(self.lista_neuroni[j])):
					
					self.archi_entranti[j][m].append(n)
					self.archi_uscenti[i][n].append(m)
			
			for m in range(len(self.lista_neuroni[j])):
				if not j==len(self.lista_neuroni)-1:
					self.lista_neuroni[j][m].initialize ("sigmoid", len(self.archi_entranti[j][m]), Params.random_weigth_initializer )
				else:
					# REGRESSION: linear output unit
					self.lista_neuroni[j][m].initialize ("id", len(self.archi_entranti[j][m]), Params.random_weigth_initializer )
					# CLASSIFICATION: sigmoid output unit
					#~ self.lista_neuroni[j][m].initialize ("sigmoid", len(self.archi_entranti[j][m]), Params.random_weigth_initializer )
				
		#archi uscenti dall'output
		self.archi_uscenti.append([])
		for i in range(len(self.lista_neuroni[-1])):
			self.archi_uscenti[-1].append([-1])
			
	
	
	def addUnits(self):
		'''
		 adds the units specified calling the `add_layer` function.
		'''
		
		lista = []
		for i in range(self.inputDim):
			lista.append(Neuron.Neuron(self.params))
		
		self.lista_neuroni.append(lista)
		
		for dim in self.layersDim:
			lista = []
			for i in range(dim):
				lista.append(Neuron.Neuron(self.params))
			self.lista_neuroni.append(lista)
			
		lista = []	
		for i in range(self.outputDim):
			lista.append(Neuron.Neuron(self.params))
		self.lista_neuroni.append(lista)
	
	def set_train (self, train_set, train_labels):
		'''
		 specify the instances to use as train set, and the corresponding labels.
		 
		 :params:
		  train_set:    sequence of instance to use as train set
		  train_labels: expected outputs for the instance of train_set. train_labels[i] is the expected output for the pattern train_set[i]
		'''
		
		self.train_set = copy.deepcopy (train_set)
		self.train_labels = copy.deepcopy (train_labels)
		self.setInputDim (len(train_set[0]))
		self.setOutputDim (len(train_labels[0]))

	def set_validation (self, validation_set, validation_labels):
		'''
		 specify the instances to use as validation set, and the corresponding labels.
		 
		 :params:
		  validation_set:    sequence of instance to use as validation set
		  validation_labels: expected outputs for the instance of validation_set. validation_labels[i] is the expected output for the pattern validation_set[i]
		'''
		
		self.validation_set = copy.deepcopy (validation_set)
		self.validation_labels = copy.deepcopy (validation_labels)
	
	def learn (self):
		'''
		  learns the weights of connections between neurons that minimize the Mean Squared Error Loss function
		  between predicted output for the train set and trai labels.
		  
		  Memorizes the MSE Loss on the training set for each epoch, and the MSE Loss and the Accuracy on the validation set for each epoch.
		'''
		
		self.buildGraph()
		
		self.train_losses = []
		self.validation_losses = None if self.validation_set is None else []
		self.validation_accuracies = None if self.validation_set is None else []
		epoch = 0
		
		xtrain = self.train_set
		ytrain = self.train_labels
		
		#~ print ("[DEBUG]len(xtrain): {}".format(len(xtrain)))
		#~ print ("[DEBUG]first 10 elements of xtrain: {}".format(xtrain[:10]))
		
		while (epoch < self.params.MAX_EPOCH):
			
			if self.params.ETA_DECAY:
				if epoch>=self.params.ETA_DECREASING_PERIOD * self.params.MAX_EPOCH:
					self.params.ETA = self.params.ETA_RANGE[0]
				else:
					self.params.ETA = self.params.ETA_RANGE[1] - (self.params.ETA_RANGE[1] - self.params.ETA_RANGE[0]) * ( epoch / (self.params.MAX_EPOCH * self.params.ETA_DECREASING_PERIOD) )
				#~ print ("[DEBUG] epoch {} eta {}".format(epoch, self.params.ETA))
			
			if self.params.MINIBATCH:
				xtrain, ytrain = parallel_shuffle (xtrain, ytrain)
				#~ print ("[DEBUG]after shuffling xtrain")
				#~ print ("[DEBUG]first 10 elements of xtrain: {}".format(xtrain[:10]))
		
			
			#~ print ("epoch {}".format(epoch))		
			loss = Statistics.MSELoss()
			self.normalization_factor = self.sum_weights()
			
			for i,x,y in zip (range(1,len(xtrain)+1), xtrain, ytrain):
				self.fire_network(x)
				self.update_backpropagation(y)
				#~ print ("[DEBUG] after feeding with example {}".format(x))
				#~ print ("[DEBUG] out is {}".format([neuron.getValue() for neuron in self.lista_neuroni[-1]]))
				#~ print ("[DEBUG]   y is {}".format(y))
				#~ input()
				loss.update([neuron.getValue() for neuron in self.lista_neuroni[-1]], y)
				
				if self.params.MINIBATCH and i%self.params.MINIBATCH_SAMPLE == 0:
					#~ print ("epoch {} num samples {}: weights updating".format(epoch, i))
					#~ print ("end epoch: {}".format(i==len(xtrain)))
					self.update_weights (self.params.MINIBATCH_SAMPLE, end_epoch=(i==len(xtrain)) )
				
				#~ print ("after feeding")
				#~ self.dump()
				#~ input()
			
			if self.params.MINIBATCH:
				#~ print ("MINIBATCH: updating the weights at the end of the {} epoch, for the remaininng {} patterns".format(epoch, len(xtrain)%self.params.MINIBATCH_SAMPLE ))
				self.update_weights( len(xtrain)%self.params.MINIBATCH_SAMPLE )
			else:
				#~ print ("    BATCH: updating the weights at the end of the {} epoch, for the remaininng {} patterns".format(epoch, len(xtrain)))
				self.update_weights( len(xtrain) )
			
			#~ print ("after weight update")
			#~ self.dump()
			#~ input()
			
			self.train_losses.append(loss.get())
			
			if self.validation_set is not None:
				loss = Statistics.MSELoss()
				#~ accuracy = Statistics.MulticlassificationAccuracy ()
				#~ accuracy = Statistics.Accuracy ()
				#~ accuracy = Statistics.MSELoss ()
				accuracy = Statistics.MEELoss ()
				for x,y in zip (self.validation_set, self.validation_labels):
					self.fire_network(x)
					loss.update([neuron.getValue() for neuron in self.lista_neuroni[-1]], y)
					accuracy.update ([neuron.getValue() for neuron in self.lista_neuroni[-1]], y)
				self.validation_losses.append (loss.get())
				self.validation_accuracies.append (accuracy.get())
								
			epoch += 1
			
			
				
	def update_weights(self, examples_number, end_epoch=True):
		'''
		  update the connection weights after the network was feeded with certain patterns.
		  
		  :params:
		   examples_number: the number of examples that were given in input to the network.
		   end_epoch:       wheter the weight update is performed at the end of a learnng epoch or not.
		'''
		if not examples_number == 0:
			for l in self.lista_neuroni:
				for n in l:
					n.update_weights(self.normalization_factor, examples_number=examples_number,  end_epoch=end_epoch)
		
	#TODO: try not considering the biases
	def sum_weights (self):
		'''
		  return the squared sum of all the weights of all the connections in the neural network
		'''
		s=0
		
		for layer in self.lista_neuroni:
			for n in layer:
				s+=n.sum_weights()
				
		return s
				
	
	def update_backpropagation(self, d):
		'''
		  updates the gradient of the loss function for every Neuron of the network, after the network was feeded with a pattern p.
		  
		  :params:
		   d: expected output for the feeded pattern (p). 
		'''
		
		output_neurons = self.lista_neuroni[-1]
		
		i=0
		for n in output_neurons:
			n.update_backpropagation_output(d[i])
			i+=1
		
		for l in range(len(self.lista_neuroni)-2, 0, -1):
			
			for i in range(len(self.lista_neuroni[l])):
				
				neuroni_uscenti = []
				pesi_uscenti = []
				
				for n in self.archi_uscenti[l][i]:
					neurone = self.lista_neuroni[l+1][n]
					archi_entranti = self.archi_entranti[l+1][n]
					indice_mio_peso = archi_entranti.index(i)
					mio_peso = neurone.getNthWeight(indice_mio_peso)
					
					neuroni_uscenti.append(neurone)
					pesi_uscenti.append(mio_peso)
					
				self.lista_neuroni[l][i].update_backpropagation_hidden(neuroni_uscenti, pesi_uscenti)
	
	def getOutputs (self):
		'''
		  :returns: a list with the output values of all the output units of the network
		'''
		
		ret = []
		for i in range(len(self.lista_neuroni)-self.outputDim, len(self.lista_neuroni)):
			ret.append(self.lista_neuroni[i].getValue())
			
		return ret
	
	def predict (self, instance):
		'''
		  prdicts the output value for an instance.
		  
		  :params:
		   instance: list of values to give in input to the neural network. 
		  :returns: the predicted output for `instance`.
		'''
		
		self.fire_network(instance)
		return [n.getValue() for n in self.lista_neuroni[-1]]
		
	def dump (self):
		'''
		  prints on standard output the internal representation of the network.
		  useful for debugging.
		'''
		print("*** INPUT LAYER ***")
		for i in range (len(self.lista_neuroni[0])):
			print("neurone", i, ":", self.lista_neuroni[0][i])
			print("output: ",self.lista_neuroni[0][i].getValue())
			print("archi uscenti:")
			for j in self.archi_uscenti[0][i]:
				print("->",j)
			print()
		
		for layer in range (1,len (self.lista_neuroni)-1 ):
			print("*** LAYER",layer,"***")
			for i in range (len(self.lista_neuroni[layer])):
				print("archi entranti:")
				for j in self.archi_entranti[layer][i]:
					print(j,"w=",self.lista_neuroni[layer][i].weights[j],"->")
					print(j,"-----",self.lista_neuroni[layer][i].dw[j],"->")
				print("neurone", i, ":", self.lista_neuroni[layer][i])
				print("output: ",self.lista_neuroni[layer][i].getValue())
				print("archi uscenti:")
				for j in self.archi_uscenti[layer][i]:
					print("->",j)
				print()
		
		print("*** OUTPUT LAYER ***")
		for i in range (len(self.lista_neuroni[-1])):
			print("archi entranti:")
			for j in self.archi_entranti[-1][i]:
				print(j,"w=",self.lista_neuroni[-1][i].weights[j],"->")
				print(j,"-----",self.lista_neuroni[-1][i].dw[j],"->")
			print("neurone", i, ":", self.lista_neuroni[-1][i])
			print("output: ",self.lista_neuroni[-1][i].getValue())
			print()

#unit tests	
if __name__=="__main__":
	
	#MONKs
	#~ train_sets   = [ Monk.monk1_training_set, Monk.monk2_training_set, Monk.monk3_training_set ]
	#~ train_labels = [ Monk.monk1_training_labels, Monk.monk2_training_labels, Monk.monk3_training_labels ]
	#~ test_sets    = [ Monk.monk1_test_set, Monk.monk2_test_set, Monk.monk3_test_set ]
	#~ test_labels  = [ Monk.monk1_test_labels, Monk.monk2_test_labels, Monk.monk3_test_labels ]
	
	#only one MONK
	#~ train_sets   = [ Monk.monk3_training_set ]
	#~ train_labels = [ Monk.monk3_training_labels ]
	#~ test_sets    = [ Monk.monk3_test_set ]
	#~ test_labels  = [ Monk.monk3_test_labels ]
	
	# XOR
	#~ train_sets = [ [[0, 0], [0,1], [1,0], [1,1]] ]
	#~ train_labels = [ [[0],  [1],   [1],  [0] ] ]
	#~ test_sets = [ [[1,1] ] ]
	#~ test_labels = [ [[0] ] ]
	
	# Iris
	#~ train_sets = [ Iris.iris_train_set[int(len(Iris.iris_train_set)/8):] ] 
	#~ train_labels = [ Iris.iris_train_labels[int(len(Iris.iris_train_set)/8):] ] 
	#~ test_sets = [ Iris.iris_train_set[:int(len(Iris.iris_train_set)/8)] ] 
	#~ test_labels = [ Iris.iris_train_labels[:int(len(Iris.iris_train_set)/8)] ] 

	# Machine
	#~ train_sets = [ Machine.machine_train_set ] 
	#~ train_labels = [ Machine.machine_train_labels ] 
	#~ test_sets = [ Machine.machine_test_set ] 
	#~ test_labels = [ Machine.machine_test_labels ] 

	# MLCUP2017
	test_len = int(len(MLCUP2017.cup_train_set)/10)
	train_sets = [ MLCUP2017.cup_train_set[test_len:] ] 
	train_labels = [ MLCUP2017.cup_train_labels[test_len:] ]
	test_sets = [ MLCUP2017.cup_train_set[:test_len] ] 
	test_labels = [ MLCUP2017.cup_train_labels[:test_len] ]

	for i, train_s, train_l, test_s, test_l in zip ( range(1,len(train_sets)+1), train_sets, train_labels, test_sets, test_labels ):
		
		print ("--- TEST {} ---".format(i))
		
		myNN = NeuralNetwork()
		myNN.addLayer(6)
		#~ myNN.addLayer(4)
		#~ myNN.addLayer(2)
		
		myNN.set_train (train_s, train_l)
		myNN.set_validation (test_s, test_l)
		
		myNN.learn()
		
		Plotting.plot_loss_accuracy_per_epoch (myNN)
		
		#~ a = Statistics.MulticlassificationAccuracy ()
		#~ a = Statistics.Accuracy ()
		#~ a = Statistics.MSELoss ()
		a = Statistics.MEELoss ()
		for x,y in zip(train_s, train_l):
			o = myNN.predict (x)
			a.update (o, y)
			
		print ("Accuracy on train set {}".format ( a.get() ))

		#~ a = Statistics.MulticlassificationAccuracy ()
		#~ a = Statistics.Accuracy ()
		#~ a = Statistics.MSELoss ()
		a = Statistics.MEELoss ()
		for x,y in zip(test_s, test_l):
			o = myNN.predict (x)
			a.update (o, y)
			
		print ("Accuracy on test set {}".format ( a.get() ))

