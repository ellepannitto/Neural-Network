import time
import collections
import random
import numpy as np
import matplotlib.pyplot as plt


import Params
import Neuron
import Statistics
import Monk

class NeuralNetwork:
	
	def __init__(self):
		self.lista_neuroni = []
		
		self.archi_entranti = []
		self.archi_uscenti = []
		
		self.weights = {}
		
		self.inputDim = 0
		self.outputDim = 0
		self.layersDim = []
		
		self.normalization_factor = 0
		
		self.validation_set = None
		
	def fire_network (self, instance):
		
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
		
		self.inputDim = n
	
	def setOutputDim(self, n):		
	
		self.outputDim = n
		
	def addLayer (self, n):
		self.layersDim.append(n)
		
	def addBias(self, lista):
		bias = Neuron.Neuron()
		bias.initialize ("id", 1, Params.unit_weigth_initializer )
		lista.append( bias )
		
		
	def buildGraph (self):
		
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
				#~ self.lista_neuroni[j][m].initialize ("sigmoid", len(self.archi_entranti[j][m]), constant_weigth_initializer )
				self.lista_neuroni[j][m].initialize ("sigmoid", len(self.archi_entranti[j][m]), Params.random_weigth_initializer )
			
		#archi uscenti dall'output
		self.archi_uscenti.append([])
		for i in range(len(self.lista_neuroni[-1])):
			self.archi_uscenti[-1].append([-1])
			
	
	
	def addUnits(self):
		
		lista = []
		for i in range(self.inputDim):
			lista.append(Neuron.Neuron())
		
		self.lista_neuroni.append(lista)
		
		for dim in self.layersDim:
			lista = []
			for i in range(dim):
				lista.append(Neuron.Neuron())
			self.lista_neuroni.append(lista)
			
		lista = []	
		for i in range(self.outputDim):
			lista.append(Neuron.Neuron())
		self.lista_neuroni.append(lista)
	
	def set_train (self, train_set, train_labels):
		self.train_set = train_set
		self.train_labels = train_labels

	def set_validation (self, validation_set, validation_labels):
		self.validation_set = validation_set
		self.validation_labels = validation_labels
	
	def learn (self):
	
		self.buildGraph()
		
		self.train_losses = []
		self.validation_losses = None if self.validation_set is None else []
		epoch = 0
		
		xtrain = self.train_set
		ytrain = self.train_labels
		
		while (epoch < Params.MAX_EPOCH):
					
			loss = Statistics.MSELoss()
			self.normalization_factor = self.sum_weights()
			
			for x,y in zip (xtrain, ytrain):
				#~ print ("feeding network with example {}".format(x))
				self.fire_network(x)
				self.update_backpropagation(y)
				loss.update([neuron.getValue() for neuron in self.lista_neuroni[-1]], y)
				
				#~ print ("after feeding")
				#~ self.dump()
				#~ input()
			
			self.update_weights( len(xtrain) )
			
			#~ print ("after weight update")
			#~ self.dump()
			#~ input()
			
			self.train_losses.append(loss.loss/len(xtrain))
			
			if self.validation_set is not None:
				loss = Statistics.MSELoss()
				for x,y in zip (self.validation_set, self.validation_labels):
					self.fire_network(x)
					loss.update([neuron.getValue() for neuron in self.lista_neuroni[-1]], y)
				self.validation_losses.append (loss.loss/len(self.validation_set))
			
			epoch += 1
			
			
				
	def update_weights(self, examples_number):
		
		for l in self.lista_neuroni:
			for n in l: 
				n.update_weights(self.normalization_factor, examples_number=1)
	
	#TODO: try not considering the biases
	def sum_weights (self):
		s=0
		
		for layer in self.lista_neuroni:
			for n in layer:
				s+=n.sum_weights()
				
		return s
				
	
	def update_backpropagation(self, d):
		
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
		ret = []
		for i in range(len(self.lista_neuroni)-self.outputDim, len(self.lista_neuroni)):
			ret.append(self.lista_neuroni[i].getValue())
			
		return ret
	
	def predict (self, instance):
		
		self.fire_network(instance)		
		return [n.getValue() for n in self.lista_neuroni[-1]]	
	
	def dump (self):
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
	
import OneHotEncoder

if __name__=="__main__":
	
	
	#MONKs
	#~ train_sets   = [ Monk.monk1_training_set, Monk.monk2_training_set, Monk.monk3_training_set ]
	#~ train_labels = [ Monk.monk1_training_labels, Monk.monk2_training_labels, Monk.monk3_training_labels ]
	#~ test_sets    = [ Monk.monk1_test_set, Monk.monk2_test_set, Monk.monk3_test_set ]
	#~ test_labels  = [ Monk.monk1_test_labels, Monk.monk2_test_labels, Monk.monk3_test_labels ]
	
	#only one MONK
	train_sets   = [ Monk.monk3_training_set ]
	train_labels = [ Monk.monk3_training_labels ]
	test_sets    = [ Monk.monk3_test_set ]
	test_labels  = [ Monk.monk3_test_labels ]
	
	# XOR
	#~ train_sets = [ [[0, 0], [0,1], [1,0], [1,1]] ]
	#~ train_labels = [ [[0],  [1],   [1],  [0] ] ]
	#~ test_sets = [ [[1,1] ] ]
	#~ test_labels = [ [[0] ] ]
	
	for i, train_s, train_l, test_s, test_l in zip ( range(1,len(train_sets)+1), train_sets, train_labels, test_sets, test_labels ):
		
		#~ print ("--- TEST {} ---".format(i))
		print ("--- MONK {} ---".format(i))
		
		#~ enc = OneHotEncoder (sparse=False)
		#~ encoded_train_s = enc.fit_transform (train_s)
		#~ encoded_test_s = enc.fit_transform (test_s)
		
		encoded_train_s = OneHotEncoder.encode_int_matrix (train_s)
		encoded_test_s = OneHotEncoder.encode_int_matrix (test_s)
		
		myNN = NeuralNetwork()	
		myNN.setInputDim (len(encoded_train_s[0]))
		print ("input dim: {}".format(len(encoded_train_s[0])))
		myNN.setOutputDim (len(train_l[0]))
		myNN.addLayer(6)
		
		myNN.set_train (encoded_train_s, train_l)
		myNN.set_validation (encoded_test_s, test_l)
		
		myNN.learn()
		
		plt.plot(list(range(len(myNN.train_losses))), myNN.train_losses, 'r--', label='train error')
		plt.plot(list(range(len(myNN.validation_losses))), myNN.validation_losses, 'b-', label='validation error')
		plt.legend()
		plt.ylabel('Loss')
		plt.xlabel('epoch')
		axes = plt.gca()
		axes.set_xlim([0,Params.MAX_EPOCH])
		axes.set_ylim([0,0.2])
		plt.show()
		
		l=[]
		for x,y in zip(encoded_train_s, train_l):
			l.append(1 if ((myNN.predict(x)[0]>0.5 and y[0]>0.5) or (myNN.predict(x)[0]<=0.5 and y[0]<=0.5)) else 0)
			
		print ("Accuracy on train set {}".format ( (sum(l)*1.0/len(l))) )

		l=[]
		for x,y in zip(encoded_test_s, test_l):
			l.append(1 if ((myNN.predict(x)[0]>0.5 and y[0]>0.5) or (myNN.predict(x)[0]<=0.5 and y[0]<=0.5)) else 0)
			
		print ("Accuracy on test set {}".format ( (sum(l)*1.0/len(l))) )

