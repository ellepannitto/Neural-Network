import time
import collections
import random
import numpy as np
import matplotlib.pyplot as plt


import Params
import Neuron
import Loss
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
		
		constant_weigth_initializer = Params.constant_weigth_initializer_initializer ()
		
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
	
		
	def learn (self, xtrain, ytrain):
		losses = []
		
		
		epoch = 0
		while (epoch < Params.MAX_EPOCH):
			
			loss = Loss.Loss()
			self.normalization_factor = self.sum_weights()
			
			n=0
			for x in xtrain:
				self.fire_network(x)
				self.update_backpropagation(ytrain[n])
				loss.update([neuron.getValue() for neuron in self.lista_neuroni[-1]], ytrain[n])
				#~ print [k.getValue() for k in self.lista_neuroni[-1]]
				n+=1
			
			self.update_weights()
			
			losses.append(loss.loss)
			#~ print "------------"
			#~ time.sleep(0.5)
			#~ print [vars(j) for j in self.lista_neuroni]
			#~ self.dump ()
			#~ raw_input()
			epoch += 1
			
			
		plt.plot(range(len(losses)), losses)
		plt.show()
				
	def update_weights(self):
		
		for l in self.lista_neuroni:
			for n in l: 
				n.update_weights(self.normalization_factor)			
	
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
		print "*** INPUT LAYER ***"
		for i in range (len(self.lista_neuroni[0])):
			print "neurone", i, ":", self.lista_neuroni[0][i]
			print "output: ",self.lista_neuroni[0][i].getValue()
			print "archi uscenti:"
			for j in self.archi_uscenti[0][i]:
				print "->",j
			print
		
		for layer in range (1,len (self.lista_neuroni)-1 ):
			print "*** LAYER",layer,"***"
			for i in range (len(self.lista_neuroni[layer])):
				print "archi entranti:"
				for j in self.archi_entranti[layer][i]:
					print j,"w=",self.lista_neuroni[layer][i].weights[j],"->"
					print j,"-----",self.lista_neuroni[layer][i].old_dw[j],"->"
				print "neurone", i, ":", self.lista_neuroni[layer][i]
				print "output: ",self.lista_neuroni[layer][i].getValue()
				print "archi uscenti:"
				for j in self.archi_uscenti[layer][i]:
					print "->",j
				print
		
		print "*** OUTPUT LAYER ***"
		for i in range (len(self.lista_neuroni[-1])):
			print "archi entranti:"
			for j in self.archi_entranti[-1][i]:
				print j,"w=",self.lista_neuroni[-1][i].weights[j],"->"
				print j,"-----",self.lista_neuroni[-1][i].old_dw[j],"->"
			print "neurone", i, ":", self.lista_neuroni[-1][i]
			print "output: ",self.lista_neuroni[-1][i].getValue()
			print
		
if __name__=="__main__":
	
	myNN = NeuralNetwork()
	
	dataset = Monk.training_set

	testset = Monk.test_set
	
	labels = Monk.training_labels
	
	testlabels = Monk.test_labels
		
	myNN.setInputDim (len(dataset[0]))
	myNN.setOutputDim (len(labels[0]))
	
	myNN.addLayer(8)
	
	myNN.buildGraph()
	
	
	myNN.learn(dataset, labels)
	#~ myNN.learn(dataset[:100], labels[:100])

	i=0
	l=[]
	for x in testset:
		l.append(1 if ((myNN.predict(x)[0]>0.5 and testlabels[i][0]>0.5) or (myNN.predict(x)[0]<=0.5 and testlabels[i][0]<=0.5)) else 0)
		i+=1
	
	print sum(l)*1.0/len(l)


	i=0
	l=[]
	for x in dataset:
		l.append(1 if ((myNN.predict(x)[0]>0.5 and labels[i][0]>0.5) or (myNN.predict(x)[0]<=0.5 and labels[i][0]<=0.5)) else 0)
		i+=1
	print sum(l)*1.0/len(l)
	
