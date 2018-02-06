'''
  this module trains a NeuralNetwork with a train set, then predicts the
  output for a blind test sets and de-normalize it, using the inverse z-score
  transformation that was applied to the known output labels.
'''

import NeuralNetwork
import os
import Params
import MLCUP2017
import Plotting
import Statistics

from model_selection import parse_result_file

'''
  prediction on the test set
'''
if __name__ == "__main__":
	
	model_name = "MLCUP2017_567_ludo"
		
	train_s = MLCUP2017.cup_train_set + MLCUP2017.cup_validation_set
	train_l = MLCUP2017.cup_train_labels + MLCUP2017.cup_validation_labels
	test_s = MLCUP2017.cup_test_set
	
	mues = MLCUP2017.mean_per_attribute_dataset[len(train_s[0])+1:]
	sigmas = MLCUP2017.std_per_attribute_dataset[len(train_s[0])+1:]
	
	print ("mues {}".format(mues))
	print ("sigmas {}".format(sigmas))		
	
	print ("MODEL {}".format(model_name))
	
	params_dict = parse_result_file ("../dumps/"+model_name)
	params = Params.ConfigurableParams ( params_dict )
	params.MAX_EPOCH = int (params_dict["mean epochs"])
			
	myNN = NeuralNetwork.NeuralNetwork(params)
	
	for size in params.LAYERS_SIZE:
		myNN.addLayer(size)
	
	myNN.set_train (train_s, train_l)
	
	myNN.learn()
	
	Plotting.plot_loss_accuracy_per_epoch (myNN)
	
	a = Statistics.MEELoss ()
	for x,y in zip(train_s, train_l):
		o = myNN.predict (x)
		a.update (o, y)
		
	print ("Accuracy on train set {}".format ( a.get() ))

	with open("../results/"+model_name+"_predicted_test", "w") as fout:

		for i,x in zip(range(1,len(test_s)+1), test_s):
			o = myNN.predict (x)
			
			o_not_normalized = []
			y_not_normalized = []
			
			#~ print ("for this pattern: gold {}".format(y))
			#~ print ("for this pattern: predicted {}".format(o))
			
			for out, mu, sigma in zip(o,mues,sigmas):
				o_not_normalized.append(out*sigma + mu)
				
				#~ print ("gold {} predicted {}".format(gold, out))
				#~ print ("gold not normalized {} predicted not normalized {}".format(y_not_normalized[-1], o_not_normalized[-1]))
				
			
			fout.write (str(i)+","+",".join([ str(el) for el in o_not_normalized ]))
			fout.write ("\n")
					
