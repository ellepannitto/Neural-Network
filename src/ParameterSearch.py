
import numpy as np
import random
from multiprocessing.pool import Pool
import Params
from CrossValidation import KFoldCrossValidation
import itertools

import Iris
import Monk

coarse_choices_dict = {
				"ALPHA" : np.arange (0,1.1,0.1),
				"ETA" : np.arange (0.1,1.1,0.1),
				"LAMBDA" : np.arange (0.001,0.01,0.001),
				"ETA_DECAY" : [True],
				"MINIBATCH" : [True, False],
				"LAYERS_SIZE" : [(6,)],
			   }

fine_choices_dict = {
				"ALPHA" : [0.8],
				"ETA" : [0.7],
				"LAMBDA" : [0.004],
				"ETA_DECAY" : [True],
				"MINIBATCH" : [True],
				"LAYERS_SIZE" : [(6,),(4,)],
			   }

def normalize_configuration ( conf ):
	if conf["ETA_DECAY"]:
		eta_min = conf["ETA"]/2
		eta_max = min ( 1, conf["ETA"]*3/2 ) 
		conf["ETA_RANGE"] = [ eta_min, eta_max ]
		conf["ETA_DECREASING_PERIOD"]=Params.ETA_DECREASING_PERIOD
	if conf["MINIBATCH"]:
		conf["MINIBATCH_SAMPLE"]=Params.MINIBATCH_SAMPLE
	return conf



class RandomSearch:
	
	def __init__ (self, train_set, train_labels, configurations_number=100, threads_number=4, model_name="random_search_test" ):
		self.train_set = train_set
		self.train_labels = train_labels
		self.configurations_number = configurations_number
		self.threads_number = threads_number
		self.model_name = model_name
		self.results = {}
	
	def test_configuration (self, conf_dict):
		params = Params.ConfigurableParams (conf_dict)
		print ("[RandomSearch] Starting CrossValidation with id {}".format(params.ID))
		name = self.model_name + "_" + str(params.ID)
		cv = KFoldCrossValidation ( self.train_set, self.train_labels, params=params, K=Params.NUM_FOLDS, model_name = name )
		cv.perform ()
		cv.dump ()
		self.results[name] = cv.mean_accuracy
		print ("[RandomSearch] Ended CrossValidation with id {}".format(params.ID))
		
		
	def perform ( self, choices_dict ):
		print ("[RandomSearch] Initializing...")
		
		configurations = []
		for i in range (self.configurations_number):
			conf = { "ID": i, "MAX_EPOCH": Params.MAX_EPOCH }
			for key, values in choices_dict.items():
				conf[key] = random.choice (values)
			normalize_configuration (conf)
			configurations.append (conf)
		print ("[RandomSearch] created {} random configurations".format(self.configurations_number))
		
		pool = Pool ( self.threads_number )
		pool.map (test_configuration, configurations)
		
		print ("[RandomSearch] Finished.")
		
	def dump_results ( self ):
		sorted_results = sorted ( self.results.items(), key=lambda x: x[1], reverse=True )
		with open("../dumps/" + self.model_name + "_results", "w") as fout:
			fout.write ("RESULTS SORTED BY ACCURACY\n\n")
			fout.write ( "\n".join( [name + "\t" + str(acc) for name, acc in sorted_results] ) )

class GridSearch:
	def __init__ (self, train_set, train_labels, threads_number=4, model_name="grid_search_test" ):
		self.train_set = train_set
		self.train_labels = train_labels
		self.threads_number = threads_number
		self.model_name = model_name
		self.results = {}
	
	def test_configuration (self, conf_dict):
		params = Params.ConfigurableParams (conf_dict)
		print ("[GridSearch] Starting CrossValidation with id {}".format(params.ID))
		name = self.model_name + "_" + str(params.ID)
		cv = KFoldCrossValidation ( self.train_set, self.train_labels, params=params, K=Params.NUM_FOLDS, model_name = name )
		cv.perform ()
		cv.dump ()
		self.results[name] = cv.mean_accuracy
		print ("{}\t{}".format(name, cv.mean_accuracy))
		print ("[GridSearch] Ended CrossValidation with id {}".format(params.ID))
	
	def perform ( self, choices_dict ):
		print ("[GridSearch] Initializing...")
		
		print ("MODEL\tACCURACY")
		
		param_names = []
		param_choices = []
		for param, values in choices_dict.items():
			param_names.append(param)
			param_choices.append(values)
		
		#~ print ("[GridSearch] parameters: {}".format(param_names))
		#~ print ("[GridSearch] grid: {}".format(param_choices))
		
		combinations = itertools.product (*param_choices)
		configurations = []
		
		for i, comb in enumerate (combinations):
			conf = { "ID": i, "MAX_EPOCH": Params.MAX_EPOCH }
			#~ print ("combination {}".format(comb))
			for key, value in zip(param_names, comb):
				conf[key] = value
				#~ print ("conf[{}] = {}".format(key, value))
			normalize_configuration (conf)
			configurations.append (conf)
			#~ print ("[GridSearch] created conf {}".format(conf))
		print ("[GridSearch] created all the {} configurations".format(len(configurations)))
		
		pool = Pool ( self.threads_number )
		pool.map (self.test_configuration, configurations)
		
		print ("[GridSearch] Finished.")
	
	def dump_results ( self ):
		sorted_results = sorted ( self.results.items(), key=lambda x: x[1], reverse=True )
		#~ print ("RESULTS: {}".format(sorted_results))
		with open("../dumps/" + self.model_name + "_results", "w") as fout:
			fout.write ("RESULTS SORTED BY ACCURACY\n\n")
			fout.write ( "\n".join( [name + "\t" + str(acc) for name, acc in sorted_results] ) )
	
	
if __name__=="__main__":
	
	# Iris
	#~ train_s = Iris.iris_train_set[int(len(Iris.iris_train_set)/8):] 
	#~ train_l = Iris.iris_train_labels[int(len(Iris.iris_train_set)/8):]
	#~ test_s =  Iris.iris_train_set[:int(len(Iris.iris_train_set)/8)] 
	#~ test_l =  Iris.iris_train_labels[:int(len(Iris.iris_train_set)/8)] 
	
	# Monk 3
	train_s = Monk.monk3_training_set
	train_l = Monk.monk3_training_labels
	test_s  = Monk.monk3_test_set
	test_l  = Monk.monk3_test_labels
	
	#random parameter search
	#~ rs = RandomSearch ( train_s, train_l, configurations_number=20, threads_number=4, model_name="Monk3" )
	#~ rs.perform (coarse_choices_dict)
	#~ rs.dump_results ()
	
	#grid search
	gs = GridSearch ( train_s, train_l, threads_number=4, model_name="Monk3" )
	gs.perform (fine_choices_dict)
	#~ gs.dump_results ()
	
