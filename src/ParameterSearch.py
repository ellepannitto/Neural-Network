'''
  this module contains classes to find the best hyperparameters using a random search or a grid search on the hyperparameters space. 
'''

import numpy as np
import random
from multiprocessing.pool import Pool
import Params
from CrossValidation import KFoldCrossValidation
import itertools

#~ import Iris
#~ import Monk
#~ import Machine
import MLCUP2017

coarse_choices_dict = {
				"ALPHA" : np.arange (0.01,1.01,0.02),
				"ETA" : np.arange (0.01,1.01,0.02),
				"LAMBDA" : np.arange (0.001,0.501,0.001),
				"ETA_DECAY" : [True, False],
				"MINIBATCH" : [True, False],
				"LAYERS_SIZE" : [(6,),(8,),(4,),(7,3),(7,5,3)],
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
	'''
	  normalizes a configuration conf, generating the parameters ETA_RANGE and ETA_DECREASING_PERIOD if conf.ETA_DECAY is True,
	  and the parameters MINIBATCH_SAMPLE if conf.MINIBATCH is True
	  
	  :params:
	   conf: the configuration to be normalized
	   
	  :returns: the normalized configuration
	'''
	
	if conf["ETA_DECAY"]:
		eta_min = conf["ETA"]/2
		eta_max = min ( 1, conf["ETA"]*3/2 ) 
		conf["ETA_RANGE"] = [ eta_min, eta_max ]
		conf["ETA_DECREASING_PERIOD"]=Params.ETA_DECREASING_PERIOD
	if conf["MINIBATCH"]:
		conf["MINIBATCH_SAMPLE"]=Params.MINIBATCH_SAMPLE
	return conf



class RandomSearch:
	'''
	  Performs a random search in an hyperparameters space. It takes in input the name and ranges of each hyperparameter
      and generates a number of configurations, each formed by choosing a random value for every hyperparameter. Then, it uses a CrossValidation to
	  test the goodness of every generated configuration, and reports the test results.
	  The tests are done in parallel on a given number of threads.
	'''
	
	def __init__ (self, train_set, train_labels, configurations_number=100, threads_number=4, model_name="random_search_test" ):
		'''
		  Creates a new instance of class RandomSearch
		  
		  :parameters:
		   train_set:             sequence of instance to use as train set 
		   train_labels:          expected outputs for the instance of train_set. train_labels[i] is the expected output for the pattern train_set[i]
		   configurations_number: the number of configurations to be generated. Default:100
		   threads_number:        the number of threads to be spawn to parallelize the tests. Default: 4. 
		   model_name:            a string that identifies this model.
		'''
		
		self.train_set = train_set
		self.train_labels = train_labels
		self.configurations_number = configurations_number
		self.threads_number = threads_number
		self.model_name = model_name
		self.results = {}
	
	def test_configuration (self, conf_dict):
		'''
		  Test a single configuration using K-fold cross validation, and then reports the results on a file.
		  If there was an error during the execution of the k-fold cross validation, reports it in the file, instead of the results
		  
		  :params:
		    conf_dict: a dictionary with elements in the form PARAM_NAME : value
		'''
		params = Params.ConfigurableParams (conf_dict)
		print ("[RandomSearch] Starting CrossValidation with id {}".format(params.ID))
		name = self.model_name + "_" + str(params.ID)
		cv = KFoldCrossValidation ( self.train_set, self.train_labels, params=params, K=Params.NUM_FOLDS, model_name = name )
		try:
			cv.perform ()
			cv.dump ()			
			self.results[name] = cv.mean_accuracy
			print ("{}\t{}".format(name, cv.mean_accuracy))
		except:
			print ("[RandomSearch] Problem with CrossValidation with id {}".format(params.ID))
			cv.report_error ()
			
		print ("[RandomSearch] Ended CrossValidation with id {}".format(params.ID))
		
		
	def perform ( self, choices_dict ):
		'''
		  performs the random search, given a dictionary with all the possible choices for every parameter.
		  
		  :params:
		   choices_dict: a dictionary with elements in the form PARAM_NAME: [list of values]
		'''
		
		print ("[RandomSearch] Initializing...")
		print ("MODEL\tACCURACY")
		
		configurations = []
		for i in range (self.configurations_number):
			conf = { "ID": i, "MAX_EPOCH": Params.MAX_EPOCH }
			for key, values in choices_dict.items():
				conf[key] = random.choice (values)
			normalize_configuration (conf)
			configurations.append (conf)
		print ("[RandomSearch] created {} random configurations".format(self.configurations_number))
		
		pool = Pool ( self.threads_number )
		pool.map (self.test_configuration, configurations)
		
		print ("[RandomSearch] Finished.")
		
	def dump_results ( self ):
		'''
		  print the results of all the tests on a file
		'''
		sorted_results = sorted ( self.results.items(), key=lambda x: x[1], reverse=True )
		with open("../dumps/" + self.model_name + "_results", "w") as fout:
			fout.write ("RESULTS SORTED BY ACCURACY\n\n")
			fout.write ( "\n".join( [name + "\t" + str(acc) for name, acc in sorted_results] ) )

class GridSearch:
	'''
	  performs a grid search on an hyperparameters space.
      It takes in input the name and ranges of each hyperparameter and generates
	  all the possible configurations obtained by choosing all the possible combinations of values for every hyperparameter.
	  Then, it uses a CrossValidation to test the goodness of every generated configuration, and reports the test
	  results.
	  The tests are done in parallel on a given number of threads.
	'''
	
	def __init__ (self, train_set, train_labels, threads_number=4, model_name="grid_search_test" ):
		'''
		  Creates a new instance of class GridSearch
		  
		  :parameters:
		   train_set:             sequence of instance to use as train set 
		   train_labels:          expected outputs for the instance of train_set. train_labels[i] is the expected output for the pattern train_set[i]
		   threads_number:        the number of threads to be spawn to parallelize the tests. Default: 4. 
		   model_name:            a string that identifies this model.
		'''
		
		self.train_set = train_set
		self.train_labels = train_labels
		self.threads_number = threads_number
		self.model_name = model_name
		self.results = {}
	
	def test_configuration (self, conf_dict):
		'''
		  Test a single configuration using K-fold cross validation, and then reports the results on a file.
		  If there was an error during the execution of the k-fold cross validation, reports it in the file, instead of the results
		  
		  :params:
		    conf_dict: a dictionary with elements in the form PARAM_NAME : value
		'''
		
		params = Params.ConfigurableParams (conf_dict)
		print ("[GridSearch] Starting CrossValidation with id {}".format(params.ID))
		name = self.model_name + "_" + str(params.ID)
		cv = KFoldCrossValidation ( self.train_set, self.train_labels, params=params, K=Params.NUM_FOLDS, model_name = name )
		try:
			cv.perform ()
			cv.dump ()
			self.results[name] = cv.mean_accuracy
			print ("{}\t{}".format(name, cv.mean_accuracy))
		except:
			print ("[RandomSearch] Problem with CrossValidation with id {}".format(params.ID))
			cv.report_error ()
		print ("[GridSearch] Ended CrossValidation with id {}".format(params.ID))
	
	def perform ( self, choices_dict ):
		'''
		  performs the grid search, given a dictionary with all the possible choices for every parameter.
		  
		  :params:
		   choices_dict: a dictionary with elements in the form PARAM_NAME: [list of values]
		'''
		
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
		'''
		  print the results of all the tests on a file
		'''
		
		sorted_results = sorted ( self.results.items(), key=lambda x: x[1], reverse=True )
		#~ print ("RESULTS: {}".format(sorted_results))
		with open("../dumps/" + self.model_name + "_results", "w") as fout:
			fout.write ("RESULTS SORTED BY ACCURACY\n\n")
			fout.write ( "\n".join( [name + "\t" + str(acc) for name, acc in sorted_results] ) )
	
#unit tests
if __name__=="__main__":
	
	# Iris
	#~ train_s = Iris.iris_train_set[int(len(Iris.iris_train_set)/8):] 
	#~ train_l = Iris.iris_train_labels[int(len(Iris.iris_train_set)/8):]
	#~ test_s =  Iris.iris_train_set[:int(len(Iris.iris_train_set)/8)] 
	#~ test_l =  Iris.iris_train_labels[:int(len(Iris.iris_train_set)/8)] 
	
	# Monk 3
	#~ train_s = Monk.monk3_training_set
	#~ train_l = Monk.monk3_training_labels
	#~ test_s  = Monk.monk3_test_set
	#~ test_l  = Monk.monk3_test_labels
	
	#Machine
	#~ train_s = Machine.machine_train_set
	#~ train_l = Machine.machine_train_labels
	#~ test_s = Machine.machine_test_set
	#~ test_l = Machine.machine_test_labels 
	
	# MLCUP2017
	train_s = MLCUP2017.cup_train_set
	train_l = MLCUP2017.cup_train_labels
	
	#random parameter search
	rs = RandomSearch ( train_s, train_l, configurations_number=1000, threads_number=4, model_name="MLCUP2017" )
	rs.perform (coarse_choices_dict)
	#~ rs.dump_results ()
	
	#grid search
	#~ gs = GridSearch ( train_s, train_l, threads_number=4, model_name="Monk3" )
	#~ gs.perform (fine_choices_dict)
	#~ gs.dump_results ()
	
