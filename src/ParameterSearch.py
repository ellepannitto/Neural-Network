
import numpy as np
import random
from multiprocessing import Pool
import Params
from CrossValidation import KFoldCrossValidation

import Iris
import Monk

choices_dict = {
				"ALPHA" : np.arange (0,1.1,0.1),
				"ETA" : np.arange (0.1,1.1,0.1),
				"LAMBDA" : np.arange (0.001,0.01,0.001),
				"ETA_DECAY" : [True],
				"MINIBATCH" : [True, False],
				"LAYERS_SIZE" : [(6,)],
			   }
class RandomSearch:
	
	def __init__ (self, train_set, train_labels, configurations_number=100, threads_number=4, model_name="grid_search_test" ):
		self.train_set = train_set
		self.train_labels = train_labels
		self.configurations_number = configurations_number
		self.threads_number = threads_number
		self.model_name = model_name
		
	
	def test_configuration (self, conf_dict):
		params = Params.ConfigurableParams (conf_dict)
		print ("[RandomSearch] Starting CrossValidation with id {}".format(params.ID))
		cv = KFoldCrossValidation ( self.train_set, self.train_labels, params=params, K=10, model_name = self.model_name + "_" + str(params.ID) )
		cv.perform ()
		cv.dump ()
		print ("[RandomSearch] Ended CrossValidation with id {}".format(params.ID))


	
	def perform ( self ):
		print ("[RandomSearch] Initializing...")
		
		configurations = []
		for i in range (self.configurations_number):
			conf = { "ID": i, "MAX_EPOCH": Params.MAX_EPOCH }
			for key, values in choices_dict.items():
				conf[key] = random.choice (values)
			if conf["ETA_DECAY"]:
				eta_min = conf["ETA"]/2
				eta_max = min ( 1, conf["ETA"]*3/2 ) 
				conf["ETA_RANGE"] = [ eta_min, eta_max ]
				conf["ETA_DECREASING_PERIOD"]=Params.ETA_DECREASING_PERIOD
			if conf["MINIBATCH"]:
				conf["MINIBATCH_SAMPLE"]=Params.MINIBATCH_SAMPLE
			configurations.append (conf)
		print ("[RandomSearch] created {} random configurations".format(self.configurations_number))
		
		pool = Pool ( self.threads_number )
		pool.map (self.test_configuration, configurations)
		
		print ("[RandomSearch] Finished.")

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
	
	rs = RandomSearch ( train_s, train_l, configurations_number=20, threads_number=4, model_name="Monk3" )
	rs.perform ()	
