
import NeuralNetwork
import os
import Params
import MLCUP2017
import Plotting
import Statistics

def parse_result_file (filename):
	ret = {}
	float_parameters = ["ETA", "LAMBDA", "ALPHA"]
	bool_parameters = ["MINIBATCH", "ETA_DECAY"]
	with open(filename) as fin:
		for line in fin:
			line = line.strip()
			#~ print ("{}".format( line, "mean accuracy" in line or "mean epochs" in line) )
			if line.startswith ("LAYERS_SIZE"):
				ret["LAYERS_SIZE"] = tuple ( [ int(el) for el in line.split("=")[1][1:-1].split(",") if not el=="" ])
			elif any( [ par in line for par in bool_parameters ] ):
				par = line.split("=")[0]
				val = line.split("=")[1] == "True"
				ret[par] = val
			elif "ETA_RANGE" in line:
				val = [ float(el) for el in line.split("=")[1][1:-1].split(",") ]
				ret["ETA_RANGE"] = val
			elif any( [ par in line for par in float_parameters ] ):
				par = line.split("=")[0]
				val = float(line.split("=")[1])
				ret[par] = val	
			elif "mean accuracy" in line or "mean epochs" in line:
				par = line.split(":")[0]
				val = float (line.split(":")[1].strip().split(" ")[0] )
				ret[par] = val
	
	ret["MINIBATCH_SAMPLE"] = 35
	return ret

if __name__ == "__main__":
	
	models = {}
	
	for filename in os.scandir("../dumps"):
		result_dict = parse_result_file (filename.path)
		#~ print ("{}".format(filename.path))
		#~ print ("{}".format(result_dict))
		#~ input()
		models[filename.path] = result_dict
	
	sorted_models = sorted ( models.items(), key=lambda x: x[1]["mean accuracy"])
	#~ print ("\n".join( [ str(el[0]) + "\t" + str(el[1]["mean accuracy"]) + "\t" + str(el[1]["mean epochs"]) for el in sorted_models][:100]))
	
	best_models = sorted_models[:50]
	best_models_with_high_epochs = sorted ( best_models, key=lambda x:x[1]["mean epochs"], reverse=True)[:10]
	
	print ("best models:\n{}".format("\n".join( [ str(el[0]) + "\t" + str(el[1]["mean accuracy"]) + "\t" + str(el[1]["mean epochs"]) for el in best_models])))
	print ("best with high epochs:\n{}".format("\n".join( [ str(el[0]) + "\t" + str(el[1]["mean accuracy"]) + "\t" + str(el[1]["mean epochs"]) for el in best_models_with_high_epochs])))
	
	train_s = MLCUP2017.cup_train_set 
	train_l = MLCUP2017.cup_train_labels
	validation_s =  MLCUP2017.cup_validation_set
	validation_l =  MLCUP2017.cup_validation_labels
	
	mues = MLCUP2017.mean_per_attribute_dataset[len(train_s[0])+1:]
	sigmas = MLCUP2017.std_per_attribute_dataset[len(train_s[0])+1:]
	
	print ("mues {}".format(mues))
	print ("sigmas {}".format(sigmas))
		
	
	for model_name, params_dict in best_models_with_high_epochs:
		
		model_name = os.path.basename (model_name)
		
		print ("MODEL {}".format(model_name))
		
		params = Params.ConfigurableParams ( params_dict )
		params.MAX_EPOCH = int (params_dict["mean epochs"])
				
		myNN = NeuralNetwork.NeuralNetwork(params)
		
		for size in params.LAYERS_SIZE:
			myNN.addLayer(size)
		
		myNN.set_train (train_s, train_l)
		myNN.set_validation (validation_s, validation_l)
		
		myNN.learn()
		
		Plotting.plot_loss_accuracy_per_epoch (myNN)
		
		
		a = Statistics.MEELoss ()
		for x,y in zip(train_s, train_l):
			o = myNN.predict (x)
			a.update (o, y)
			
		print ("Accuracy on train set {}".format ( a.get() ))

		with open("../results/"+model_name+"_predicted_validation", "w") as fout:

			a = Statistics.MEELoss ()
			a_not_normalized = Statistics.MEELoss ()
			for x,y in zip(validation_s, validation_l):
				o = myNN.predict (x)
				a.update (o, y)
				
				o_not_normalized = []
				y_not_normalized = []
				
				#~ print ("for this pattern: gold {}".format(y))
				#~ print ("for this pattern: predicted {}".format(o))
				
				for out, gold, mu, sigma in zip(o,y,mues,sigmas):
					o_not_normalized.append(out*sigma + mu)
					y_not_normalized.append(gold*sigma + mu)
					#~ print ("gold {} predicted {}".format(gold, out))
					#~ print ("gold not normalized {} predicted not normalized {}".format(y_not_normalized[-1], o_not_normalized[-1]))
					
				a_not_normalized.update ( o_not_normalized, y_not_normalized )
				
				fout.write (",".join([ str(el) for el in o_not_normalized ]))
				fout.write ("\n")
				
			print ("Accuracy on validation set (    normalized labels) {}".format ( a.get() ))
			print ("Accuracy on validation set (not normalized labels) {}".format ( a_not_normalized.get() ))
		
