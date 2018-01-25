
import Monk
import Params
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":
	
	#only one MONK
	train_sets   = [ Monk.monk3_training_set ]
	train_labels = [ [ y[0] for y in Monk.monk3_training_labels ] ]
	test_sets    = [ Monk.monk3_test_set ]
	test_labels  = [ [ y[0] for y in Monk.monk3_test_labels ] ]
		
	for i, train_s, train_l, test_s, test_l in zip ( range(1,len(train_sets)+1), train_sets, train_labels, test_sets, test_labels ):
		
		print ("--- MONK {} ---".format(i))
		
		enc = OneHotEncoder (sparse=False)
		encoded_train_s = enc.fit_transform (train_s)
		encoded_test_s = enc.fit_transform (test_s)
		
		clf = MLPClassifier(activation='logistic', solver='lbfgs', learning_rate_init=Params.ETA,
		                    momentum=Params.ALPHA, alpha=Params.LAMBDA, hidden_layer_sizes=(6,))

		clf.fit (encoded_train_s, train_l)
		
		l=[]
		predicted_train_l = clf.predict (encoded_train_s)
		for o,y in zip(predicted_train_l, train_l):
			l.append(1 if ((o>0.5 and y>0.5) or (o<=0.5 and y<=0.5)) else 0)
			
		print ("Accuracy on train set {}".format ( (sum(l)*1.0/len(l))) )

		l=[]
		predicted_test_l = clf.predict (encoded_test_s)
		for o,y in zip(predicted_test_l, test_l):
			l.append(1 if ((o>0.5 and y>0.5) or (o<=0.5 and y<=0.5)) else 0)
			
		print ("Accuracy on test set {}".format ( (sum(l)*1.0/len(l))) )
