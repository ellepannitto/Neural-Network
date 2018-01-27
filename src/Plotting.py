
import matplotlib.pyplot as plt

x_limit = 0

def plot_loss_accuracy_per_epoch ( nn, show=True ):
	global x_limit
	x_limit = len(nn.train_losses)
	plt.plot(list(range(x_limit)), nn.train_losses, 'r--', label='train error')
	if nn.validation_losses is not None:
		plt.plot(list(range(x_limit)), nn.validation_losses, 'b-', label='validation error')
		plt.plot(list(range(x_limit)), nn.validation_accuracies, 'k-', label='validation Accuracy')
	
	if show:
		show_plot()

def show_plot ():
		plt.legend()
		plt.ylabel('Loss')
		plt.xlabel('epoch')
		axes = plt.gca()
		axes.set_xlim([0,x_limit])
		axes.set_ylim([0,1])
		plt.tight_layout()
		plt.show()
	
def plot_vertical_line (x):
	plt.plot([x,x], [0,1],'g')
	
