import pandas as pd
import numpy as np

def accuracy_score(dataset_input):
	"""
	Lazy way to compute the accuracy of the dataset, the column survived 
	has to exist in the input DataFrame and the order has to be
	the same

	Parameters
	-----------
	dataset_input : Dataframe
		Dataset to verify
	"""

	# Requires a colum called survived
	assert 'survived' in dataset_input.columns,\
	"The input dataset for compute the accuracy requires one column named 'survived'"

	# Load the dataset
	sol = pd.read_csv("../input/sol.csv", index_col='id')

	# Compute the accuracy
	return np.sum(sol["survived"] == dataset_input["survived"]) / float(sol.shape[0]) 


def accuracy_score_numpy(y_test):
	sol = pd.read_csv("../input/sol.csv", index_col='id')["survived"].values
	return np.sum(sol==y_test)/float(len(sol))


if __name__ == "__main__":
	test = pd.read_csv("../input/sol.csv")
	print(accuracy_score(test))