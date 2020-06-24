# King_Rook_King_ML
Analysis of various ML techniques and their effectiveness on UC Irvine's King-Rook-King dataset

To run this code, use the following arguments: The first command line argument will be one of three: -bagging: use random forest with bagging classification -svm: use SVM with a nonlinear kernel classification -nn: use neural network classification The second command line argument dictates whether to perform a Cross Validation: -0: do not run crossval on train_x data -1: run crossval on train_x data The third commandline argument dictates whether or not to train, then predict, then calculate score: -0: it doesn't -1: it does this

THIS CODE REQUIRES SCIKIT AND NUMPY. Simply install these through pip and you should be good Runs on Python 3.7+

This code does not ouput graphs. It will output the raw numbers that I used to construct my graph (for bagging) and the data that fills the tables for SVM and NN. The code defaults to 50% training data and 50% testing data unless the code is modified. I had to do this to have any semblance of realistic training times.

The relevant files are this readme, chessPrediction.py, krkopt.data (the data set), and report.pdf'
