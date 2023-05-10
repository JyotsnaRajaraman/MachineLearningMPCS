Files
-----
pegasos.py: where you will implement Pegasos
svm_qp.py: where you will implement the quadratic program approach to SVM
utils.py: contains a function for loading the data
news.mat: data file

Sample Usage
------------
Run the following scripts with specific settings for the arguments.
python pegasos.py --epochs 1000 --lamda 1.0 --batch_size 128

python svm_qp.py --C 1.0

If you run  `python pegasos.py` or `python svm_qp.py`, the arg_parse arguments
will be set to their default values.
