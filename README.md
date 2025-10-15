# Benchmarking counterfactual explanations


This repository is partially based on work by Eoin M. Kenny and Mark T. Keane in their paper _On Generating Plausible Counterfactual and
Semi-Factual Explanations for Deep Learning_. See original repository here:  https://github.com/EoinKenny/AAAI-2021


--------- Quick start ---------
All (cleaned) results are saved in /results as csv files, including all tested evaluation metrics. These cleaned results are created by running notebooks/create_final_csv_mnist.ipynb or notebooks/create_final_csv_cifar.ipynb with the raw results from all models which are stored in /raw-results-data as tar balls.

To visualize these results use notebooks/visualize_mnist_final_results.ipynb or notebooks/visualize_cifar_final_results.ipynb. 


--------- Large files ---------
Large files (such as the pickle files) will be added via a google drive link after acceptance.

--------- Experiments PIECE -------------
To run the main script, run the file run_experiments_PIECE.py

To create the pickle file, change _new_model_ variable to True in run_experiments_PIECE.py

To get the latent z files, run get_latent.py

To change the neural network you want to explain, you can adapt the local_models.py file and certain functions in the helper_functions.py file. 

To install all needed packages, activate a virtual environment with Python version 3.7 and run the requirements_PIECE.txt file.

Not all relevant data is present due to being unable to upload large files. The original data folder can be downloaded via the original repository by Kenny mentioned above. 


--------- Experiments Alibi ------------------
Run the file run_experiments_alibi.py

The files mnist_ae.h5 and mnist_enc.h5 are for the CF-Proto model and are based on what is used in the alibi documentation. 

To install all needed packages, activate a virtual environment with Python version 3.9 and run the requirements_alibi.txt file


--------- Experiments Captum -----------------
These experiments were left out of the paper as the explanation method could not produce targeted explanations.







