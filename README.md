# Evaluating the correctness and compactness of counterfactual explanations with formal verification


This repository is heavily based on work by Eoin M. Kenny and Mark T. Keane in their paper _On Generating Plausible Counterfactual and
Semi-Factual Explanations for Deep Learning_. See original repository here:  https://github.com/EoinKenny/AAAI-2021


To run the main script, run the file run_counterfactual_expt.py

To create the pickle file, change _new_model_ variable to True in run_counterfactual_expt.py

To get the latent z files, run get_latent.py

To change the neural network you want to explain, adapt the local_models.py file and certain functions in the helper_functions.py file. 

To install all needed packages, activate a virtual environment with Python version 3.7 and run the requirements.txt file.

Not all relevant data is present due to being unable to upload large files. The original data folder can be downloaded via the original repository by Kenny mentioned above. 


Still a work in progress...
