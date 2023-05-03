DeepFrag has been enhanced, additional functionality was added. It is possible to reconstruct entire ligands from fragments predicted by DeepFrag.

New scripts:
- search.py
- tan_simi.py
- download.py
- free_test.py
- test.py
- smina_score.py
- sminascore_split.py
- sminatest.py
- sminatest_eval.py
- smina_correlation.py
- sminacor_eval.py
- ligand_reconstruction.py

Scripts will be shortly documented here, there is some further documentation in the scpripts themselves and the thesis may contain helpful information.

search.py:
This is a very short and simple script for sorting lists of pdb-ids according to the partition (TRAIN, VAL, TEST, specified in the paper) of the MOAD-Dataset, unknown pdb-ids will be given back in a seperate list.

tan_simi.py:
This is an algorithm to compare protein sequences for their Tanimoto similarity.

download.py:
Downloads and prepares files from RSCB for further use, input is a list of pdb-ids. It also creates a file with a list of tuples containing protein- and ligand-ids. This is used as input for other scripts.

free_test.py:
Method for the evaluation of DeepFrag that can be used independent from the MOAD-Dataset. Takes a list of any pdb protein- and ligand ids as input. USE_CPU set to true, if GPU is available the setting can be changed on the top of the script. (WARNING: there might be a mistake in the script!)
Use case example CLI: python3 free_test.py --pdbs ./pdb_data/protein_ligand_complexes --model ./models/retrained_model_01 --top_k 1 --save_path ./results/free_test_rm01.csv
--pdbs: path to pickle file containing a list of tuples of pdb-protein-ids and pdb-ligand-ids
--model: specify path to model to be tested
--top_k: specify cutoff value up to which predicted fragment the algorithm will look for the correct fragment
--save_path: give path to file, where the output is to be saved

test.py:
Method for evaluation of DeepFrag devised from existing methods, it takes the relevant data from a file that was given in the source code so it can only consider data from the MOAD-Dataset.
Use case example CLI: python3 test.py --save_path ./models/final_model --model ./models/final_model --samples_per_example 32
--save_path: path to save the numpy files of the predicted and correct fingerprints
--model: specify path to model to be tested
--samples_per_example: number of fingerprints to be predicted for one fragment of a protein ligand complex, will be averaged. The purpose is to counter lacking rotation invariance of the model.

smina_score.py:
Scoring a list of tuples of protein and ligand pdb-ids for affinity. Output will be saved to a csv-file.
Use case example CLI: python3 smina_score.py --pdbs ./pdb_data/protein_ligand_complexes --save_path ./results/smina_scores.csv
--pdbs: path to pickle file containing a list of tuples of pdb-protein-ids and pdb-ligand-ids
--save_path: give path to file, where the output is to be saved

sminascore_split.py:
Script for splitting data according to their smina scores into different subsets. Takes a csv file as input as produced by smina_score.py. 
Use case example CLI: python3 sminascore_split.py --csv_file ./results/smina_scores.csv --split_fn subset --threshold 200 --save_path ./results/new_partitions.txt
--csv_file: Give path to csv file containing protein id, ligand id and affinity for the complex.
--split_fn: If subset is specified as split function the Train-set will only contain data that is in the original train-set, same goes for the validation set.
--threshold: Set a cutoff value for affinity, complexes that score below that value wont be considered for the train- or val-set.
--save_path: path to file, where the output is to be saved

sminatest.py:
This script serves evaluating models according to the affinity score that protein ligand complexes achieve with their predicted fragments. So it checks whether the model is able to improve the binding affinity of a ligand with the fragments it predicts.
The output is saved to a csv file as affinity value for protein ligand complex, similar to the output file from smina_score.py and can be processed by sminatest_eval.py
Use case example CLI: python3 sminatest.py --pdbs ./pdb_data/protein_ligand_complexes --model ./models/final_model --save_path ./results/sminatest_fm.csv --x 0
--pdbs: path to pickle file containing a list of tuples of pdb-protein-ids and pdb-ligand-ids
--model: specify path to model to be tested
--save_path: give path to file, where the output is to be saved
--x: can be used to specify startpoint in the input list

sminatest_eval.py:
Computes the average affinity score for each model so that they are comparable, means it only takes complexes into account that can be found in the output of all models to be compared.

smina_correlation.py:
This Script checks for a correlation of the smina scores for the top k specified fragments. It checks the affinity score for the protein/ligand complex with the top predicted fragment replaced, the second fragment replaced etc. until the kth fragment and saves them to a csv file. The averages are computed with sminacor_eval.py.
Use case example CLI: python3 smina_correlation.py --pdbs /projects/mai/users/kkxw544_magdalena/deepfrag_data/protein_ligand_complexes --model /projects/mai/users/kkxw544_magdalena/deepfrag_enhanced/models/smin_04 --save_path /projects/mai/users/kkxw544_magdalena/deepfrag_enhanced/results/sminacor_smin04.csv --k 5 --x 0
--pdbs: path to pickle file containing a list of tuples of pdb-protein-ids and pdb-ligand-ids
--model: specify path to model to be tested
--save_path: give path to file, where the output is to be saved
--k: specify up to which top predicted fragment the algorithm will check the binding affinity score
--x: can be used to specify startpoint in the input list

sminacor_eval.py:
Computes the averages for the output files from smina_correlation.py, might need to be adjusted as the input files are specified in the script itself.

ligand_reconstruction.py:
Ligand_reconstruction.py provides three different methods for reconstructing ligands. The third method is not working yet. The interactive method performs best, usually the results have higher binding affinity then the original ligands. The iterative method performs usually worse. Currently the user only needs to provide  path to a pickle file containing a list of tuples of protein and ligand pdb-ids and a path for saving the output files. 
The output is a file containing info about the protein ligand complex, the original binding affinity and the binding affinities of the reconstructed ligands. It will also save pictures of 2D structures of the different ligands.
If the third method should be implemented the main function also needs adjustment.
Careful: the smina_score.py needs to be run first, the output file should be saved at the direction of --save_path, can be changed in the script.
Use case example CLI: python3 ligand_reconstruction.py --pdbs /projects/mai/users/kkxw544_magdalena/deepfrag_data/protein_ligand_complexes --model /projects/mai/users/kkxw544_magdalena/deepfrag_enhanced/models/smin_04 --save_path /projects/mai/users/kkxw544_magdalena/deepfrag_enhanced/results/
--pdbs: path to pickle file containing a list of tuples of pdb-protein-ids and pdb-ligand-ids
--model: specify path to model to be used for the reconstruction
--save_path: give path to folder, where the output is to be saved

Reminder: sminates_eval and sminacor_eval were created to run from the folder /my_scripts/, they need to be adjusted to run in the main folder.

Adapted Scripts:
- model_conf.py
- train.py