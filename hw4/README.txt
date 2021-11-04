Instructions to run main_generate.py

NOTE: use "./main_generate.py" to print the usage and "./main_generate.py --help" for the help message
      this should get you 90% of the functionality of the script

Dataset assumptions: this model simply loads a .txt file as a raw string, and samples randomly from this, so the data
                     the data does not really need to be in any format.

Required arg: --data /path/to/data
              this is the path to the dataset with which you want to run your model on.
	      
Training:
./main_generate.py --data /path/to/training.txt \
		   --train \
		   --out_model /path/to/output_model.dat \
		   --saved_model /path/to/checkpointed_model.dat

Decoding:
./main_genearte.py --data /path/to/training.txt \
		   --decode \
		   --saved_model /path/to/checkpointed_model.dat \		   
		   --npred 150 \
		   --temp 0.8

Evaluation:
./main_generate.py --data /path/to/training.txt \
		   --saved_model /path/to/checkpointed_model.dat \
		   --eval_data /path/to/evaluation.txt


