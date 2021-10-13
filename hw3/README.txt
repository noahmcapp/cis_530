
## NOTE: moved pandas requirement to end of requirements.txt
         pip install was hanging with the original requirements.txt

Steps to install:

1) cd to hw3/
2) pip install virtualenv
3) virtualenv env
4) ./env/bin/pip install -r requirements.txt

Steps to Run:
1) source ./env/bin/activate
2) Create folder hw3/output
3) Train POS Tagger and generate prediction:
./pos_tagger.py -method <method: can be greedy, viterbi or beam> -ngram <n: 2,3,4>
For example: ./pos_tagger.py -method viterbi -ngram 3
4) Evaluate prediction result: ./evaluate.py -d data/dev_y.csv -p output/pred_y.csv

# NOTE: run deactivate to exit hw3 environment
