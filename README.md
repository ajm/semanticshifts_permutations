# Statistically Significant Detection of Semantic Shifts using Contextual Word Embeddings

**ALL DIRECTORIES GIVEN AS COMMAND LINE PARAMETERS ARE ASSUMED TO EXIST!**  

**If a command needs both data sets, then give them in chronological order!**

## 0. Data

    cd data
    for i in \*.bz2 ; do 
        bunzip2 -k $i 
    done
    cd ..


## 1. Setup

    module purge
    module load Python/3.7.0-intel-2018b # (don't use gcc8 version of py3)  
  
    python3 -m venv env
    source ./env/bin/activate
    pip install --upgrade pip
    pip install torch pandas

    git clone https://github.com/huggingface/transformers
    cd transformers
    python setup.py build
    python setup.py install
    cd examples
    pip install -r requirements.txt
    cd ../..


## 1. BERT finetuning

Finetune a BERT model. You need to edit `#SBATCH --workdir=/proj/ajmedlar/semanticshifts/bert_finetuning` line in finetuning.job to point to your PROJ directory. The three parameters are training data, output directory and number of epoches. The data was cleaned using `./scripts/clean_lfc_dataset_for_bert_finetuning.py` (does not include titles of comment threads, only comments with body tags).

    cd bert_finetuning

    sbatch finetuning.job ../data/LiverpoolFC_ALL_CLEAN.txt $WRKDIR/bert_finetuning/bert_lfc_5epoch 5

## 2. Extract embeddings and calculate P-values using permutation test

Extract embeddings and run permutations tests. You need to edit `#SBATCH --chdir=/projappl/project_2002983/ajmedlar/semanticshifts/bert_extract` line in finetuning.job to point to your PROJ directory. The 5 parameters are the BERT model directory, corpus 1, corpus 2, word list and results file name.

    cd bert_extract

    sbatch bert_extract.job $WRKDIR/bert_finetuning/bert_lfc_5epoch ../data/LiverpoolFC_13_CLEAN.txt ../data/LiverpoolFC_17_CLEAN.txt ../data/annotated_words.csv /scratch/project_2002983/ajmedlar/lfc/lfc.txt

## 3. Postprocess results using FDR correction

The results file only contains raw P-values. Run `postprocess.py` to apply Benjamini-Hochberg Procedure and add groundtruth data to results.

    python postprocess.py ../data/annotated_words.csv /scratch/project_2002983/ajmedlar/lfc/lfc.txt > lfc2.txt

The following words achieved statistical significance:

    tail -n+2 lfc2.txt | awk '( $7 < 0.05 )'

