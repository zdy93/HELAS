# HUG-Framework
Codes and Data for Explainable Text Classification with Partially Labeled Human Attention

## Requirement
### Language
* Python3 == 3.7.2
### Module
* torch==1.6.0+cu101
* transformers==3.0.2
* tqdm==4.48.2
* dotmap==1.3.17
* scikit-learn==0.23.2
* six==1.15.0
* matplotlib==3.3.1
* numpy==1.19.1
* pandas==1.1.0

## Word Embeddings
* [Glove](https://nlp.stanford.edu/projects/glove/), the embedding file that we used is the glove.6B.100d.txt in the [glove6B.zip](http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip)
* [BioMed](http://bio.nlplab.org/), the embedding file that we used is the [PubMed-w2v.bin](http://evexdb.org/pmresources/vec-space-models/PubMed-w2v.bin)

## Datasets
* [YELP-HAM](https://github.com/cansusen/Human-Attention-for-Text-Classification), we used the [ham_part3.csv](https://github.com/cansusen/Human-Attention-for-Text-Classification/blob/master/raw_data/ham_part3.csv) for our task.
* [N2C2 2014 Challenge](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)
* [Movie Reviews](http://www.eraserbenchmark.com/)
* [ZuCo](https://osf.io/q3zws/)


Preprocessed Data can be found in [data](./data) folders. We did not provide N2C2 data in the folder, because access to this dataset requires a license and agreement forms, which can be founded in the [link](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/).

Below lists the description of each file in the data folder
* [human_intersection.npy](./data/human_intersection.npy) Human attention signals of YELP-HAT dataset
* [mfd_scores.npy](./data/mfd_scores.npy) Mean Fix Duration vectors of ZuCo dataset, can be converted to human attention signals, corresponding to zuco_new_text.npy and zuco_gloves.npy
* [raw_reviews.npy](./data/raw_reviews.npy) text of YELP-HAT dataset
* [x_train.zip](./data/x_train.zip) zipped x_train.npy file, which is the Glove embedding vectors of YELP-HAT dataset
* [y_train.npy](./data/y_train.npy) classification label of YELP-HAT dataset (There is no x_test.npy or y_test.npy file, x_train.npy and y_train.npy contain all instances that we used for both model training and evaluation)
* [zuco_gloves.npy](./data/zuco_gloves.npy) Glove embedding vectors of ZuCo dataset
* [zuco_new_text.npy](./data/zuco_new_text.npy) text of ZuCo dataset
* [zuco_pubmed.npy](./data/zuco_pubmed.npy) BioMed embedding vectors of ZuCo dataset
* [zuco_pubmed_att_labels.npy](./data/zuco_pubmed_att_labels.npy) Mean Fix Duration vectors of ZuCo dataset, corresponding to zuco_pubmed.npy



## Model Prediction
* [main_bert.py](./main_bert.py) implements HUG-BERT and other method with BERT as the core sequence model (except Unguided BERT model) using YELP-HAT dataset
* [main_rnn.py](./main_rnn.py) implements HUG-GRU, HUG-LSTM and all other methods with GRU or LSTM as the core sequence model

Below is the sample script for running prediction.
```cmd
python main_bert.py
   --lamda 100
   --seed 11
   --annotator human_intersection
   --log_dir log-BertHUGAttention
   --model_type BertHUGAttention
```
Here, ```BertHUGAttention``` in the script refers to the model that we proposed in the paper, which utilizes BERT as the core sequence model.

We refer users to [main_bert.py](./main_bert.py) and other scripts to see the usage of all parameters.

[Download_ZuCo_and_Preprocessing.ipynb](./Download_ZuCo_and_Preprocessing.ipynb) , [Preprocessing_YELP.ipynb](./Preprocessing_YELP.ipynb), and [i2b2_process.ipynb](./i2b2_process.ipynb) are used for pre-processing ZuCo, YELP, and N2C2 dataset, respectively. 
