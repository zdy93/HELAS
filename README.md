# HELAS
This repository is the official implementation of the IEEE BigData 2021 paper [Human-like Explanation for Text Classification with Limited Attention Supervision](https://ieeexplore.ieee.org/abstract/document/9671444).

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
* [Standard  Sentiment  Treebank (SST)](https://huggingface.co/datasets/sst#source-data)


Preprocessed Data can be found in [data](./data) folders. We did not provide N2C2 data in the folder, because access to this dataset requires a license and agreement forms, which can be founded in the [link](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/).

Below lists the description of files in each data folder
* [yelp_data](./yelp_data) data files for Yelp-HAT dataset
   * [human_intersection.npy](./data/human_intersection.npy) Human attention maps of YELP-HAT dataset
   * [raw_reviews.npy](./yelp_data/raw_reviews.npy) text of Yelp-HAT dataset
   * [x.zip](./yelp_data/x.zip) compressed x_train.npy file, which is the Glove embedding vectors of YELP-HAT dataset
   * [y_train.npy](./data/y.npy) classification label of Yelp-HAT dataset
* [eye_data](./eye_data) data files for external eye_tracking data
   * [zuco_gloves.npy](./eye_data/zuco_gloves.npy) Glove embedding vectors of ZuCo dataset
   * [zuco_new_text.npy](./eye_data/zuco_new_text.npy) text of ZuCo dataset
   * [mfd_scores.npy](./eye_data/mfd_scores.npy) Mean Fix Duration vectors of ZuCo dataset, can be converted to human attention maps, corresponding to zuco_gloves.npy and zuco_new_text.npy
   * [zuco_pubmed.npy](./eye_data/zuco_pubmed.npy) BioMed embedding vectors of ZuCo dataset
   * [zuco_pubmed_att_labels.npy](./eye_data/zuco_pubmed_att_labels.npy) Mean Fix Duration vectors of ZuCo dataset, corresponding to zuco_pubmed.npy
* [movie_data](./movie_data) data files for Movie Reviews Dataset
   * [raw_text_train.npy](./movie_data/raw_text_train.npy) text of Movie Reviews training data
   * [raw_text_val_test.npy](./movie_data/raw_text_val_test.npy) text of Movie Reviews test data
   * [x_train.rar](./movie_data/x_train.rar) compressed x_train.npy file, which is the Glove embedding vector of Movie Reviews training data
   * [x_val_test.zip](./movie_data/x_val_test.zip) compressed x_val_test.npy file, which is the Glove embedding vector of Movie Reviews test data
   * [att_labels_train.npy](./movie_data/att_labels_train.npy) Human attention maps of Movie Reviews training data
   * [att_labels_val_test.npy](./movie_data/att_labels_val_test.npy) Human attention maps of Movie Reviews test data  
   * [y_train.npy](./movie_data/y_train.npy) classification label of Movie Reviews training data
   * [y_val_test.npy](./movie_data/y_test.npy) classification label of Movie Reviews test data  
* [senti_data](.senti_data) data files for SST dataset
  * [att_labels_with_att_train.npy](./senti_data/att_labels_with_att_train.npy) Human attention maps of SST training data which were annotated with human attention map
  * [att_labels_with_att_val_test.npy](./senti_data/att_labels_with_att_val_test.npy) Human attention maps of SST test data which were annotated with human attention map
  * [raw_text_with_att_train.npy](./senti_data/raw_text_with_att_train.npy) text of SST training data which were annotated with human attention map
  * [raw_text_with_att_val_test.npy](./senti_data/raw_text_with_att_val_test.npy) text of SST test data which were annotated with human attention map
  * [raw_text_without_att_train.npy](./senti_data/raw_text_without_att_train.npy) text of SST training data which were not annotated with human attention map
  * [raw_text_without_att_val_test.npy](./senti_data/raw_text_without_att_val_test.npy) text of SST test data which were not annotated with human attention map
  * [x_with_att_train.npy](./senti_data/x_with_att_train.npy) Glove embedding vector of SST training data which were annotated with human attention map
  * [x_with_att_val_test.npy](./senti_data/x_with_att_val_test.npy) Glove embedding vector of SST test data which were annotated with human attention map
  * [x_without_att_train.rar](./senti_data/x_without_att_train.rar) compressed x_without_att_train.npy file, which is the Glove embedding vector of SST training data which were not annotated with human attention map
  * [x_without_att_val_test.rar](./senti_data/x_without_att_val_test.rar) compressed x_without_att_val_test.npy file, which is the Glove embedding vector of SST test data which were not annotated with human attention map
  * [y_with_att_train.npy](./senti_data/y_with_att_train.npy) classification label of SST training data which were annotated with human attention map
  * [y_with_att_val_test.npy](./senti_data/y_with_att_val_test.npy) classification label of SST test data which were annotated with human attention map
  * [y_without_att_train.npy](./senti_data/y_without_att_train.npy) classification label of SST training data which were not annotated with human attention map
  * [y_without_att_val_test.npy](./senti_data/y_without_att_val_test.npy) classification label of SST test data which were not annotated with human attention map


## Model Training
* [main_bert.py](./main_bert.py) implements HELAS and Barrett et al. with BERT as the core sequence model
* [main_rnn.py](./main_rnn.py) implements HELAS and Barrett et al. with GRU or LSTM as the core sequence model
* [main_bert_self_label_first.py](./main_bert_self_label_first.py) implements Self-labeling RA with BERT as the core sequence model
* [main_rnn_self_label_first.py](./main_rnn_self_label_first.py) implements Self-labeling RA with GRU or LSTM as the core sequence model
* [main_bert_two_steps.py](./main_bert_two_steps.py) implements Limited Supervised RA with BERT as the core sequence model
* [main_rnn_two_steps.py](./main_rnn_two_steps.py) implements Limited Supervised RA with GRU or LSTM as the core sequence model

See sample scripts [example_rnn.sh](./example_rnn.sh) and [example_bert.sh](./example_bert.sh) for running experiments. 

We refer users to [main_bert.py](./main_bert.py) and other scripts to see the usage of all parameters.

[Download_ZuCo_and_Preprocessing.ipynb](./Download_ZuCo_and_Preprocessing.ipynb) , [Preprocessing_YELP.ipynb](./Preprocessing_YELP.ipynb), [Preprocessing_movie.ipynb](./Preprocessing_movie.ipynb), and [Process_SST.ipynb](./Process_SST.ipynb) are used for pre-processing ZuCo, YELP, Movie review, and SST dataset.
