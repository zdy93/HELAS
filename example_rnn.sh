#!/bin/bash
export DATA=[yelp|movie|n2c2|senti]
export MODEL=[HELAS|HELASW|HELASS|HELASA|Bar]
export ANN=[human|eye_tracking]
export RNN=[GRU|LSTM]
export LR=0.001
export seed=1
export HAM=0.02
if [ $MODEL == "Bar" ]
then
    export LOSS=MSE
else
    export LOSS=CE
fi
if [ $ANN == "eye_tracking" ]
then
    export HAM=1.0
else
    export HAM=$HAM
fi
if [ $RNN == "LSTM" ]
then
    export LAM=20
else
    export LAM=30
fi
if [ $DATA == "n2c2" ]
then
    export EMD=200
    export EBS=511
elif [ $DATA == "movie" ]
then
    export EMD=256
    export EBS=320
elif [ $DATA == "yelp" ]
then
    export EMD=133
    export EBS=300
else
    export EMD=128
    export EBS=300
fi

python main_rnn.py \
   --lamda ${LAM} \
   --seed $seed \
   --embedding_dim ${EMD} \
   --eval_batch_size ${EBS} \
   --annotator ${ANN} \
   --data_source ${DATA} \
   --log_dir ${DATA}-${RNN}-${MODEL}-${LOSS} \
   --ham_percent ${HAM} \
   --learning_rate ${LR} \
   --model_type ${RNN}-${MODEL}-${LOSS} \

python main_rnn_self_label_first.py \
   --lamda 2 \
   --seed $seed \
   --embedding_dim ${EMD} \
   --eval_batch_size ${EBS} \
   --annotator human \
   --data_source ${DATA} \
   --log_dir ${DATA}-${RNN}-RA-CE \
   --ham_percent ${HAM} \
   --learning_rate ${LR} \
   --model_type ${RNN}-RA-CE \
   --conf_thres 0.5 \
   --early_stop \

python main_rnn_two_steps.py \
   --lamda 2 \
   --seed $seed \
   --embedding_dim ${EMD} \
   --eval_batch_size ${EBS} \
   --annotator human \
   --data_source ${DATA} \
   --log_dir ${DATA}-${RNN}-RA-CE \
   --ham_percent 0.02 \
   --learning_rate ${LR} \
   --model_type ${RNN}-RA-CE \
   --early_stop \

