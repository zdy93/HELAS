#!/bin/bash
export DATA=[yelp|movie|n2c2|senti]
export MODEL=[HELAS|HELASW|HELASS|HELASA|Bar]
export ANN=[human|eye_tracking]
export HAM=0.02
export seed=1
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
    export HAM=${HAM}
fi
if [ $DATA == "senti" ]
then
    export MAX=128
elif [ $DATA == "yelp" ]
then
    export MAX=133
elif [ $DATA == "movie" ]
then
    export MAX=256
elif [ $DATA == "n2c2" ]
then
    export MAX=416
else
    export MAX=128
fi


python main_bert.py \
   --lamda 4 \
   --seed $seed \
   --annotator ${ANN} \
   --n_epochs 20 \
   --max_length ${MAX} \
   --ham_percent ${HAM} \
   --data_source ${DATA} \
   --log_dir ${DATA}-Bert-${MODEL}-${LOSS} \
   --model_type Bert-${MODEL}-${LOSS} \

python main_bert_self_label_first.py \
   --lamda 2 \
   --seed $seed \
   --annotator ${ANN} \
   --log_dir ${DATA}-self-Bert-RA-CE \
   --data_source ${DATA} \
   --ham_percent ${HAM} \
   --model_type Bert-RA-CE \
   --max_length ${MAX} \
   --batch_size 32 \
   --conf_thres 0.5 \
   --early_stop \

python main_bert_two_steps.py \
   --lamda 2 \
   --seed $seed \
   --annotator ${ANN} \
   --log_dir ${DATA}-self-Bert-RA-CE \
   --data_source ${DATA} \
   --ham_percent ${HAM} \
   --model_type Bert-RA-CE \
   --max_length ${MAX} \
   --batch_size 32 \
   --early_stop \


