#####################
#### check_ortho ####
#####################
GPU_num='0'
net_name=simsiam ## simsiam (baseline), DDCL, CLeVER_DDCL
DATASET=IN100
DATASET_PATH=<path to imagenet-100 or imagenet>
OTHER_PARA=("_tv11" "1" "" "" "") ## ("_tv11" "1" "None" "None" "_reg0.001" for CLeVER_DDCL)
EPOCH=500
HP1=("1.0") ## 1.0, 0.8
BATCH=("256")
LINERA_EPOCH=("199") ## require to be total_epoch-1
AUG_TYPE=("aug1")
SEP_LAMBD=("1.0")
WARMUP=30
####################
GPU_num='0'
net_name=DDCL ## simsiam (baseline), DDCL, CLeVER_DDCL
DATASET=IN100
DATASET_PATH=<path to imagenet-100 or imagenet>
OTHER_PARA=("_tv11" "1" "" "" "") ## ("_tv11" "1" "None" "None" "_reg0.001" for CLeVER_DDCL)
EPOCH=500
HP1=("0.8") ## 1.0, 0.8
BATCH=("256")
LINERA_EPOCH=("199") ## require to be total_epoch-1
AUG_TYPE=("aug1")
SEP_LAMBD=("1.0")
WARMUP=30
####################
GPU_num='0'
net_name=CLeVER_DDCL ## simsiam (baseline), DDCL, CLeVER_DDCL
DATASET=IN100
DATASET_PATH=<path to imagenet-100 or imagenet>
OTHER_PARA=("_tv11" "1" "" "" "_reg0.001") ## ("_tv11" "1" "None" "None" "_reg0.001" for CLeVER_DDCL)
EPOCH=500
HP1=("0.8") ## 1.0, 0.8
BATCH=("256")
LINERA_EPOCH=("199") ## require to be total_epoch-1
AUG_TYPE=("aug1")
SEP_LAMBD=("1.0")
WARMUP=30
####################
for rep in {0..0..1}
do
for batch in {0..0..1}
do
for hp1 in {0..0..1}
do
for aug_type in {0..0..1}
do
for sep_lambd in {0..0..1}
do
	PARA_dir=${DATASET}_${EPOCH}_WP${WARMUP}${OTHER_PARA[0]}/${net_name}/${AUG_TYPE[${aug_type}]}/${BATCH[${batch}]}/${HP1[${hp1}]}/
	TRIAL_name=${net_name}_${BATCH[${batch}]}_${HP1[${hp1}]}_${SEP_LAMBD[${sep_lambd}]}${OTHER_PARA[4]}_rep${rep}
	echo ${PARA_dir} ${TRIAL_name}
	OTH_DIR=./Results/${PARA_dir}/othor/
	mkdir -p ./${OTH_DIR}/

	for check_ep in {9..199..10} ## save checkpoint, depending on the "--save_freq 10" used in run_train.sh
	do
	python check_ortho.py --resume ./logs/${PARA_dir}/${TRIAL_name}/ckpt_epoch_${check_ep}_${TRIAL_name}.pth \
	  -a resnet50 --data ${DATASET_PATH} --save_freq 10 --fix-pred-lr \
	  --dist-url 'tcp://localhost:10016' --multiprocessing-distributed --world-size 1 --rank 0 --ava-gpu ${GPU_num} --workers 1 --epochs ${EPOCH} \
	  --exp_dir ./logs/${PARA_dir}/ --trial ${TRIAL_name} --batch-size 128 --net ${net_name} --hp1 ${HP1[${hp1}]} --aug ${AUG_TYPE[${aug_type}]} --sep_lambd ${SEP_LAMBD[${sep_lambd}]} --warmup_epochs ${WARMUP} --dataset_type ${DATASET} --avg_size ${OTHER_PARA[1]} > ./${OTH_DIR}/${TRIAL_name}_ep${check_ep}.txt

	done
done
done
done
done
done