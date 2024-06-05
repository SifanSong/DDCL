###################
#### only eval ####
###################
GPU_num='0,1'
net_name=simsiam ## simsiam (baseline), DDCL, CLeVER_DDCL
DATASET=IN100 ## IN100 or Imagenet
DATASET_PATH=<path to imagenet-100 or imagenet>
OTHER_PARA=("_tv11" "1" "" "" "") ## ("_tv11" "1" "None" "None" "_reg0.001" for CLeVER_DDCL)
EPOCH=200
HP1=("1.0") ## 1.0 for simsiam, 0.8 for DDCL or CLeVER_DDCL
BATCH=("256")
LINERA_EPOCH=("199") ## require to be total_epoch-1
AUG_TYPE=("aug1") ## different augmentation types: aug1=BAug, aug1_2=CAug, aug1_4_2=CAug+ (identical to the manuscript)
SEP_LAMBD=("1.0") ## The coefficient of Orthogonal loss (default 1.0)
WARMUP=30

GPU_num='0,1'
net_name=DDCL ## simsiam (baseline), DDCL, CLeVER_DDCL
DATASET=IN100 ## IN100 or Imagenet
DATASET_PATH=<path to imagenet-100 or imagenet>
OTHER_PARA=("_tv11" "1" "" "" "") ## ("_tv11" "1" "None" "None" "_reg0.001" for CLeVER_DDCL)
EPOCH=200
HP1=("0.8") ## 1.0 for simsiam, 0.8 for DDCL or CLeVER_DDCL
BATCH=("512")
LINERA_EPOCH=("199") ## require to be total_epoch-1
AUG_TYPE=("aug1") ## different augmentation types: aug1=BAug, aug1_2=CAug, aug1_4_2=CAug+ (identical to the manuscript)
SEP_LAMBD=("1.0") ## The coefficient of Orthogonal loss (default 1.0)
WARMUP=30

GPU_num='0,1'
net_name=CLeVER_DDCL ## simsiam (baseline), DDCL, CLeVER_DDCL
DATASET=IN100 ## IN100 or Imagenet
DATASET_PATH=<path to imagenet-100 or imagenet>
OTHER_PARA=("_tv11" "1" "" "" "_reg0.001") ## ("_tv11" "1" "None" "None" "_reg0.001" for CLeVER_DDCL)
EPOCH=200
HP1=("0.8") ## 1.0 for simsiam, 0.8 for DDCL or CLeVER_DDCL
BATCH=("256")
LINERA_EPOCH=("199") ## require to be total_epoch-1
AUG_TYPE=("aug1") ## different augmentation types: aug1=BAug, aug1_2=CAug, aug1_4_2=CAug+ (identical to the manuscript)
SEP_LAMBD=("1.0") ## The coefficient of Orthogonal loss (default 1.0)
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

	for lr_epoch in {0..0..1}
	do

	TEST_AUG_TYPE=("basic" "aug1" "aug1_2" "aug1_4_2")
	for test_aug in {0..3..1}
	do
	#### Performance evaluation of perturbed input images
	## evaluation using full representation
	python only_eval.py --arch resnet50 --data ${DATASET_PATH} --ava-gpu ${GPU_num} --net ${net_name} \
	 --batch-size 1 --pretrained ./logs/${PARA_dir}/${TRIAL_name}/checkpoint_${LINERA_EPOCH[${lr_epoch}]}_best.pth.tar --hp1 1.0 --test_aug_type ${TEST_AUG_TYPE[${test_aug}]} --dataset_type ${DATASET} --avg_size ${OTHER_PARA[1]} > ./Results/${PARA_dir}/only_eval_${TEST_AUG_TYPE[${test_aug}]}_linear${LINERA_EPOCH[${lr_epoch}]}_${TRIAL_name}_lars_lr0.1_ep90.txt

	## evaluation using only DIR representation
	python only_eval.py --arch resnet50 --data ${DATASET_PATH} --ava-gpu ${GPU_num} --net ${net_name} \
	  --batch-size 1 --pretrained ./logs/${PARA_dir}/${TRIAL_name}/linear_checkpoint_${LINERA_EPOCH[${lr_epoch}]}_best.pth.tar --hp1 ${HP1[${hp1}]} --test_aug_type ${TEST_AUG_TYPE[${test_aug}]} --dataset_type ${DATASET} --avg_size ${OTHER_PARA[1]} > ./Results/${PARA_dir}/only_eval_${TEST_AUG_TYPE[${test_aug}]}_linear${LINERA_EPOCH[${lr_epoch}]}_${TRIAL_name}_hp${HP1[${hp1}]}_lars_lr0.1_ep90.txt

	done
	done
done
done
done
done
done
