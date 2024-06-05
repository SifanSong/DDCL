#### configuration of pretrianing for simsiam
GPU_num='0,1'
DIST_URL=tcp://localhost:10014
net_name=simsiam ## simsiam (baseline), DDCL, CLeVER_DDCL
DATASET_PATH=<path to imagenet-100 or imagenet>
OTHER_PARA=("_tv11" "1" "" "" "") ## ("_tv11" "1" "None" "None" "_reg0.001" for CLeVER_DDCL)
EPOCH=200 ## total_epoch
HP1=("1.0") ## 1.0 for simsiam, 0.8 for DDCL or CLeVER_DDCL
BATCH=("256")
LINERA_EPOCH=("199") ## require to be total_epoch-1
AUG_TYPE=("aug1") ## different augmentation types: aug1=BAug, aug1_2=CAug, aug1_4_2=CAug+ (identical to the manuscript)
SEP_LAMBD=("1.0") ## The coefficient of Orthogonal loss (default 1.0)
WARMUP=30
OTHER_LINEAR_PARA=("sgd" "30" "256" "200") ## for IN100 dataset
# OTHER_LINEAR_PARA=("lars" "0.1" "4096" "90") ## for ImageNet dataset

#### configuration of pretrianing for DDCL
GPU_num='0,1'
DIST_URL=tcp://localhost:10006
net_name=DDCL ## simsiam (baseline), DDCL, CLeVER_DDCL
DATASET_PATH=<path to imagenet-100 or imagenet>
OTHER_PARA=("_tv11" "1" "" "" "") ## ("_tv11" "1" "None" "None" "_reg0.001" for CLeVER_DDCL)
EPOCH=200 ## total_epoch
HP1=("0.8") ## 1.0 for simsiam, 0.8 for DDCL or CLeVER_DDCL
BATCH=("256")
LINERA_EPOCH=("199") ## require to be total_epoch-1
AUG_TYPE=("aug1") ## different augmentation types: aug1=BAug, aug1_2=CAug, aug1_4_2=CAug+ (identical to the manuscript)
SEP_LAMBD=("1.0") ## The coefficient of Orthogonal loss (default 1.0)
WARMUP=30
OTHER_LINEAR_PARA=("sgd" "30" "256" "200") ## for IN100 dataset
# OTHER_LINEAR_PARA=("lars" "0.1" "4096" "90") ## for ImageNet dataset

#### configuration of pretrianing for CLeVER_DDCL
GPU_num='0,1'
DIST_URL=tcp://localhost:10010
net_name=CLeVER_DDCL ## simsiam (baseline), DDCL, CLeVER_DDCL
DATASET_PATH=<path to imagenet-100 or imagenet>
OTHER_PARA=("_tv11" "1" "" "" "_reg0.001") ## ("_tv11" "1" "None" "None" "_reg0.001" for CLeVER_DDCL)
EPOCH=200 ## total_epoch
HP1=("0.8") ## 1.0 for simsiam, 0.8 for DDCL or CLeVER_DDCL
BATCH=("256")
LINERA_EPOCH=("199") ## require to be total_epoch-1
AUG_TYPE=("aug1") ## different augmentation types: aug1=BAug, aug1_2=CAug, aug1_4_2=CAug+ (identical to the manuscript)
SEP_LAMBD=("1.0") ## The coefficient of Orthogonal loss (default 1.0)
WARMUP=30
OTHER_LINEAR_PARA=("sgd" "30" "256" "200") ## for IN100 dataset
# OTHER_LINEAR_PARA=("lars" "0.1" "4096" "90") ## for ImageNet dataset

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
	mkdir -p ./logs/${PARA_dir}/
	mkdir -p ./Results/${PARA_dir}/

	#### Code for Pre-training
	python main_simsiam.py \
	  -a resnet50 --data ${DATASET_PATH} --save_freq 10 --fix-pred-lr \
	  --dist-url ${DIST_URL} --multiprocessing-distributed --world-size 1 --rank 0 --ava-gpu ${GPU_num} --workers 64 --epochs ${EPOCH} \
	  --exp_dir ./logs/${PARA_dir}/ --trial ${TRIAL_name} --batch-size ${BATCH[${batch}]} --net ${net_name} --hp1 ${HP1[${hp1}]} --aug ${AUG_TYPE[${aug_type}]} --sep_lambd ${SEP_LAMBD[${sep_lambd}]} --warmup_epochs ${WARMUP} --avg_size ${OTHER_PARA[1]} > ./Results/${PARA_dir}/${TRIAL_name}.txt ## add --reg_lambd ${OTHER_PARA[4]} only when using configuration of pretrianing for CLeVER_DDCL 

	#### Code for Linear Probe
	for lr_epoch in {0..0..1}
	do
	## For simsiam and DDCL and CLeVER_DDCL
	## linear probe using full representation
	python main_lincls.py \
	  -a resnet50 --data ${DATASET_PATH} \
	  --dist-url ${DIST_URL} --multiprocessing-distributed --world-size 1 --rank 0 --ava-gpu ${GPU_num} --workers 32 \
	  --lr ${OTHER_LINEAR_PARA[1]} --batch-size ${OTHER_LINEAR_PARA[2]} --epochs ${OTHER_LINEAR_PARA[3]} --net ${net_name} \
	  --pretrained ./logs/${PARA_dir}/${TRIAL_name}/ckpt_epoch_${LINERA_EPOCH[${lr_epoch}]}_${TRIAL_name}.pth \
	  --exp_dir ./logs/${PARA_dir}/${TRIAL_name}/ --filename1 checkpoint_${LINERA_EPOCH[${lr_epoch}]}${OTHER_PARA[3]}.pth.tar --filename2 checkpoint_${LINERA_EPOCH[${lr_epoch}]}${OTHER_PARA[3]}_best.pth.tar --avg_size ${OTHER_PARA[1]} > ./Results/${PARA_dir}/linear${LINERA_EPOCH[${lr_epoch}]}${OTHER_PARA[3]}_${TRIAL_name}_${OTHER_LINEAR_PARA[0]}_lr${OTHER_LINEAR_PARA[1]}_ep${OTHER_LINEAR_PARA[3]}.txt ## add --lars only when using OTHER_LINEAR_PARA=("lars" "0.1" "4096" "90") for linear probe on ImageNet dataset 
	
	## For DDCL and CLeVER_DDCL
	## linear probe using only DIR representation
	python main_lincls.py \
	  -a resnet50 --data ${DATASET_PATH} \
	  --dist-url ${DIST_URL} --multiprocessing-distributed --world-size 1 --rank 0 --ava-gpu ${GPU_num} --workers 32 \
	  --lr ${OTHER_LINEAR_PARA[1]} --batch-size ${OTHER_LINEAR_PARA[2]} --epochs ${OTHER_LINEAR_PARA[3]} --net ${net_name} \
	  --pretrained ./logs/${PARA_dir}/${TRIAL_name}/ckpt_epoch_${LINERA_EPOCH[${lr_epoch}]}_${TRIAL_name}.pth \
	  --exp_dir ./logs/${PARA_dir}/${TRIAL_name}/ --filename1 linear_checkpoint_${LINERA_EPOCH[${lr_epoch}]}${OTHER_PARA[3]}.pth.tar --filename2 linear_checkpoint_${LINERA_EPOCH[${lr_epoch}]}${OTHER_PARA[3]}_best.pth.tar --hp1 ${HP1[${hp1}]} --avg_size ${OTHER_PARA[1]} > ./Results/${PARA_dir}/linear${LINERA_EPOCH[${lr_epoch}]}${OTHER_PARA[3]}_${TRIAL_name}_hp${HP1[${hp1}]}_${OTHER_LINEAR_PARA[0]}_lr${OTHER_LINEAR_PARA[1]}_ep${OTHER_LINEAR_PARA[3]}.txt ## add --lars only when using OTHER_LINEAR_PARA=("lars" "0.1" "4096" "90") for linear probe on ImageNet dataset 
	done
done
done
done
done
done