#############################
#### semi and downstream ####
#############################
GPU_num='0'
net_name=simsiam ## simsiam (baseline), DDCL, CLeVER_DDCL
DATASET=IN100
DATASET_PATH=<path to imagenet-100 or imagenet>
OTHER_PARA=("_tv11" "1" "" "" "") ## _tv11
EPOCH=200
HP1=("1.0") ## 1.0, 0.8
BATCH=("256")
LINERA_EPOCH=("199")
AUG_TYPE=("aug1")
SEP_LAMBD=("1.0")
WARMUP=30
OTHER_LINEAR_PARA=("sgd" "" "" "")

GPU_num='0'
net_name=DDCL ## simsiam (baseline), DDCL, CLeVER_DDCL
DATASET=IN100
DATASET_PATH=<path to imagenet-100 or imagenet>
OTHER_PARA=("_tv11" "1" "" "" "") ## _tv11
EPOCH=200
HP1=("0.8") ## 1.0, 0.8
BATCH=("256")
LINERA_EPOCH=("199")
AUG_TYPE=("aug1")
SEP_LAMBD=("1.0")
WARMUP=30
OTHER_LINEAR_PARA=("sgd" "" "" "")

GPU_num='0'
net_name=CLeVER_DDCL ## simsiam (baseline), DDCL, CLeVER_DDCL
DATASET=IN100
DATASET_PATH=<path to imagenet-100 or imagenet>
OTHER_PARA=("_tv11" "1" "" "" "_reg0.001") ## _tv11
EPOCH=200
HP1=("0.8") ## 1.0, 0.8
BATCH=("256")
LINERA_EPOCH=("199")
AUG_TYPE=("aug1")
SEP_LAMBD=("1.0")
WARMUP=30
OTHER_LINEAR_PARA=("sgd" "" "" "")
####################
SCRIPT_NAME="main_lincls"
DS_DATA=("CUB200" "Flowers102" "Food101" "OxfordIIITPet" ${DATASET} ${DATASET} "CUB200" "Flowers102" "Food101" "OxfordIIITPet")
DS_MODE=("linear" "linear" "linear" "linear" "semi_1" "semi_10" "ds_ft" "ds_ft" "ds_ft" "ds_ft")
DS_BATCH=("32" "32" "256" "32" "32" "32" "32" "32" "256" "32")
DS_EPOCH=("200" "200" "200" "200" "200" "200" "200" "200" "200" "200")
DS_WEIGHT_DECAY=( "0.0" "0.0" "0.0" "0.0" "0.0001" "0.0001" "0.0001" "0.0001" "0.0001" "0.0001")
LR_CLASSIFER=("30" "30" "30" "30" "3" "3" "30" "30" "30" "30")
LR_BACKBONE=("0" "0" "0" "0" "0.03" "0.03" "0.03" "0.03" "0.03" "0.03")
DS_AUG=("_li_aug1" "_li_aug1" "_li_aug1" "_li_aug1" "_li_aug1" "_li_aug1" "_li_aug1" "_li_aug1" "_li_aug1" "_li_aug1")
DATASET_PATH_LIST=("/home/sifan2/Datasets/CUB_200_2011/" ${DATASET_PATH} ${DATASET_PATH} ${DATASET_PATH} ${DATASET_PATH} ${DATASET_PATH} "/home/sifan2/Datasets/CUB_200_2011/" ${DATASET_PATH} ${DATASET_PATH} ${DATASET_PATH})
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
for ds_num in {0..9..1}
do
	PARA_dir=${DATASET}_${EPOCH}_WP${WARMUP}${OTHER_PARA[0]}/${net_name}/${AUG_TYPE[${aug_type}]}/${BATCH[${batch}]}/${HP1[${hp1}]}/
	TRIAL_name=${net_name}_${BATCH[${batch}]}_${HP1[${hp1}]}_${SEP_LAMBD[${sep_lambd}]}${OTHER_PARA[4]}_rep${rep}

	FILE_name=lr${LR_CLASSIFER[${ds_num}]}_lrbk${LR_BACKBONE[${ds_num}]}_ep${DS_EPOCH[${ds_num}]}_bs${DS_BATCH[${ds_num}]}${DS_AUG[${ds_num}]}
	echo ${PARA_dir}/ds/ds_${DS_MODE[${ds_num}]}_${DS_DATA[${ds_num}]}_linear${LINERA_EPOCH[${lr_epoch}]}${OTHER_PARA[3]}_${TRIAL_name}_${OTHER_LINEAR_PARA[0]}_${FILE_name}.txt
	## train
	mkdir -p ./logs/${PARA_dir}/
	mkdir -p ./Results/${PARA_dir}/ds/

	## linear
	for lr_epoch in {0..0..1}
	do
	## all final tensor
	python ${SCRIPT_NAME}.py \
	 -a resnet50 --data ${DATASET_PATH_LIST[${ds_num}]} \
	 --ava-gpu ${GPU_num} --workers 32 \
	 --lr ${LR_CLASSIFER[${ds_num}]} --backbone_lr ${LR_BACKBONE[${ds_num}]} --batch-size ${DS_BATCH[${ds_num}]} --epochs ${DS_EPOCH[${ds_num}]} --net ${net_name} --wd ${DS_WEIGHT_DECAY[${ds_num}]} --ds_mode ${DS_MODE[${ds_num}]} \
	 --pretrained ./logs/${PARA_dir}/${TRIAL_name}/ckpt_epoch_${LINERA_EPOCH[${lr_epoch}]}_${TRIAL_name}.pth \
	 --exp_dir ./logs/${PARA_dir}/${TRIAL_name}/ --filename1 checkpoint_${LINERA_EPOCH[${lr_epoch}]}${OTHER_PARA[3]}_${DS_MODE[${ds_num}]}_${DS_DATA[${ds_num}]}_${OTHER_LINEAR_PARA[0]}_${FILE_name}.pth.tar --filename2 checkpoint_${LINERA_EPOCH[${lr_epoch}]}${OTHER_PARA[3]}_${DS_MODE[${ds_num}]}_${DS_DATA[${ds_num}]}_${OTHER_LINEAR_PARA[0]}_${FILE_name}_best.pth.tar --dataset_type ${DS_DATA[${ds_num}]} --avg_size ${OTHER_PARA[1]} --li_aug ${DS_AUG[${ds_num}]} > ./Results/${PARA_dir}/ds/ds_${DS_MODE[${ds_num}]}_${DS_DATA[${ds_num}]}_linear${LINERA_EPOCH[${lr_epoch}]}${OTHER_PARA[3]}_${TRIAL_name}_${OTHER_LINEAR_PARA[0]}_${FILE_name}.txt
	
	# hp1 part tensor
	python ${SCRIPT_NAME}.py \
	 -a resnet50 --data ${DATASET_PATH_LIST[${ds_num}]} \
	 --ava-gpu ${GPU_num} --workers 32 \
	 --lr ${LR_CLASSIFER[${ds_num}]} --backbone_lr ${LR_BACKBONE[${ds_num}]} --batch-size ${DS_BATCH[${ds_num}]} --epochs ${DS_EPOCH[${ds_num}]} --net ${net_name} --wd ${DS_WEIGHT_DECAY[${ds_num}]} --ds_mode ${DS_MODE[${ds_num}]} \
	 --pretrained ./logs/${PARA_dir}/${TRIAL_name}/ckpt_epoch_${LINERA_EPOCH[${lr_epoch}]}_${TRIAL_name}.pth \
	 --exp_dir ./logs/${PARA_dir}/${TRIAL_name}/ --filename1 linear_checkpoint_${LINERA_EPOCH[${lr_epoch}]}${OTHER_PARA[3]}_${DS_MODE[${ds_num}]}_${DS_DATA[${ds_num}]}_${OTHER_LINEAR_PARA[0]}_${FILE_name}.pth.tar --filename2 linear_checkpoint_${LINERA_EPOCH[${lr_epoch}]}${OTHER_PARA[3]}_${DS_MODE[${ds_num}]}_${DS_DATA[${ds_num}]}_${OTHER_LINEAR_PARA[0]}_${FILE_name}_best.pth.tar --hp1 ${HP1[${hp1}]} --dataset_type ${DS_DATA[${ds_num}]} --avg_size ${OTHER_PARA[1]} --li_aug ${DS_AUG[${ds_num}]} > ./Results/${PARA_dir}/ds/ds_${DS_MODE[${ds_num}]}_${DS_DATA[${ds_num}]}_linear${LINERA_EPOCH[${lr_epoch}]}${OTHER_PARA[3]}_${TRIAL_name}_hp${HP1[${hp1}]}_${OTHER_LINEAR_PARA[0]}_${FILE_name}.txt

	done
done
done
done
done
done
done