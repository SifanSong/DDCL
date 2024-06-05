##################
#### only vis ####
##################
GPU_num='0' ## running
net_name=DDCL ## simsiam (baseline), DDCL, CLeVER_DDCL
DATASET=IN100
DATASET_PATH=<path to imagenet-100 or imagenet>
OTHER_PARA=("_tv11" "1" "" "" "") ## ("_tv11" "1" "None" "None" "_reg0.001" for CLeVER_DDCL)
EPOCH=200
HP1=("0.8") ## 1.0, 0.8
BATCH=("256")
LINERA_EPOCH=("199")
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
	for lr_epoch in {0..0..1}
	do

	TEST_AUG_TYPE=("basic" "basic_aug1_4_2" "aug1" "aug1_2" "aug1_3" "aug1_3_aug1_4_2")
	for test_aug in {0..5..1}
	do
	VIS_DIR=./Results/${PARA_dir}/vis/${TRIAL_name}/${TEST_AUG_TYPE[${test_aug}]}
	LI_VIS_DIR=./Results/${PARA_dir}/vis/${TRIAL_name}/linear_${TEST_AUG_TYPE[${test_aug}]}
	LIEL_VIS_DIR=./Results/${PARA_dir}/vis/${TRIAL_name}/linearelse_${TEST_AUG_TYPE[${test_aug}]}
	mkdir -p ${VIS_DIR}
	mkdir -p ${LI_VIS_DIR}
	mkdir -p ${LIEL_VIS_DIR}
	## grad cam of full representation
	python vis.py --arch resnet50 --data ${DATASET_PATH} --gpu ${GPU_num} --net ${net_name} \
	  --batch-size 1 --seed 0 --pretrained ./logs/${PARA_dir}/${TRIAL_name}/checkpoint_${LINERA_EPOCH[${lr_epoch}]}_best.pth.tar --hp1 1.0 --test_aug_type ${TEST_AUG_TYPE[${test_aug}]} --vis_output_dir ${VIS_DIR} --dataset_type ${DATASET} --avg_size ${OTHER_PARA[1]}
	## grad cam of DIR representation
	python vis.py --arch resnet50 --data ${DATASET_PATH} --gpu ${GPU_num} --net ${net_name} \
	  --batch-size 1 --seed 0 --pretrained ./logs/${PARA_dir}/${TRIAL_name}/linear_checkpoint_${LINERA_EPOCH[${lr_epoch}]}_best.pth.tar --hp1 ${HP1[${hp1}]} --test_aug_type ${TEST_AUG_TYPE[${test_aug}]} --vis_output_dir ${LI_VIS_DIR} --dataset_type ${DATASET} --avg_size ${OTHER_PARA[1]}
	## grad cam of DVR representation
	python vis.py --arch resnet50 --data ${DATASET_PATH} --gpu ${GPU_num} --net ${net_name} \
	  --batch-size 1 --seed 0 --pretrained ./logs/${PARA_dir}/${TRIAL_name}/linearelse_checkpoint_${LINERA_EPOCH[${lr_epoch}]}_best.pth.tar --hp1 ${HP1[${hp1}]} --test_aug_type ${TEST_AUG_TYPE[${test_aug}]} --vis_output_dir ${LIEL_VIS_DIR} --else_part else_part --dataset_type ${DATASET} --avg_size ${OTHER_PARA[1]}

	done
	done
done
done
done
done
done