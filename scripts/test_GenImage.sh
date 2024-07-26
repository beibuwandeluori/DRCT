# default
MODEL_NAME=convnext_base_in22k
MODEL_PATH=../output/pretrained/DRCT-2M/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth
DEVICE_ID=0
EMBEDDING_SIZE=1024
MODEL_NAME=${1:-$MODEL_NAME}
MODEL_PATH=${2:-$MODEL_PATH}
DEVICE_ID=${3:-$DEVICE_ID}
EMBEDDING_SIZE=${4:-$EMBEDDING_SIZE}
ROOT_PATH=/disk1/chenby/dataset/AIGC_data/GenImage/
FAKE_ROOT_PATH=""
DATASET_NAME=GenImage
SAVE_TXT=../output/results/GenImage_metrics_rebuttal.txt
INPUT_SIZE=224
BATCH_SIZE=24
FAKE_INDEXES=(1 2 3 4 5 6 7 8)
for FAKE_INDEX in ${FAKE_INDEXES[@]}
do
  echo FAKE_INDEX:${FAKE_INDEX},MODEL_NAME:${MODEL_NAME},MODEL_PATH:${MODEL_PATH}
  python train.py --root_path ${ROOT_PATH} --fake_root_path '' --model_name ${MODEL_NAME} \
                  --input_size ${INPUT_SIZE} --batch_size ${BATCH_SIZE} --device_id ${DEVICE_ID} --is_test \
                  --model_path ${MODEL_PATH} --is_crop --fake_indexes ${FAKE_INDEX} \
                  --dataset_name ${DATASET_NAME} --save_txt ${SAVE_TXT} --embedding_size ${EMBEDDING_SIZE}
done