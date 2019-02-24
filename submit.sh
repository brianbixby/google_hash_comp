BUCKET=$GCS_BUCKET

TRAINER_PACKAGE_PATH="./trainer"
MAIN_TRAINER_MODULE="trainer.task"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="hashcomp_$now"

JOB_DIR=$BUCKET$JOB_NAME

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region us-west1 \
    --config config.yaml \
    --runtime-version 1.10 \
    -- \
    --output-dir $BUCKET"hashcomp_$now" \
    --learning-rate 0.01 \
    --save-checkpoint-steps 1 \

# --episodes \
# --reward-decay \
# --restore \
# --render \