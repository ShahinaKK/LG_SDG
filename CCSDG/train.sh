# Path Preparation
export OUTPUT_FOLDER="/l/users/20020135/CCSDG/TEST"
export RIGAPLUS_DATASET_FOLDER="/l/users/20020135/CCSDG_DATA"

#Training: BinRushed as source domain#########
 ccsdg_train --model unet_ccsdg --gpu 0 --tag source_BinRushed \
 --log_folder $OUTPUT_FOLDER \
 --batch_size 8 \
 --initial_lr 0.01 \
 -r $RIGAPLUS_DATASET_FOLDER \
 --tr_csv $RIGAPLUS_DATASET_FOLDER/BinRushed_train.csv \
 --ts_csv $RIGAPLUS_DATASET_FOLDER/MESSIDOR_Base1_test.csv \
 $RIGAPLUS_DATASET_FOLDER/MESSIDOR_Base2_test.csv \
 $RIGAPLUS_DATASET_FOLDER/MESSIDOR_Base3_test.csv

# ##Training: Magrabia as source domain ########
# ccsdg_train --model unet_ccsdg --gpu 0 --tag source_Magrabia \
# --log_folder $OUTPUT_FOLDER \
# --batch_size 8 \
# --initial_lr 0.01 \
# -r $RIGAPLUS_DATASET_FOLDER \
# --tr_csv $RIGAPLUS_DATASET_FOLDER/Magrabia_train.csv \
# --ts_csv $RIGAPLUS_DATASET_FOLDER/MESSIDOR_Base1_test.csv \
# $RIGAPLUS_DATASET_FOLDER/MESSIDOR_Base2_test.csv \
# $RIGAPLUS_DATASET_FOLDER/MESSIDOR_Base3_test.csv
