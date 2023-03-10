#!/bin/bash

batch_size=1024
model_name="resnet50"
image_folder="./Facial_Hair_37K/test/"
ground_truth_label="./Facial_Hair_37K/facial_hair_annotations.csv"
test_model_path="weights/model_resnet50.pth"
save_folder="./val_result/fh37k_BCE_reg_pretrained/test"

python test_pred.py \
-tip ${image_folder} \
-tlf ${ground_truth_label} \
-bs ${batch_size} \
-mn ${model_name} \
-tm ${test_model_path} \
-vr ${save_folder} \
-s \
-pna \
-lc \
-id
