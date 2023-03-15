#!/bin/bash

image_path_file="/path/to/file/contains/image/paths"
test_model_path="/path/to/model/weights"
output_file="/output/file/name"
save_folder="/output/folder/path"
batch_size=1024
model_name="which/model"

echo ${output_file}
python test_pred.py \
-tm ${test_model_path} \
-o ${output_file} \
-bs ${batch_size} \
-s \
-vr ${save_folder} \
-mn ${model_name} \
-ipf ${image_path_file} \
-lc
