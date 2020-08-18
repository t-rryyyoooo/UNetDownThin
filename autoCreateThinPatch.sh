#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name createDownThinPatch.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file path.
if [ $which = "y" ];then
 JSON_NAME="createDownThinPatch.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"

# From json file, read required variable.
readonly IMAGE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".image_directory"))
readonly SAVE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".save_directory"))
readonly IMAGE_NAME=$(cat ${JSON_FILE} | jq -r ".image_name")
readonly MASK_NAME=$(cat ${JSON_FILE} | jq -r ".mask_name")
readonly LABEL_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".label_patch_size")
readonly PLANE_SIZE=$(cat ${JSON_FILE} | jq -r ".plane_size")
readonly OVERLAP=$(cat ${JSON_FILE} | jq -r ".overlap")
readonly NUM_REP=$(cat ${JSON_FILE} | jq -r ".num_rep")
readonly IS_LABEL=$(cat ${JSON_FILE} | jq -r ".is_label")
readonly NUM_DOWN=$(cat ${JSON_FILE} | jq -r ".num_down")
readonly NUM_ARRAY=$(cat ${JSON_FILE} | jq -r ".num_array[]")


for number in ${NUM_ARRAY[@]}
do
    image_path="${IMAGE_DIRECTORY}/case_${number}/${IMAGE_NAME}"
    save_path="${SAVE_DIRECTORY}/case_${number}"

    echo "IMAGE_PATH:${image_path}"
    echo "SAVE_PATH:${save_path}"
    echo "LABEL_PATCH_SIZE:${LABEL_PATCH_SIZE}"
    echo "PLANE_SIZE:${PLANE_SIZE}"
    echo "OVERLAP:${OVERLAP}"
    echo "NUM_REP:${NUM_REP}"
    echo "NUM_DOWN:${NUM_DOWN}"

    if [ $MASK_NAME = "No" ]; then
        mask=""
        echo "Mask:No"
    else
        mask_path="${IMAGE_DIRECTORY}/case_${number}/${MASK_NAME}"
        mask="--mask_path ${mask_path}"
        echo "MASK_PATH:${mask_path}"
    fi

    if [ $IS_LABEL = "No" ];then
        label=""
        echo "IS_LABEL:No"
    else
        label="--is_label"
        echo "IS_LABEL:Yes"
    fi

    python3 createDownThinPatch.py ${image_path} ${save_path} --label_patch_size ${LABEL_PATCH_SIZE} --plane_size ${PLANE_SIZE} --overlap ${OVERLAP} --num_rep ${NUM_REP} --num_down ${NUM_DOWN} ${label} ${mask}

# Judge if it works.
    if [ $? -eq 0 ]; then
     echo "Done."

    else
     echo "Fail"

    fi
done
