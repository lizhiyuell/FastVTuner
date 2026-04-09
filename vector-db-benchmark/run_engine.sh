#!/usr/bin/env bash

set -e

# DATASETS=${DATASETS:-"*"}

SERVER_HOST=${SERVER_HOST:-"localhost"}

# SERVER_USERNAME=${SERVER_USERNAME:-"qdrant"}

SOURCE_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

function run_exp() {
    # sync 
    # sudo bash -c "echo 1 > /proc/sys/vm/drop_caches" 
    sudo rm -rf $SOURCE_DIR/results/*
    sudo rm -rf $SOURCE_DIR/engine/servers/milvus-single-node/volumes

    SERVER_PATH=$1
    ENGINE_NAME=$2
    DATASETS=$3
    LSTART_PERCENT=$4
    LEND_PERCENT=$5
    QSTART_PERCENT=$6
    QEND_PERCENT=$7
    MONITOR_PATH=$(echo "$ENGINE_NAME" | sed -e 's/[^A-Za-z0-9._-]/_/g')
    nohup bash -c "cd $SOURCE_DIR/monitoring && rm -f docker.stats.jsonl && bash monitor_docker.sh" > /dev/null 2>&1 &
    cd $SOURCE_DIR/engine/servers/$SERVER_PATH ; docker compose down > /dev/null; docker compose up -d > /dev/null
    
    # sleep 30

    until curl -sf http://localhost:9091/healthz > /dev/null 2>&1; do
        sleep 1
    done

    /root/anaconda3/envs/fastvtuner/bin/python $SOURCE_DIR/run.py --engines "$ENGINE_NAME" --datasets "${DATASETS}" --host "$SERVER_HOST" > /dev/null

    # exit
    cd $SOURCE_DIR/engine/servers/$SERVER_PATH ; docker compose down > /dev/null
    cd $SOURCE_DIR/monitoring && mkdir -p results && sudo mv docker.stats.jsonl ./results/${MONITOR_PATH}-docker.stats.jsonl
}

# function get_result() {
#     res_file=`ls $SOURCE_DIR/results/ | grep -v 'upload'` 
#     cat $SOURCE_DIR/results/$res_file | grep -E "mean_precisions|rps|p95_time" | awk '{print $2}' | sed 's#,##g'
# }
# get both upload and search time
function get_result() {
    result_dir="$SOURCE_DIR/results"

    upload_file=$(ls "$result_dir" | grep 'upload')
    search_file=$(ls "$result_dir" | grep 'search')

    upload_time=$(grep '"upload_time"' "$result_dir/$upload_file" \
                    | awk -F':' '{print $2}' | sed 's#[, ]##g')

    total_time=$(grep '"total_time"' "$result_dir/$upload_file" \
                    | awk -F':' '{print $2}' | sed 's#[, ]##g')

    search_vals=$(cat "$result_dir/$search_file" \
                    | grep -E "total_time|mean_precisions|rps|p95_time" \
                    | awk '{print $2}' \
                    | sed 's#,##g')

    echo "$upload_time"
    echo "$total_time"
    echo "$search_vals"
}

SERVER_PATH=${1:-milvus-single-node}
ENGINE_NAME=${2:-milvus-p10}
# DATASETS=${3:-deep-image-96-angular}
DATASETS=${3:-glove-100-angular-p-10}
# DATASETS=${3:-h-and-m-2048-angular-filters}
# DATASETS=${3:-h-and-m-2048-angular-no-filters}
LSTART_PERCENT=${4:-0}
LEND_PERCENT=${5:-100}
QSTART_PERCENT=${6:-0}
QEND_PERCENT=${7:-100}

run_exp $SERVER_PATH $ENGINE_NAME $DATASETS
get_result


# "nlist": 32768, "m":5, "nbits":8
# "nprobe": 16384
