export CUDA_VISIBLE_DEVICES=0

TARGET="lmsys/vicuna-7b-v1.5"
DRAFT="double7/vicuna-68m"
DATASET="benchmarks/adaspec/datasets/spec-bench.jsonl"
TRACES=("benchmarks/adaspec/traces/conv.csv" "benchmarks/adaspec/traces/mooncake.csv" "benchmarks/adaspec/traces/code.csv" )

for TRACE in "${TRACES[@]}"
do
    # Threshold-based
    python benchmarks/adaspec/scripts/sweep_server.py     \
                    --model $TARGET   \
                    --speculative-model $DRAFT  \
                    --num-speculative-tokens 8 \
                    --port 10001 \
                    --result-file "7b_specbench"\
                    --dataset $DATASET \
                    --trace $TRACE \
                    --dynamic-spec \

    # AdaSpec
    python benchmarks/adaspec/scripts/sweep_server.py     \
                    --model $TARGET   \
                    --speculative-model $DRAFT  \
                    --num-speculative-tokens 8 \
                    --port 10003 \
                    --result-file "7b_specbench"\
                    --dataset $DATASET \
                    --trace $TRACE \
                    --rsd \

    # Baseline without SD
    python benchmarks/adaspec/scripts/sweep_server.py     \
                    --model $TARGET \
                    --port 10005 \
                    --trace $TRACE \
                    --dataset $DATASET \
                    --result-file "7b_specbench"

    # SD with static length
    for i in 1 3 5
    do
        python benchmarks/adaspec/scripts/sweep_server.py     \
                        --model $TARGET   \
                        --speculative-model $DRAFT   \
                        --num-speculative-tokens $i \
                        --port $((10005 + i)) \
                        --trace $TRACE \
                        --dataset $DATASET \
                        --result-file "7b_specbench"
    done
done
