export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

TARGET="meta-llama/Llama-3.3-70B-Instruct"
DRAFT="meta-llama/Llama-3.2-1B-Instruct"
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
                    --tp 8 \
                    --result-file "70b_specbench"\
                    --dataset $DATASET \
                    --trace $TRACE \
                    --dynamic-spec \

    # AdaSpec
    python benchmarks/adaspec/scripts/sweep_server.py     \
                    --model $TARGET   \
                    --speculative-model $DRAFT  \
                    --num-speculative-tokens 8 \
                    --port 10003 \
                    --tp 8 \
                    --result-file "70b_specbench"\
                    --dataset $DATASET \
                    --trace $TRACE \
                    --rsd \

    # Baseline without SD
    python benchmarks/adaspec/scripts/sweep_server.py     \
                    --model $TARGET \
                    --port 10005 \
                    --tp 8 \
                    --trace $TRACE \
                    --dataset $DATASET \
                    --result-file "70b_specbench"

    # SD with static length
    for i in 1 3 5
    do
        python benchmarks/adaspec/scripts/sweep_server.py     \
                        --model $TARGET   \
                        --speculative-model $DRAFT   \
                        --num-speculative-tokens $i \
                        --trace $TRACE \
                        --dataset $DATASET \
                        --port $((10005 + i)) \
                        --tp 8 \
                        --trace $TRACE \
                        --dataset $DATASET \
                        --result-file "70b_specbench"
    done
done
