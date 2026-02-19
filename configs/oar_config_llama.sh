NAME="job_name"
PROJECT_NAME="project_name"
HOME="/home/user"
PROJECT_DIR="/path/to/your/project"
EMAIL="user@example.com"
LOGDIR="$HOME/logs"
export HUGGINGFACE_HUB_TOKEN="your_token_here"

# Make sure the log directory exists
mkdir -p "$LOGDIR"

# LLaMA 8B specific directories
MODEL_NAME="llama-8b"
OUTPUT_DIR="7B_Mistral_Llama/results_micro_llama-8b_finetune_finegrained"
DATA_DIR="data/jsonl/combined_finegrained"

W_HOURS=10                 # Walltime in hours
L_NGPUS=1                  # Number of GPUs
P_MINCUDACAPABILITY=7      # Minimum compute capability
P_MINGPUMEMORY=24000       # Minimum GPU memory in MB 

# Submit the job
OAR_OUT=$(oarsub \
    --name "$NAME" \
    --directory "$PROJECT_DIR" \
    --stdout="$LOGDIR/%jobid%.stdout" \
    --stderr="$LOGDIR/%jobid%.stderr" \
    --property="gpu_compute_capability>='$P_MINCUDACAPABILITY' and gpu_mem>='$P_MINGPUMEMORY'" \
    --l "nodes=1/gpu=$L_NGPUS,walltime=$W_HOURS" \
    --notify "[ERROR,INFO]mail:$EMAIL" \
    "export HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN; \
     module load conda; \
     source /path/to/miniconda3/bin/activate /path/to/your/env; \
     
     echo 'Starting LLaMA 8B fine-tuning...'; \
     python3 src/training/mistral_7B_llama_8B_finetune_finegrained.py \
        --model_name $MODEL_NAME \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR; \
     
     echo 'Starting LLaMA 8B evaluation...'; \
     python3 src/evaluation/mistral_7B_llama_8B_evaluate_finegrained.py \
        --model_name $MODEL_NAME \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --pred_dir $OUTPUT_DIR/predictions \
        --split test; \
     
     echo 'LLaMA 8B completed successfully!'
    " \
)

# Print the job ID / submission output
echo "$OAR_OUT"