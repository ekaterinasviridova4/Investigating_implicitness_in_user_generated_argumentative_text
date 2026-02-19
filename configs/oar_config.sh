NAME="job_name"
PROJECT_NAME="project_name"
HOME="/home/user"
PROJECT_DIR="/path/to/your/project"
EMAIL="user@example.com"
LOGDIR="$HOME/logs"
export HUGGINGFACE_HUB_TOKEN="your_token_here"

# Make sure the log directory exists
mkdir -p "$LOGDIR"


W_HOURS=10                 # Walltime in hours 
L_NGPUS=2                  # Number of GPUs 
P_MINCUDACAPABILITY=7      # Minimum compute capability 
P_MINGPUMEMORY=40000       # Minimum GPU memory in MB 

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
     echo 'Starting Mistral classification...'; \
     python3 src/training/mistral_24B_finetune_imp_exp.py \
        --data_dir data/jsonl/cmv_imp_exp \
        --output_dir results_finetune_imp_exp; \
     python3 src/evaluation/mistral_24B_evaluate_imp_exp.py \
        --data_dir data/jsonl/cmv_imp_exp \
        --output_dir results_finetune_imp_exp \
        --pred_dir results_finetune_imp_exp/predictions \
        --split test; \
     echo 'Mistral classification completed.'
    " \
)
    #--stdout=logs/%jobid%.stdout \
    #--stderr=logs/%jobid%.stderr \
   
# Print the job ID / submission output
echo "$OAR_OUT"

