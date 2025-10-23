working_dir=/workspace/ppo
REGION=us-east-1
model_name=Llama-3-it31-8B-pm-hs_bin-scaled_bt-lr4e-6-ep3-sn1gp10_4096_fullcov
# model_name=Llama-3-it31-8B-pm-hs_bin-sigmoid-lr4e-6-ep3-sn1gp10_4096_fullcov
save_path=/checkpoint/${model_name}
save_path_s3=${S3_BUCKET}/pretrain/${model_name}

sn_range=1
amplitude=10

# cd ${working_dir} && \
# huggingface-cli login --token ${HF_TOKEN} && \
# aws s3 cp --recursive ${save_path_s3} ${save_path} --region ${REGION} && \
# pip install seaborn && \
# export CUDA_VISIBLE_DEVICES=4,5,6,7
cd ${working_dir} && \
# torchrun --nproc_per_node=8 -m openrlhf.cli.eval_rewardbench --model_path ${save_path} --save_path ./eval_results/${model_name} --batch_size 1 --value_head_prefix score --model_type preference --use_sn --sn_range ${sn_range} --use_gp --gp_amplitude ${amplitude} --router random
torchrun --nproc_per_node=8 -m openrlhf.cli.eval_rmbench \
    --model_path ${save_path} \
    --save_path ./eval_results/${model_name} \
    --batch_size 1 \
    --value_head_prefix score \
    --model_type preference \
    --use_sn --sn_range ${sn_range} \
    --use_gp --gp_amplitude ${amplitude} \
    --router threshold \
    --aggregate mean \
    --threshold 1.45
# torchrun --nproc_per_node=8 -m openrlhf.cli.eval_sky --data_path ${data_path} --model_path ${save_path} --save_path /eval_results/${model_name} --batch_size 1 --value_head_prefix score --model_type preference --use_sn --sn_range ${sn_range} --use_gp --gp_amplitude ${amplitude} && \
# aws s3 cp /eval_results/${model_name} ${S3_BUCKET}/eval_results/rm/${save_model_name} --recursive "
