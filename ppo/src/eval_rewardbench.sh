# model_name=Llama-3-it-8B-pm-hs-sigmoid-lr5e-6-ep3-mcd0.1
model_name=Llama-3-it-8B-pm-hs-sigmoid-lr5e-6-ep1-mcd0.0
# pip install seaborn
# aws s3 cp ${S3_BUCKET}/pretrain/${model_name} /workspace/ppo/checkpoint/${model_name} --recursive
save_path=/workspace/ppo/checkpoint/${model_name}
# python -m openrlhf.cli.eval_rewardbench --model_path /workspace/ppo/checkpoint/${model_name} --batch_size 8 --value_head_prefix score --model_type preference --n_dropout 1
cd /workspace/ppo
python -m openrlhf.cli.eval_rmbench --model_path ${save_path} --save_path ./eval_results/${model_name} --batch_size 1 --value_head_prefix score --model_type preference --use_mcd --mcd_p 0.0 --n_dropout 1