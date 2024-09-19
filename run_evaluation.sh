export CUDA_VISIBLE_DEVICES=7
# Real Estate 10k
python3 -m src.main +experiment=re10k mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_re10k.json checkpointing.load=checkpoints/re10k.ckpt

# ACID
python3 -m src.main +experiment=acid mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_acid.json checkpointing.load=checkpoints/acid.ckpt