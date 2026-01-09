from transformers import AutoModel, AutoTokenizer

model_name = "Dream-org/Dream-v0-Instruct-7B"
AutoModel.from_pretrained(model_name, cache_dir="/gpfs/workdir/aswalda/DiffusionLayerIntegratedGradients/Dream-v0", trust_remote_code=True)