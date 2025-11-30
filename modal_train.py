import modal
from pathlib import Path

app = modal.App(
    'heliumSLM-training'
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch",
        "datasets",
        "wandb",
        "tqdm",
        "tokenizers"
    )
    .add_local_dir(local_path="src", remote_path="/root/src")
    .add_local_dir(local_path="data", remote_path="/root/data")
    .add_local_dir(local_path="config", remote_path="/root/config")
    .add_local_file(local_path="train.py", remote_path="/root/train.py")
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600*4,
    volumes={"/root/checkpoints": modal.Volume.from_name("heliumSLM-checkpoints", create_if_missing=True)}
)
def run_remote_training():
    import sys
    sys.path.insert(0, "/root")
    
    from train import train, CONFIG
    
    print("Starting remote training on Modal GPU...")

    CONFIG['save_dir'] = "/root/checkpoints"
    CONFIG['num_workers'] = 4
    
    CONFIG['train_file'] = '/root/data/raw/hybrid_textbook_data.jsonl'
    CONFIG['val_file'] = '/root/data/raw/validation_textbook.jsonl'
    CONFIG['tokenizer_path'] = '/root/data/tokenizer/tiny_slm_tokenizer.json'
    
    train()
    
    print("Training completed.")
    
@app.local_entrypoint()
def main():
    print("Launching remote training job...")
    run_remote_training.remote()