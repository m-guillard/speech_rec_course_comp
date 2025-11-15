#!/usr/bin/env python
# finetune_optimized.py
import argparse
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import nemo.collections.asr as nemo_asr
import torch
from omegaconf import OmegaConf

def freeze_bottom_encoder_layers(asr_model, num_layers_to_freeze=12):
    enc = getattr(asr_model, 'encoder', None)
    if enc is None:
        return
    
    layer_stack = None
    for candidate in ['encoder_layers', 'layers']:
        if hasattr(enc, candidate):
            layer_stack = getattr(enc, candidate)
            break
    if layer_stack is None and hasattr(enc, '_modules') and 'layers' in enc._modules:
        layer_stack = enc._modules['layers']

    if layer_stack is None:
        print("Warning: could not find encoder layers to freeze.")
        return

    n = min(num_layers_to_freeze, len(layer_stack))
    print(f"Freezing bottom {n} encoder layers out of {len(layer_stack)}")
    for i in range(n):
        for p in layer_stack[i].parameters():
            p.requires_grad = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", type=str, required=True)
    ap.add_argument("--dev_manifest", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze_bottom_layers", type=int, default=12)
    ap.add_argument("--specaugment", action="store_true", help="Enable SpecAugment")
    args = ap.parse_args()

    print("Loading pre-trained model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_eo_conformer_transducer_large")

    # Data configs
    asr_model.cfg.train_ds.manifest_filepath = args.train_manifest
    asr_model.cfg.train_ds.batch_size = args.batch_size
    asr_model.cfg.train_ds.shuffle = True
    asr_model.cfg.train_ds.num_workers = 4

    asr_model.cfg.validation_ds.manifest_filepath = args.dev_manifest
    asr_model.cfg.validation_ds.batch_size = args.batch_size
    asr_model.cfg.validation_ds.shuffle = False
    asr_model.cfg.validation_ds.num_workers = 4

    # SpecAugment - configured at MODEL level, not train_ds level
    if args.specaugment:
        print("Enabling SpecAugment...")
        OmegaConf.set_struct(asr_model.cfg, False)
        asr_model.cfg.spec_augment = {
            "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
            "freq_masks": 2,
            "freq_width": 27,
            "time_masks": 2,
            "time_width": 70
        }
        OmegaConf.set_struct(asr_model.cfg, True)

    # Optimizer config
    asr_model.cfg.optim.name = 'adamw'
    asr_model.cfg.optim.lr = args.lr
    asr_model.cfg.optim.weight_decay = 0.01
    asr_model.cfg.optim.sched.name = 'CosineAnnealing'
    asr_model.cfg.optim.sched.warmup_steps = 4000

    # Setup
    asr_model.setup_training_data(asr_model.cfg.train_ds)
    asr_model.setup_validation_data(asr_model.cfg.validation_ds)
    freeze_bottom_encoder_layers(asr_model, num_layers_to_freeze=args.freeze_bottom_layers)
    asr_model.setup_optimization(asr_model.cfg.optim)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='{epoch}-{val_wer:.4f}',
        save_top_k=5,
        verbose=True,
        monitor='val_wer',
        mode='min'
    )

    trainer = pl.Trainer(
        devices=1, 
        accelerator='gpu', 
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        precision=16,
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=5.0,
    )
    
    print("Starting fine-tuning...")
    trainer.fit(asr_model)
    print(f"Training complete. Best model: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()