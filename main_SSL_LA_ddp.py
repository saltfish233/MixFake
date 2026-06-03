import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from data_utils_SSL_ddp import GeneralDatamodule, load_protocol
from model_prompt_ddp import PromptAASISTLightningModel

__author__ = "Hemlata Tak"

__email__ = "tak@eurecom.fr"


class ScoreWriterCallback(Callback):

    def __init__(self, output_path):
        super().__init__()
        self.output_path = output_path

    def on_test_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            dir_name = os.path.dirname(self.output_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            with open(self.output_path, "w") as f:
                f.write("")
            print(f"📄 Created evaluation file at: {self.output_path}")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.global_rank == 0:
            paths = outputs["file_paths"]
            scores = outputs["scores"].detach().float().cpu().tolist()
            with open(self.output_path, "a") as f:
                for p, s in zip(paths, scores):
                    f.write(f"{p} {s}\n")

    def on_test_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print(f"✅ Evaluation complete. Scores saved to {self.output_path}")


def load_weights(model, checkpoint_path):
    print(f"🔄 Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    prompt_keys = [k for k in state_dict.keys() if "prompt_embeddings" in k]
    hht_keys = [k for k in state_dict.keys() if "hht_block" in k]
    missing_prompts = [k for k in missing if "prompt_embeddings" in k]
    missing_hht = [k for k in missing if "hht_block" in k]
    if len(prompt_keys) > 0 and len(missing_prompts) == 0:
        print("✨ [Success] Learned Prompt Embeddings loaded successfully!")
    elif len(missing_prompts) > 0:
        print(
            "⚠️ [Info] Prompt Embeddings not found in checkpoint. Initialized RANDOMLY (Train from scratch?)."
        )
    if len(hht_keys) > 0 and len(missing_hht) == 0:
        print("✨ [Success] HHT Block weights loaded successfully!")
    real_missing = [
        k
        for k in missing
        if "prompt_embeddings" not in k
        and "hht_block" not in k
    ]
    if len(real_missing) > 0:
        print(f"⚠️ Warning: Missing core keys (First 5): {real_missing[:5]}")
    else:
        print(f"✅ Weights loaded successfully.")


def main(args):
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    data_module = GeneralDatamodule(
        args=args,
        protocol_path=args.protocols_path,
        batch_size=args.batch_size,
        num_workers=4,
    )
    _, _, class_weights = load_protocol(args.protocols_path, "train")
    args.class_weights = class_weights
    model = PromptAASISTLightningModel(args)
    has_model_path = args.model_path and os.path.exists(args.model_path)
    has_resume = args.resume_checkpoint and os.path.exists(args.resume_checkpoint)
    if args.eval:
        if has_model_path:
            load_weights(model, args.model_path)
        else:
            print("⚠️ Warning: Evaluating with RANDOM weights!")
    else:
        if has_model_path:
            if not has_resume:
                print("🚀 [Train-Init] Preparing base weights...")
                load_weights(model, args.model_path)
    comment_str = f"_{args.comment}" if args.comment else ""
    logger_name = (
        f"epoch_{args.num_epochs}_bs_{args.batch_size}_lr_{args.lr}{comment_str}"
    )
    logger = TensorBoardLogger(save_dir="logs", name=logger_name)
    callbacks = []
    if not args.eval:
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            save_on_train_epoch_end=False,
            every_n_epochs=1,
        )
        callbacks.append(checkpoint_callback)
    if args.eval:
        if args.eval_output and args.eval_dataset:
            save_path = os.path.join(
                args.eval_output,
                args.comment if args.comment else "",
                f"{args.eval_dataset}_scores.txt",
            )
        else:
            save_path = "eval_scores.txt"
        score_writer = ScoreWriterCallback(output_path=save_path)
        callbacks.append(score_writer)
    if args.gpus:
        devices = args.gpus
        num_devices = len(devices)
        print(f"Using specified GPUs: {devices}")
    else:
        devices = torch.cuda.device_count()
        num_devices = devices
        print(f"Using all available GPUs: {num_devices}")
    use_ddp = num_devices > 1
    strategy = "ddp_find_unused_parameters_true" if use_ddp else "auto"
    if args.eval and num_devices > 1:
        print(
            "Warning: Switching to single device for evaluation to ensure safe file writing."
        )
        if isinstance(devices, list):
            devices = [devices[0]]
        else:
            devices = 1
        strategy = "auto"
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        max_epochs=args.num_epochs,
        callbacks=callbacks,
        logger=logger,
        deterministic=not args.cudnn_deterministic_toggle,
        benchmark=args.cudnn_benchmark_toggle,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
    )
    if args.eval:
        print(f"🚀 Starting Evaluation on {args.eval_dataset}...")
        trainer.test(model, datamodule=data_module)
    else:
        print("🚀 Starting Training...")
        ckpt_path = args.resume_checkpoint if has_resume else None
        if ckpt_path:
            print(f"Resuming training state from {ckpt_path}")
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASVspoof2021 baseline system (PyTorch Lightning)"
    )
    parser.add_argument("--protocols_path", type=str, default="database/")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to base model weights (General Model)",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training state (Training Only)",
    )
    parser.add_argument("--eval_output", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.000001)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--comment", type=str, default=None)
    parser.add_argument("--gpus", type=int, nargs="+", default=None)
    parser.add_argument("--eval_dataset", type=str, default=None)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument(
        "--cudnn-deterministic-toggle", action="store_false", default=True
    )
    parser.add_argument("--cudnn-benchmark-toggle", action="store_true", default=False)
    parser.add_argument("--algo", type=int, default=5)
    parser.add_argument("--nBands", type=int, default=5)
    parser.add_argument("--minF", type=int, default=20)
    parser.add_argument("--maxF", type=int, default=8000)
    parser.add_argument("--minBW", type=int, default=100)
    parser.add_argument("--maxBW", type=int, default=1000)
    parser.add_argument("--minCoeff", type=int, default=10)
    parser.add_argument("--maxCoeff", type=int, default=100)
    parser.add_argument("--minG", type=int, default=0)
    parser.add_argument("--maxG", type=int, default=0)
    parser.add_argument("--minBiasLinNonLin", type=int, default=5)
    parser.add_argument("--maxBiasLinNonLin", type=int, default=20)
    parser.add_argument("--N_f", type=int, default=5)
    parser.add_argument("--P", type=int, default=10)
    parser.add_argument("--g_sd", type=int, default=2)
    parser.add_argument("--SNRmin", type=int, default=10)
    parser.add_argument("--SNRmax", type=int, default=40)
    args = parser.parse_args()
    if not os.path.exists("models"):
        os.mkdir("models")
    main(args)
