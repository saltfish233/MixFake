import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import librosa
import random
from RawBoost import (
    ISD_additive_noise,
    LnL_convolutive_noise,
    SSI_additive_noise,
    normWav,
)

___author__ = "Hemlata Tak"

__email__ = "tak@eurecom.fr"


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def load_protocol(protocol_path, stage="train"):
    if stage == "train":
        target_stage_str = "train"
    elif stage == "val":
        target_stage_str = "dev"
    elif stage == "test":
        target_stage_str = "eval"
    else:
        raise ValueError(f"未知的 stage: {stage}")
    with open(protocol_path, "r") as f:
        l_meta = f.readlines()
    list_IDs = []
    labels_dict = {}
    count_real = 0
    count_fake = 0
    for line in l_meta:
        parts = line.strip().split()
        if len(parts) >= 9 and parts[-1] == target_stage_str:
            full_path = parts[5]
            label_str = parts[4]
            list_IDs.append(full_path)
            if label_str == "bonafide":
                labels_dict[full_path] = 1
                count_real += 1
            else:
                labels_dict[full_path] = 0
                count_fake += 1
    total = count_real + count_fake
    if total > 0:
        n_real = max(count_real, 1)
        n_fake = max(count_fake, 1)
        max_count = max(n_real, n_fake)
        w_fake = max_count / n_fake
        w_real = max_count / n_real
        if count_fake == 0:
            w_fake = 0.0
        if count_real == 0:
            w_real = 0.0
        class_weights = [w_fake, w_real]
    else:
        class_weights = [1.0, 1.0]
    if stage == "train":
        print(f"\n[Protocol Stats] Stage: {stage}")
        print(f"  - Real samples: {count_real}")
        print(f"  - Fake samples: {count_fake}")
        print(
            f"  - Calculated Weights -> Fake: {class_weights[0]:.4f}, Real: {class_weights[1]:.4f}"
        )
        if count_real > 0 and count_fake > 0:
            multiplier = (
                class_weights[1] / class_weights[0]
                if count_fake > count_real
                else class_weights[0] / class_weights[1]
            )
            print(
                f"  - Imbalance Multiplier: {multiplier:.2f}x (少数类Loss被放大了这么多倍)"
            )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return list_IDs, labels_dict, class_weights


def process_Rawboost_feature(feature, sr, args, algo):
    if algo == 1:
        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    elif algo == 3:
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )
    elif algo == 4:
        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )
    elif algo == 5:
        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    elif algo == 6:
        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )
    elif algo == 7:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )
    elif algo == 8:
        feature1 = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)
        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)
    else:
        feature = feature
    return feature


class DS_General(Dataset):

    def __init__(self, protocol_path, stage, args, algo):
        list_IDs, labels, class_weights = load_protocol(protocol_path, stage)
        self.list_IDs = list_IDs
        self.labels = labels
        self.stage = stage
        self.args = args
        self.algo = algo
        self.target_len = 64600
        self.target_duration_sec = self.target_len / 16000.0

    def get_weights(self):
        return self.class_weights

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        file_path = self.list_IDs[index]
        target = self.labels[file_path]
        use_partial_load = True
        total_duration_sec = 0.0
        try:
            total_duration_sec = librosa.get_duration(path=file_path)
        except Exception as e:
            print(f"警告: 无法获取 {file_path} 的时长. {e}. 将回退到完整加载。")
            use_partial_load = False
        if self.stage == "train":
            if use_partial_load and (total_duration_sec > self.target_duration_sec):
                max_offset_sec = total_duration_sec - self.target_duration_sec
                offset_sec = random.uniform(0.0, max_offset_sec)
                X_clip, fs = librosa.load(
                    file_path,
                    sr=16000,
                    offset=offset_sec,
                    duration=self.target_duration_sec,
                )
                if X_clip.shape[0] != self.target_len:
                    X_clip = librosa.util.fix_length(X_clip, size=self.target_len)
                Y_enhanced = process_Rawboost_feature(X_clip, fs, self.args, self.algo)
                x_inp = Tensor(Y_enhanced)
            else:
                X, fs = librosa.load(file_path, sr=16000)
                X_len = X.shape[0]
                if X_len < self.target_len:
                    Y_enhanced_short = process_Rawboost_feature(
                        X, fs, self.args, self.algo
                    )
                    num_repeats = int(self.target_len / Y_enhanced_short.shape[0]) + 1
                    Y_pad = np.tile(Y_enhanced_short, (1, num_repeats))[
                        :, : self.target_len
                    ][0]
                    x_inp = Tensor(Y_pad)
                elif X_len > self.target_len:
                    start_frame = random.randrange(0, X_len - self.target_len + 1)
                    X_crop = X[start_frame : start_frame + self.target_len]
                    Y_pad = process_Rawboost_feature(X_crop, fs, self.args, self.algo)
                    x_inp = Tensor(Y_pad)
                else:
                    Y_pad = process_Rawboost_feature(X, fs, self.args, self.algo)
                    x_inp = Tensor(Y_pad)
            return x_inp, target
        elif self.stage == "test" or self.stage == "val":
            X, fs = librosa.load(file_path, sr=16000)
            X_len = X.shape[0]
            if X_len > self.target_len:
                start_frame = random.randrange(0, X_len - self.target_len + 1)
                X_processed = X[start_frame : start_frame + self.target_len]
            else:
                num_repeats = int(self.target_len / X_len) + 1
                X_processed = np.tile(X, (1, num_repeats))[:, : self.target_len][0]
            x_inp = Tensor(X_processed)
            if self.stage == "val":
                return x_inp, target
            return x_inp, file_path, target


class GeneralDatamodule(pl.LightningDataModule):

    def __init__(self, args, protocol_path, batch_size=14, num_workers=8):
        super().__init__()
        self.args = args
        self.protocol_path = protocol_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.algo = args.algo

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = DS_General(
                protocol_path=self.protocol_path,
                stage="train",
                args=self.args,
                algo=self.algo,
            )
            self.val_ds = DS_General(
                protocol_path=self.protocol_path,
                stage="val",
                args=self.args,
                algo=self.algo,
            )
            print(
                f"[DataModule] Train samples: {len(self.train_ds)}, Val samples: {len(self.val_ds)}"
            )
        if stage == "test" or stage is None:
            self.test_ds = DS_General(
                protocol_path=self.protocol_path,
                stage="test",
                args=self.args,
                algo=self.algo,
            )
            print(f"[DataModule] Test samples: {len(self.test_ds)}")

    def get_train_ds(self):
        return self.train_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
