import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
from typing import Union
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

___author__ = "Hemlata Tak"

__email__ = "tak@eurecom.fr"


class HHTMultiScaleBlock(nn.Module):

    def __init__(self, input_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.input_dim = input_dim
        self.fusion = nn.Linear(input_dim * num_scales, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def compute_analytic_signal(self, x):
        N = x.shape[1]
        Xf = torch.fft.fft(x, dim=1)
        h = torch.zeros(N, device=x.device)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1 : N // 2] = 2
        else:
            h[0] = 1
            h[1 : (N + 1) // 2] = 2
        return torch.fft.ifft(Xf * h.view(1, -1, 1), dim=1)

    def forward(self, x):
        B, T, D = x.shape
        if_features = []
        for k in range(self.num_scales):
            if k == 0:
                sub_signal = x[:, 1:, :] - x[:, :-1, :]
                sub_signal = torch.nn.functional.pad(sub_signal, (0, 0, 1, 0))
            elif k == 1:
                sub_signal = x
            else:
                sub_signal = x.transpose(1, 2)
                sub_signal = torch.nn.functional.avg_pool1d(
                    sub_signal, kernel_size=3, stride=1, padding=1
                )
                sub_signal = sub_signal.transpose(1, 2)
            z = self.compute_analytic_signal(sub_signal)
            phase = torch.angle(z)
            inst_freq = torch.diff(phase, dim=1, prepend=phase[:, :1, :])
            if_features.append(torch.abs(inst_freq))
        scale_stack = torch.cat(if_features, dim=-1)
        feat = self.fusion(scale_stack)
        return self.norm(x + feat)


class SignalTextureBlock(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.texture_extractor = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, feat_seq):
        x_n = feat_seq[:, 1:-1, :]
        x_prev = feat_seq[:, :-2, :]
        x_next = feat_seq[:, 2:, :]
        tke = torch.abs(x_n**2 - x_prev * x_next)
        tke_stat = torch.mean(tke, dim=1, keepdim=True)
        tke_stat = F.pad(tke_stat, (0, 0, 0, 0))
        flux = torch.abs(feat_seq[:, 1:, :] - feat_seq[:, :-1, :])
        flux_stat = torch.std(
            flux, dim=1, keepdim=True
        )
        combined = torch.cat([tke_stat, flux_stat], dim=-1)
        texture_gate = self.texture_extractor(combined)
        return self.norm(x * texture_gate + (1 - texture_gate) * tke_stat)


class SSLModel(torch.nn.Module):

    def __init__(
        self,
        model_dir,
        prompt_dim=1024,
        num_hht_tokens=6,
        num_prompt_tokens=10,
        num_aux_tokens=6,
        dropout=0.1,
    ):
        super(SSLModel, self).__init__()
        self.sampling_rate = 16000
        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        self.model = Wav2Vec2Model.from_pretrained(
            model_dir, attn_implementation="eager"
        )
        self.model.config.output_hidden_states = True
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.prompt_dim = prompt_dim
        self.num_hht_tokens = num_hht_tokens
        self.num_prompt_tokens = num_prompt_tokens
        self.num_aux_tokens = num_aux_tokens
        self.prompt_embedding = nn.Parameter(
            torch.zeros(24, num_prompt_tokens, prompt_dim)
        )
        self.fprompt_embedding = nn.Parameter(
            torch.zeros(24, num_hht_tokens, prompt_dim)
        )
        self.hht_block = HHTMultiScaleBlock(input_dim=prompt_dim)
        self.aprompt_embedding = nn.Parameter(
            torch.zeros(24, num_aux_tokens, prompt_dim)
        )
        self.texture_block = SignalTextureBlock(input_dim=prompt_dim)
        val = math.sqrt(6.0 / float(2 * prompt_dim))
        nn.init.uniform_(self.prompt_embedding.data, -val, val)
        nn.init.uniform_(self.fprompt_embedding.data, -val, val)
        nn.init.uniform_(self.aprompt_embedding.data, -val, val)
        self.prompt_dropout = nn.Dropout(p=dropout)

    def forward(self, audio_data):
        feat = self.processor(
            audio_data, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).input_values
        feat = feat.squeeze(dim=0).to(self.fprompt_embedding.device)
        with torch.no_grad():
            feat = self.model.feature_extractor(feat)
            feat = feat.transpose(1, 2)
            hidden_state, _ = self.model.feature_projection(feat)
            pos_embed = self.model.encoder.pos_conv_embed(hidden_state)
            hidden_state = self.model.encoder.dropout(hidden_state + pos_embed)
        raw_seq = hidden_state
        B = hidden_state.size(0)
        all_hidden_states = []
        total_p_len = self.num_hht_tokens + self.num_prompt_tokens + self.num_aux_tokens
        for i in range(self.model.config.num_hidden_layers):
            f = self.prompt_dropout(
                self.hht_block(self.fprompt_embedding[i].expand(B, -1, -1))
            )
            p = self.prompt_dropout(self.prompt_embedding[i].expand(B, -1, -1))
            a = self.prompt_dropout(
                self.texture_block(self.aprompt_embedding[i].expand(B, -1, -1), raw_seq)
            )
            if i == 0:
                hidden_state = torch.cat((f, p, a, hidden_state), dim=1)
            else:
                hidden_state = torch.cat(
                    (f, p, a, hidden_state[:, total_p_len:, :]), dim=1
                )
            hidden_state = self.model.encoder.layers[i](hidden_state)[0]
            all_hidden_states.append(hidden_state)
        return all_hidden_states

    def extract_feat(self, audio_data):
        return self.forward(audio_data)


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x):
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x1, x2, master=None):
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x, master):
        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)
        return master

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11
        )
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22
        )
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12
        )
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12
        )
        att_map = att_board / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x, master, att_map):
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):

    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)
        return new_h

    def top_k_graph(self, scores, h, k):
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        h = torch.gather(h, 1, idx)
        return h


class Residual_block(nn.Module):

    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(
            nb_filts[0], nb_filts[1], kernel_size=(2, 3), padding=(1, 1), stride=1
        )
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            nb_filts[1], nb_filts[1], kernel_size=(2, 3), padding=(0, 1), stride=1
        )
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                nb_filts[0], nb_filts[1], padding=(0, 1), kernel_size=(1, 3), stride=1
            )
        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        return out


class PromptAASISTLightningModel(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        cp_path = (
            "/data/h802/research_home/lliqingcao/HF/huggingface/wav2vec2-xls-r-300m"
        )
        self.ssl_model = SSLModel(
            cp_path, num_hht_tokens=6, num_prompt_tokens=10, num_aux_tokens=6
        )
        self.LL = nn.Linear(1024, 128)
        self.first_bn = nn.BatchNorm2d(1)
        self.first_bn1 = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(0.5)
        self.drop_way = nn.Dropout(0.2)
        self.selu = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(
            Residual_block(nb_filts=filts[1], first=True),
            Residual_block(nb_filts=filts[2]),
            Residual_block(nb_filts=filts[3]),
            Residual_block(nb_filts=filts[4]),
            Residual_block(nb_filts=filts[4]),
            Residual_block(nb_filts=filts[4]),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1])
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)
        self.register_buffer("class_weights", args.class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, x):
        x_ssl_all_layers = self.ssl_model.extract_feat(x.squeeze(-1))
        p_len = (
            self.ssl_model.num_hht_tokens
            + self.ssl_model.num_prompt_tokens
            + self.ssl_model.num_aux_tokens
        )
        x_audio = x_ssl_all_layers[-1][:, p_len:, :]
        x = self.selu(
            self.first_bn(
                F.max_pool2d(self.LL(x_audio).transpose(1, 2).unsqueeze(1), (3, 3))
            )
        )
        x = self.selu(self.first_bn1(self.encoder(x)))
        w = self.attention(x)
        e_S = torch.sum(x * F.softmax(w, dim=-1), dim=-1).transpose(1, 2) + self.pos_S
        out_S = self.pool_S(self.GAT_layer_S(e_S))
        e_T = torch.sum(x * F.softmax(w, dim=-2), dim=-2).transpose(1, 2)
        out_T = self.pool_T(self.GAT_layer_T(e_T))
        m1_exp = self.master1.expand(x.size(0), -1, -1)
        oT1, oS1, m1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=m1_exp)
        oS1, oT1 = self.pool_hS1(oS1), self.pool_hT1(oT1)
        oT_a, oS_a, m_a = self.HtrgGAT_layer_ST12(oT1, oS1, master=m1)
        oT1, oS1, m1 = oT1 + oT_a, oS1 + oS_a, m1 + m_a
        m2_exp = self.master2.expand(x.size(0), -1, -1)
        oT2, oS2, m2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=m2_exp)
        oS2, oT2 = self.pool_hS2(oS2), self.pool_hT2(oT2)
        oT_a, oS_a, m_a = self.HtrgGAT_layer_ST22(oT2, oS2, master=m2)
        oT2, oS2, m2 = oT2 + oT_a, oS2 + oS_a, m2 + m_a
        out_T, out_S, master = (
            self.drop_way(torch.max(oT1, oT2)),
            self.drop_way(torch.max(oS1, oS2)),
            self.drop_way(torch.max(m1, m2)),
        )
        T_max, T_avg = torch.max(torch.abs(out_T), dim=1)[0], torch.mean(out_T, dim=1)
        S_max, S_avg = torch.max(torch.abs(out_S), dim=1)[0], torch.mean(out_S, dim=1)
        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        output = self.out_layer(self.drop(last_hidden))
        return output, x_ssl_all_layers

    def training_step(self, batch, batch_idx):
        x, y = batch
        res = self(x)
        y_hat = res[0]
        loss = self.criterion(y_hat, y.view(-1).long())
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        res = self(x)
        loss = self.criterion(res[0], y.view(-1).long())
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, file_paths, labels = batch
        res = self(x)
        return {"file_paths": file_paths, "scores": res[0][:, 1]}

    def configure_optimizers(self):
        from transformers import get_linear_schedule_with_warmup
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad], lr=self.args.lr
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
