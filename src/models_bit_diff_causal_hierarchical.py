from turtle import st
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import math
from einops import rearrange

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class BitDiffPredictorTCN(nn.Module):
    def __init__(self, args, num_classes, concat_channel, causal=False):
        super(BitDiffPredictorTCN, self).__init__()
        self.num_classes = num_classes
        self.goal_embedding = nn.Linear(concat_channel - num_classes, 64) 
        self.ms_tcn = DiffMultiStageModel(
            args.layer_type,
            args.kernel_size,
            args.num_stages,
            args.num_layers,
            args.model_dim,
            args.input_dim + concat_channel,#self.num_classes + self.num_classes,
            self.num_classes,
            args.channel_dropout_prob,
            args.use_features,
            causal
        )
        
        self.use_inp_ch_dropout = args.use_inp_ch_dropout
        if args.use_inp_ch_dropout:
            self.channel_dropout = torch.nn.Dropout1d(args.channel_dropout_prob)
    

        
    def forward(self, x, t, stage_masks, obs_cond=None, self_cond=None):
        # arange
        high_level_goal = self.goal_embedding(self_cond.mean(dim=1))
        
        x = rearrange(x, 'b t c -> b c t') # (16, 48, 2231)
        obs_cond = rearrange(obs_cond, 'b t c -> b c t') # features (16, 2048, 2231)
        self_cond = rearrange(self_cond, 'b t c -> b c t') # (16, 48, 2231)
        stage_masks = [rearrange(mask, "b t c -> b c t") for mask in stage_masks]
        if self.use_inp_ch_dropout:
            x = self.channel_dropout(x)
        
        # condition on input
        
        x = torch.cat((x, obs_cond), dim=1)
        x = torch.cat((x, self_cond), dim=1)
        
        #### For training
        #frame_wise_pred, frame_wise_feature = self.ms_tcn(x, t, stage_masks, high_level_goal.squeeze(0))
        #### For inference
        frame_wise_pred, frame_wise_feature = self.ms_tcn(x, t, stage_masks, high_level_goal)#.squeeze(0))
        frame_wise_pred = rearrange(frame_wise_pred, "s b c t -> s b t c")
        return frame_wise_pred, frame_wise_feature



class DiffMultiStageModel(nn.Module):
    def __init__(self, 
                layer_type,
                kernel_size,
                num_stages,
                num_layers,
                num_f_maps,
                dim, num_classes,
                dropout,
                use_features=False,
                causal=False):
        super(DiffMultiStageModel, self).__init__()

        self.use_features = use_features
        stage_in_dim = num_classes
        if self.use_features:
            stage_in_dim = num_classes + dim

        self.stage1 = DiffSingleStageModel(
            layer_type, 
            kernel_size,
            num_layers,
            num_f_maps,
            dim, 
            num_classes, 
            dropout, 
            causal_conv=causal)
        

        self.stages = nn.ModuleList(
            [
                copy.deepcopy(
                    DiffSingleStageModel(
                        layer_type,
                        kernel_size, 
                        num_layers, 
                        num_f_maps, 
                        stage_in_dim, 
                        num_classes, 
                        dropout, 
                        causal_conv=causal
                        )
                    ) for s in range(num_stages-1)
            ]
        )


    def forward(self, x, t, stage_masks, high_level_goal=None):
        out, out_features = self.stage1(x, t, stage_masks[0])
        outputs = out.unsqueeze(0)
        output_features = out_features.unsqueeze(0)
     
        for sn, s in enumerate(self.stages):
            if self.use_features:
                # 4/14 This is true
                out, out_features = s(torch.cat((F.softmax(out, dim=1) * stage_masks[sn], x), dim=1), t, stage_masks[sn], high_level_goal)
            else:
                out, out_features = s(F.softmax(out, dim=1) * stage_masks[sn], t, stage_masks[sn])
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
            output_features = torch.cat((output_features, out_features.unsqueeze(0)), dim=0)

        return outputs, output_features



class DiffSingleStageModel(nn.Module):
    def __init__(self, 
                layer_type, 
                kernel_size, 
                num_layers, 
                num_f_maps, 
                dim, 
                num_classes, 
                dropout, 
                causal_conv=False):
        super(DiffSingleStageModel, self).__init__()
        
        self.layer_types = {
            'base_dr': DiffDilatedResidualLayer,
            'gated': DiffDilatedGatedResidualLayer,
        }

        #    
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        # time cond
        time_dim = num_f_maps * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_f_maps),
            nn.Linear(num_f_maps, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # conv layers
        self.layers = nn.ModuleList(
            [ 
                copy.deepcopy(
                    self.layer_types[layer_type](
                        kernel_size,
                        2 ** i,
                        num_f_maps,
                        num_f_maps,
                        time_dim,
                        dropout,
                        causal_conv)
                    ) for i in range(num_layers)
            ]
        )

        # Feature projection layer for semantic space
        self.feature_proj = nn.Sequential(
            nn.Conv1d(num_f_maps, 512, 1),  # CLIP text dimension과 맞추기 ### 384
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

        

    # 4/14 added obs_cond
    def forward(self, x, t, mask, high_level_goal=None):
        # embed 
        out = self.conv_1x1(x) * mask  
        time = self.time_mlp(t)
        
        # pass through layers
        for layer in self.layers:
            out = layer(out, time, mask, high_level_goal)

        # Get rich semantic features
        semantic_features = self.feature_proj(out) * mask  # (B, 512, T)
        semantic_features = rearrange(semantic_features, 'b c t -> b t c')  # (B, T, 512)

        # output
        out_features = out * mask # (16, 64, 1816)
        out_logits = self.conv_out(out) * mask # (16, 10, 1816)/(16, 48, 1816)
        return out_logits, semantic_features#out_features



# BASE
class DiffDilatedResidualLayer(nn.Module):
    def __init__(self, kernel_size, dilation, in_channels, out_channels, time_channels=-1, dropout=0.2, causal_conv=False):
        super(DiffDilatedResidualLayer, self).__init__()

        # Net
        self.conv_dilated = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=int(kernel_size/2)*dilation, 
            dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.ch_dropout = nn.Dropout1d(dropout)      

        # Time Net
        self.time_channels = time_channels
        if time_channels > 0:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_channels, out_channels * 2)
            )


    def forward(self, x, t, mask, obs_cond=None):
        # conv net
        out = F.relu(self.conv_dilated(x))
        out = self.ch_dropout(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)

        # time conditioning
        if self.time_channels > 0:
            time_scale, time_shift = self.time_mlp(t).chunk(2, dim=1)
            time_scale = rearrange(time_scale, 'b d -> b d 1')
            time_shift = rearrange(time_shift, 'b d -> b d 1')
            out = out * (time_scale + 1) + time_shift

        return (x + out) * mask


# GATED
class DiffDilatedGatedResidualLayer(nn.Module):
    def __init__(self,
                kernel_size, 
                dilation, 
                in_channels, 
                out_channels, 
                time_channels=-1, 
                dropout=0.2, 
                causal_conv=False
    ):
        super(DiffDilatedGatedResidualLayer, self).__init__()
        
        self.conv_dilated = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=int(kernel_size/2)*dilation, 
            dilation=dilation
        )
        self.gate_conv_dilated = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=int(kernel_size/2)*dilation, 
            dilation=
            dilation
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.ch_dropout = nn.Dropout1d(dropout)
    

        # Time Net
        self.time_channels = time_channels
        if time_channels > 0:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_channels, out_channels * 2)
        )
        self.goal_attn = CausalGoalAttention(out_channels, num_heads=4, fusion="sum")
        
    def forward(self, x, t, mask, high_level_goal=None):
        conv_out = self.conv_dilated(x)
        gate_out = self.sigmoid(self.gate_conv_dilated(x)) 
        out = torch.mul(conv_out, gate_out)
        out = self.ch_dropout(out)

        if high_level_goal is not None:
            out = self.goal_attn(out, high_level_goal)  # (B, C, T)
        out = self.conv_1x1(out)
        out = F.relu(out)
        out = self.dropout(out)

        # time conditioning
        if self.time_channels > 0:
            time_scale, time_shift = self.time_mlp(t).chunk(2, dim=1)
            time_scale = rearrange(time_scale, 'b d -> b d 1')
            time_shift = rearrange(time_shift, 'b d -> b d 1')
            out = out * (time_scale + 1) + time_shift

        return (x + out) * mask

# 4/14 added high level goal predictor
class HighLevelGoalPredictor(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        
        self.classifier = nn.Linear(input_dim, int(num_classes))

    def forward(self, x):
        """
        x: (B, T, C)  # features per frame
        """
        x = rearrange(x, 'b c t -> b t c')
        attn_output, _ = self.attn(x, x, x)  # self-attention
        logits = self.classifier(attn_output)  # (B, T, num_classes)
        frame_probs = F.softmax(logits, dim=-1)
        goal_soft = frame_probs.mean(dim=1)
        goal_index = torch.argmax(goal_soft, dim=1)
        return logits.mean(dim=1), goal_index

# class CausalGoalAttention(nn.Module):
#     def __init__(self, x_dim, goal_dim=512, num_heads=4, fusion="sum"):
#         super().__init__()
#         self.fusion = fusion

#         # x self-attention (causal)
#         self.is_attn = nn.MultiheadAttention(embed_dim=x_dim, num_heads=num_heads, batch_first=True)
#         # cross-attention: query=x, key/value=goal
#         self.cs_attn = nn.MultiheadAttention(embed_dim=x_dim, num_heads=num_heads, batch_first=True)

#         # goal의 차원이 x_dim과 다르면 투사
#         self.goal_proj = nn.Linear(goal_dim, x_dim) if goal_dim != x_dim else nn.Identity()

#         if fusion == "sum":
#             self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable blend

#     @staticmethod
#     def _causal_mask(Tq, Tk, device):
#         # query 시점 i는 key 시점 j 중 j>i(미래) 를 보지 못하게 마스킹
#         i = torch.arange(Tq, device=device).unsqueeze(1)  # (Tq,1)
#         j = torch.arange(Tk, device=device).unsqueeze(0)  # (1,Tk)
#         # True = mask out
#         return (j > i)  # (Tq, Tk), dtype=bool

#     def forward(self, x, goal_embed):
#         """
#         x: (B, C, T)
#         goal_embed:
#             - (B, C) 또는 (B, Tg, Cg) 또는 (B, Cg)
#               * (B, C) / (B, Cg): 단일 goal → 길이 1의 시퀀스로 취급
#               * (B, Tg, Cg): 시계열 goal
#         return: (B, C, T)
#         """
#         B, C, T = x.shape
#         x_seq = x.permute(0, 2, 1)  # (B, T, C)

#         # 1) x 내부의 causal self-attention
#         self_mask = self._causal_mask(T, T, x.device)  # (T, T)
#         z_hat, _ = self.is_attn(x_seq, x_seq, x_seq, attn_mask=self_mask)  # (B, T, C)
# isnt it the same cat
#         # 2) x(query) → goal(key/value) causal cross-attention
#         #print(goal_embed.shape)
#         goal_seq = self.goal_proj(goal_embed)  # (B, Tg, C)
#         print(goal_seq.shape)
#         cross_mask = self._causal_mask(T, goal_seq.size(1), x.device)  # (T, Tg or 1)
#         x_hat, _ = self.cs_attn(query=x_seq, key=goal_seq, value=goal_seq, attn_mask=cross_mask)  # (B, T, C)

#         # 3) fusion
#         if self.fusion == "sum":
#             alpha = torch.clamp(self.alpha, 0.0, 1.0)
#             out_seq = alpha * z_hat + (1.0 - alpha) * x_hat  # (B, T, C)
#         else:
#             raise NotImplementedError("Only 'sum' fusion implemented")

#         return out_seq.permute(0, 2, 1)  # (B, C, T)

class CausalGoalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, fusion="sum"):
        super().__init__()
        self.is_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.cs_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.fusion = fusion
        # Gradient Explosion: use layner norm (8/26)
        self.ln_x = nn.LayerNorm(dim, eps=1e-5)

        if fusion == "sum":
            # learnable scalar parameter for blending
            self.alpha = nn.Parameter(torch.tensor(0.5))  # starts neutral

    def forward(self, x, goal_embed):
        """
        x: (B, C, T)
        goal_embed: (B, C)
        """
        B, C, T = x.shape
        x_seq = x.permute(0, 2, 1)  # (B, T, C)
        
        h = self.ln_x(x_seq)
        # IS-ATT: self-attention
        z_hat, _ = self.is_attn(h, h, h)  # (B, T, C)

        # CS-ATT: attention from x to goal
        x_hat_raw, _ = self.cs_attn(goal_embed.unsqueeze(1), x_seq, x_seq)  # (B, 1, C)
        x_hat = x_hat_raw.expand(-1, T, -1)

        if self.fusion == "sum":
            # learnable fusion weight
            alpha = torch.clamp(self.alpha, 0.0, 1.0)  # optional constraint
            out = alpha * z_hat + (1.0 - alpha) * x_hat  # (B, T, C)
        else:
            raise NotImplementedError("Only 'sum' mode with learnable alpha is implemented here.")

        return out.permute(0, 2, 1)  # (B, C, T)