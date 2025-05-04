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
    def __init__(self, args, causal=False):
        super(BitDiffPredictorTCN, self).__init__()
        # self.goal_predictor = HighLevelGoalPredictor(
        #     input_dim=args.input_dim, # 2048
        #     num_classes=args.num_highlevel_classes,
        #     num_heads=4
        # )
        self.goal_embedding = nn.Embedding(num_embeddings=int(args.num_highlevel_classes), embedding_dim=64)
        self.ms_tcn = DiffMultiStageModel(
            args.layer_type,
            args.kernel_size,
            args.num_stages,
            args.num_layers,
            args.model_dim,
            args.input_dim + 2 * args.num_classes,# + 64,
            args.num_classes,
            args.channel_dropout_prob,
            args.use_features,
            causal
        )
        
        self.use_inp_ch_dropout = args.use_inp_ch_dropout
        if args.use_inp_ch_dropout:
            self.channel_dropout = torch.nn.Dropout1d(args.channel_dropout_prob)
    

        
    def forward(self, x, t, stage_masks, obs_cond=None, self_cond=None, goal=None):
        # arange
        x = rearrange(x, 'b t c -> b c t') # (16, 48, 2231)
        obs_cond = rearrange(obs_cond, 'b t c -> b c t') # features (16, 2048, 2231)
        self_cond = rearrange(self_cond, 'b t c -> b c t') # (16, 48, 2231)
        stage_masks = [rearrange(mask, "b t c -> b c t") for mask in stage_masks]

        #high_level_goal_softmax, high_level_goal = self.goal_predictor(obs_cond)
        high_level_goal = goal
        high_level_goal_softmax = goal
        high_level_goal = self.goal_embedding(high_level_goal.squeeze(-1))  # (B, 2048)
        
        #goal_embed = goal_embed.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # (B, 2048, T)

        if self.use_inp_ch_dropout:
            x = self.channel_dropout(x)
        
        # condition on input
        
        x = torch.cat((x, obs_cond), dim=1)
        x = torch.cat((x, self_cond), dim=1)
        #x = torch.cat((x, goal_embed), dim=1) 
        
        #### 4/14 Added obs_cond for feature->coarse label extraction.
        frame_wise_pred, _ = self.ms_tcn(x, t, stage_masks, high_level_goal)
        frame_wise_pred = rearrange(frame_wise_pred, "s b c t -> s b t c")
        return frame_wise_pred, high_level_goal_softmax



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
     
        for sn, s in enumerate(self.stages):
            if self.use_features:
                # 4/14 This is true
                out, out_features = s(torch.cat((F.softmax(out, dim=1) * stage_masks[sn], x), dim=1), t, stage_masks[sn], high_level_goal)
            else:
                out, out_features = s(F.softmax(out, dim=1) * stage_masks[sn], t, stage_masks[sn])
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs, out_features



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

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

        

    # 4/14 added obs_cond
    def forward(self, x, t, mask, high_level_goal=None):
        # embed 
        out = self.conv_1x1(x) * mask  
        time = self.time_mlp(t)
        
        # pass through layers
        for layer in self.layers:
            out = layer(out, time, mask, high_level_goal)

        # output
        out_features = out * mask
        out_logits = self.conv_out(out) * mask
        return out_logits, out_features



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
                causal_conv=False,
                goal_predictor=None,num_goal_classes=10
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
        #self.goal_embed = nn.Embedding(num_goal_classes, out_channels * 2)
        #self.goal_attn = GoalCausalAttention(out_channels, num_heads=4)
        self.goal_attn = CausalGoalAttention(out_channels, num_heads=4, fusion="sum")
        
    def forward(self, x, t, mask, high_level_goal=None):
        conv_out = self.conv_dilated(x)
        gate_out = self.sigmoid(self.gate_conv_dilated(x)) 
        out = torch.mul(conv_out, gate_out)
        out = self.ch_dropout(out)

        if high_level_goal is not None:
            #pass
            # high_level_goal: (B, 64)
            # out: (B, 64, T)
            out = self.goal_attn(out, high_level_goal)  # (B, C, T)
            # goal_attn: (B, 64, T)
            
            #out = out + goal_attn  # residual fusion
        
        out = self.conv_1x1(out)
        out = F.relu(out)
        out = self.dropout(out)

        # time conditioning
        if self.time_channels > 0:
            time_scale, time_shift = self.time_mlp(t).chunk(2, dim=1)
            time_scale = rearrange(time_scale, 'b d -> b d 1')
            time_shift = rearrange(time_shift, 'b d -> b d 1')
            out = out * (time_scale + 1) + time_shift

        # if high_level_goal is not None:
        #     goal_embed = self.goal_embed(high_level_goal.squeeze(-1))  # (B, 2D)
        #     scale, shift = goal_embed.chunk(2, dim=1)
        #     scale = rearrange(scale, 'b d -> b d 1')
        #     shift = rearrange(shift, 'b d -> b d 1')
        #     out = out * (scale + 1) + shift

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

class GoalCausalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x, goal_embed):
        """
        x: (B, C, T)
        goal_embed: (B, D) or (B, C)
        """
        B, C, T = x.shape
        x_seq = x.permute(0, 2, 1)  # (B, T, C)

        # # Use goal as query
        # goal_query = goal_embed.unsqueeze(1)  # (B, 1, C)
        # out, _ = self.attn(goal_query, x_seq, x_seq)  # (B, 1, C)
        # goal_context = out.permute(0, 2, 1).expand(-1, -1, T)  # (B, C, T)

        # Use goal as key/value
        goal_kv = goal_embed.unsqueeze(1).expand(-1, T, -1)  # (B, T, C)
        out, _ = self.attn(x_seq, goal_kv, goal_kv)  # (B, T, C)
        goal_context = out.permute(0,2,1)
        return goal_context

class CausalGoalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, fusion="sum"):
        super().__init__()
        self.is_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.cs_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.fusion = fusion

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
        

        # IS-ATT: self-attention
        z_hat, _ = self.is_attn(x_seq, x_seq, x_seq)  # (B, T, C)

        # CS-ATT: attention from x to goal
        goal_q = goal_embed.unsqueeze(1)  # (B, 1, C)
        x_hat_raw, _ = self.cs_attn(goal_q, x_seq, x_seq)  # (B, 1, C)
        x_hat = x_hat_raw.expand(-1, T, -1)

        if self.fusion == "sum":
            # learnable fusion weight
            alpha = torch.clamp(self.alpha, 0.0, 1.0)  # optional constraint
            out = alpha * z_hat + (1.0 - alpha) * x_hat  # (B, T, C)
        else:
            raise NotImplementedError("Only 'sum' mode with learnable alpha is implemented here.")

        return out.permute(0, 2, 1)  # (B, C, T)
