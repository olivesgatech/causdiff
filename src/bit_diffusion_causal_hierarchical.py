"""
https://github.com/lucidrains/denoising-diffusion-pytorch
"""
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import nn
from tqdm import tqdm

import clip

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "pred_goal_noise", "pred_goal_start"])

def cosine_loss(x, y):
    cos_sim = F.cosine_similarity(x, y, dim=-1)
    return 1 - cos_sim

class DiffusionModel(nn.Module):
    """
    Template model for the diffusion process
    """

    def __init__(
        self,
    ):
        super().__init__()


    def forward(self, X_0, X_t, batch):
        raise NotImplementedError("Nope")


def exists(x):
    return x is not None


def identity(t, *args, **kwargs):
    return t


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # B x 1 x 1 


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def l2norm(t):
    return F.normalize(t, dim=-1)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianBitDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        goalmodel:nn.Module,
        condition_x0:False,
        clip_model,
        *,
        num_classes=48,
        timesteps=1000,  
        ddim_timesteps=100,
        betas=None,
        loss_type="l2",
        objective="pred_x0",
        beta_schedule="cosine",
    ):
        
        super().__init__()
        
        print(f'Num classes : {num_classes}')
        print(f'Loss type : {loss_type}')
        print(f'Objective: {objective}')
        print(f'Beta schedule : {beta_schedule}')

        self.model = model
        self.num_classes = num_classes
        self.condition_x0 = condition_x0
        self.goalmodel = goalmodel

        # RECONSTRUCTION OBJ
        self.objective = objective
        self.loss_type = loss_type

        assert objective in {
            "pred_noise",
            "pred_x0",
        }, "objective must be either pred_noise (predict noise) or pred_x0(predict image start)"  # noqa E501


        # VARIANCE
        if betas is None:
            if beta_schedule == "linear":
                betas = linear_beta_schedule(timesteps)
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(timesteps)
            else:
                raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)


        # SAMPLING
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # PARAMS
        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # FORWARD DIFFUSION q(x_t | x_{t-1}) 
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))


        # POSTERIOR q(x_{t-1} | x_t, x_0)
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer("posterior_variance", posterior_variance)
        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)),)
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),)
        register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),)

        # DDIM
        self.eta = 0.
        c = timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, timesteps, c)))
        
        self.ddim_timesteps = ddim_timesteps
        self.ddim_timestep_seq = ddim_timestep_seq

        self.clip = clip_model
        for param in self.clip.parameters():
            param.requires_grad = False
    
    def semantic_consistency_loss(self, subgoals, actions, global_goal):
        """
        subgoals: (S,B, T, 10) - 각 timestamp의 subgoal
        actions: (S,B, T, 48) - 각 timestamp의 action
        global_goal: (B, 10) - 전체 goal
        """
        S, B, T, C = subgoals.shape
        batch_size = 128

        # 결과를 저장할 변수들
        total_subgoal_action_sim = 0
        total_subgoal_goal_sim = 0
        num_batches = 0

        # 배치 단위로 처리
        for start_idx in range(0, S*B*T, batch_size):
            end_idx = min(start_idx + batch_size, S*B*T)
            current_batch_size = end_idx - start_idx
            
            # 현재 배치에 대한 클래스 인덱스 추출
            subgoal_classes = subgoals.reshape(-1, C)[start_idx:end_idx].argmax(dim=-1)
            action_classes = actions.reshape(-1, C)[start_idx:end_idx].argmax(dim=-1)
            
            # 텍스트 토큰 생성
            subgoal_texts = [f"action {idx.item()}" for idx in subgoal_classes]
            action_texts = [f"action {idx.item()}" for idx in action_classes]
            
            # CLIP 토큰화 및 인코딩
            with torch.amp.autocast('cuda'):  # Updated autocast
                subgoal_tokens = clip.tokenize(subgoal_texts).to(subgoals.device)
                action_tokens = clip.tokenize(action_texts).to(actions.device)
                
                # CLIP 인코딩
                subgoal_features = self.clip.encode_text(subgoal_tokens)
                action_features = self.clip.encode_text(action_tokens)
                
                # Subgoal-Action similarity
                batch_subgoal_action_sim = F.cosine_similarity(subgoal_features, action_features, dim=-1)
                total_subgoal_action_sim += batch_subgoal_action_sim.sum()
                
                # Global goal similarity
                if start_idx == 0:  # Process global goal only once
                    global_goal_classes = global_goal.argmax(dim=-1)
                    global_goal_texts = [f"action {idx.item()}" for idx in global_goal_classes]
                    global_goal_tokens = clip.tokenize(global_goal_texts).to(global_goal.device)
                    global_goal_features = self.clip.encode_text(global_goal_tokens)
                    
                    # Expand global goal features
                    global_goal_features = global_goal_features.unsqueeze(1)  # (B, 1, clip_dim)
                    
                    # Calculate similarity with current batch of subgoals
                    batch_subgoal_goal_sim = F.cosine_similarity(
                        subgoal_features, 
                        global_goal_features.expand(-1, current_batch_size, -1),
                        dim=-1
                    )
                    total_subgoal_goal_sim += batch_subgoal_goal_sim.sum()
            
            # 메모리 즉시 해제
            del subgoal_features, action_features
            torch.cuda.empty_cache()
            
            num_batches += 1

        # Final loss computation (as tensors)
        avg_subgoal_action_sim = total_subgoal_action_sim / (S*B*T)
        avg_subgoal_goal_sim = total_subgoal_goal_sim / (S*B*T)
        
        loss_semantic = (
            (1 - avg_subgoal_action_sim) +  # subgoal-action alignment
            (1 - avg_subgoal_goal_sim)      # subgoal-global goal alignment
        )
        
        # Clear any remaining cache
        torch.cuda.empty_cache()
        
        return loss_semantic
    
    @property
    def loss_fn(self):
        if self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")


    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
          - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )


    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)



    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
              extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        return (
            posterior_mean,
            posterior_variance,
            posterior_log_variance_clipped,
        )  



    # SAMPLE from q(x_t | x_o)
    def q_sample(self, x_start, t, noise=None):
        """
        :param x_start: {B x T x C}
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return ( extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

  
    # PREDICTION and LOSSES
    def p_losses(self,
                t,
                x_0,
                obs,
                mask_all,
                mask_past,gt_goal=None,gt_goal_one_hot=None,
                noise=None):
        condition = False
        # SAMPLE x_t from q(x_t | x_o)
        x_start = x_0
        
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # sample: goal
        goal_start = gt_goal_one_hot # (16, 1816, 10)
        goal_noise = None
        goal_noise = default(goal_noise, lambda: torch.randn_like(goal_start))
        
        goal_t = self.q_sample(x_start=goal_start, t=t, noise=goal_noise)

        # OBSERVATION CONDITIONING
        obs_cond = obs * mask_past

        # SELF-CONDITIONING
        self_cond = torch.zeros_like(gt_goal_one_hot).to(gt_goal_one_hot.device)
        if torch.rand((1)) < 0.5 and self.condition_x0:
            with torch.no_grad():
                infer_goal = self.goalmodel(
                    x=goal_t, 
                    t=t, 
                    stage_masks=mask_all,
                    obs_cond=obs_cond, # frames
                    self_cond=self_cond,
                )
                self_cond = infer_goal[-1].detach()

        # REVERSE STEP
        model_out_goal = self.goalmodel(
            x=goal_t,
            t=t,
            stage_masks=mask_all,
            obs_cond=obs_cond,
            self_cond=self_cond,
        )  # S x B x T x C

        subgoal_seq = model_out_goal # (S x B x T x C)
        #goal_logits = model_out_goal.mean(dim=2, keepdim=True) # (S, B, 1, C)
        gt_goal_one_hot = gt_goal_one_hot[:, :1, :]
        # gt_goal_one_hot = repeat(gt_goal_one_hot, 'b 1 c -> s b 1 c', s=goal_logits.shape[0])
        
        #loss_goal = self.loss_fn(goal_logits, gt_goal_one_hot, reduction="none")
        
        #self_cond = torch.zeros_like(x_0).to(x_0.device)
        #self_cond = torch.zeros_like(model_out_goal[-1]).to(model_out_goal.device)
        # SELF-CONDITIONING
        if torch.rand((1)) < 0.5 and self.condition_x0:
            self_cond = torch.zeros((model_out_goal.shape[1], model_out_goal.shape[2], x_0.shape[2]+model_out_goal.shape[3]), device=model_out_goal.device)

            with torch.no_grad():
                self_cond = self.model(
                    x=x_t, 
                    t=t, 
                    stage_masks=mask_all,
                    obs_cond=obs_cond, 
                    self_cond=self_cond
                )[-1]
                self_cond = self_cond.detach()
                
        else:
            self_cond = torch.zeros_like(x_0).to(x_0.device)
            
        ## concat: 
        ## self_cond: (b,t,c)
        ## goal_logits[-1]: (b,1,c)
        #goal_repeat = goal_logits[-1].expand(-1, self_cond.shape[1], -1) # (b,1,c)->(b,t,c)
        goal_repeat = subgoal_seq[-1] # (B x T x C)
        
        self_cond = torch.cat([self_cond, goal_repeat], dim=2) # (b,t,2c)
        
        # REVERSE STEP
        model_out = self.model(
            x=x_t,
            t=t,
            stage_masks=mask_all,
            obs_cond=obs_cond,
            self_cond=self_cond,
        )  # S x B x T x C
        
        # LOSS
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_0 
        else:
            raise ValueError(f"unknown objective {self.objective}")
        

        # KL (q(x_t-1 | x_o, x_t) || p(x_t-1 | x_t))
        if self.loss_type == 'l2':
            target = repeat(target, 'b t c -> s b t c', s=model_out.shape[0])
            mask_all = torch.stack(mask_all, dim=0)

            loss = self.loss_fn(model_out, target, reduction="none")  # S x B x T x C
            loss = torch.sum(torch.mean(loss * mask_all, dim=(2, 3)))
            if gt_goal is not None:
                ### goal diffusion: goal_logits (S x B x 1 x C)
                ### action diffusion: model_out (S x B x T x C)
                loss_goal = self.semantic_consistency_loss(
                    subgoals=subgoal_seq, 
                    actions=model_out, 
                    global_goal=gt_goal_one_hot
                )
                #loss_goal = torch.sum(torch.mean(loss_goal * mask_all, dim=(2, 3)))
                loss += loss_goal

        # OUT
        return loss, rearrange(model_out, 's b t c -> s b c t')



    def forward(self, batch, *args, **kwargs):
        gt_goal = batch['goal']
        gt_goal_one_hot = batch['goal_one_hot']
        x_0 = batch['x_0']  # class labels, (16, 2996, 48) # 48 is the class number
        obs = batch['obs']  # features, (16, 2996, 2048)
        
        masks_stages = batch['masks_stages'] # 5-length (16, 2996, 1) each
        masks_stages = [mask.to(torch.bool) for mask in masks_stages] # 5-length (16, 2996, 1) each
        
        mask_past = batch['mask_past'] # (16, 2996, 1)
        mask_past = mask_past.to(torch.bool)
        mask_past = repeat(mask_past, 'b t 1 -> b t c', c=obs.shape[-1]) # (16, 2996, 2048)

        # get random diff timestep
        t = torch.randint(0, self.num_timesteps, (obs.size(0),), device=obs.device).long() # (16)
        return self.p_losses(
            t = t,
            x_0 = x_0,
            obs = obs,
            mask_past = mask_past,
            mask_all = masks_stages,
            gt_goal=gt_goal, gt_goal_one_hot=gt_goal_one_hot,
            *args, **kwargs
        )

 

    # ---------------------------------- INFERENCE (DDIM) --------------------------------------

    def model_predictions(self, x, pred_x_start_prev, t, obs, stage_masks, gt_goal=None, gt_goal_one_hot=None):
        x_t = x
        B,T,C=gt_goal_one_hot.shape
        # Given x_t, reconsturct x_0
        #self_cond_goal = torch.zeros_like(gt_goal_one_hot).to(gt_goal_one_hot.device)
        self_cond = torch.zeros_like(pred_x_start_prev).to(pred_x_start_prev.device)
        #self_cond = torch.zeros((B, T, 58), device=pred_x_start_prev.device)
        if self.condition_x0:
            self_cond = pred_x_start_prev
        
        ###################################################################################
        goal_t = gt_goal_one_hot#torch.randn((B, T, C)).to(pred_x_start_prev.device)  # e.g., (16, 1, 48) shaped random noise

        infer_goal = self.goalmodel(
            x=goal_t, 
            t=t, 
            stage_masks=stage_masks,
            obs_cond=obs,
            self_cond=torch.zeros_like(gt_goal_one_hot).to(gt_goal_one_hot.device)
        )[-1]
        
        self_cond = torch.cat([self_cond, infer_goal], dim=2)
        
        ##################################################################################
        
        # PRED
        model_output = self.model(
            x=x_t,
            t=t,
            stage_masks=stage_masks,
            obs_cond=obs,
            self_cond=self_cond,
        )[-1]
        
        if self.objective == "pred_noise":
            pred_noise = model_output 
            pred_x_start = self.predict_start_from_noise(x, t, pred_noise) * stage_masks[-1]
            
        elif self.objective == "pred_x0":
            pred_x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, pred_x_start) * stage_masks[-1]

        pred_goal_start = infer_goal
        pred_goal_noise = self.predict_noise_from_start(x, t, pred_goal_start) * stage_masks[-1]
            
        return ModelPrediction(pred_noise, pred_x_start, pred_goal_noise, pred_goal_start)



    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        pred_x_start_prev,
        t,
        t_prev,
        batch,
        if_prev=False
    ):
        

        # MODEL PRED
        preds = self.model_predictions(x=x,
                                       pred_x_start_prev=pred_x_start_prev,
                                       t=t,
                                       obs=batch['obs'] * batch['mask_past'],
                                       stage_masks=batch['mask_all'], gt_goal=batch['goal'],gt_goal_one_hot=batch['goal_one_hot'])
        pred_x_start = preds.pred_x_start
        pred_noise = preds.pred_noise

        pred_goal_start = preds.pred_goal_start
        pred_goal_noise = preds.pred_goal_noise

        # PRED X_0
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        if if_prev:
            alpha_bar_prev = extract(self.alphas_cumprod_prev, t_prev, x.shape)
        else:
            alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x.shape)
        sigma = (
                self.eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # Compute mean and var
        noise = torch.randn_like(x) 
        mean_pred = (
                pred_x_start * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * pred_noise
        )

        nonzero_mask = (1 - (t == 0).float()).reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))


        # PRED goal
        alpha_bar_goal = extract(self.alphas_cumprod, t, gt_goal_one_hot.shape)
        if if_prev:
            alpha_bar_prev_goal = extract(self.alphas_cumprod_prev, t_prev, gt_goal_one_hot.shape)
        else:
            alpha_bar_prev_goal = extract(self.alphas_cumprod, t_prev, gt_goal_one_hot.shape)
        sigma_goal = (
                self.eta
                * torch.sqrt((1 - alpha_bar_prev_goal) / (1 - alpha_bar_goal))
                * torch.sqrt(1 - alpha_bar_goal / alpha_bar_prev_goal)
        )

        # Compute mean and var
        noise_goal = torch.randn_like(x) 
        mean_pred_goal = (
                pred_goal_start * torch.sqrt(alpha_bar_prev_goal)
                + torch.sqrt(1 - alpha_bar_prev_goal - sigma_goal ** 2) * pred_goal_noise
        )

        nonzero_mask_goal = (1 - (t == 0).float()).reshape(gt_goal_one_hot.shape[0], *((1,) * (len(gt_goal_one_hot.shape) - 1)))
        return mean_pred + nonzero_mask * sigma * noise, pred_x_start, mean_pred_goal+nonzero_mask_goal*sigma_goal*noise_goal, pred_goal_start

   

    @torch.no_grad()
    def p_sample_loop_with_input(
        self,
        batch,
        init_rand=None,
        n_diffusion_steps=-1,
    ):
        
        # INPUT
        device = self.betas.device
        x_0_pred = torch.zeros_like(batch["x_0"]).to(batch["x_0"].device)  # only used for shape

    
        # INIT PREDICTION (normal distr noise)
        pred = torch.randn_like(x_0_pred, device=device) if init_rand is None else init_rand  # BS x T x C
        init_noise = pred.clone()
        pred = pred.contiguous()


        # SAMPLE
        assert n_diffusion_steps == len(self.ddim_timestep_seq) # 50
        
        # Resample (DDIM)
        for t in tqdm(
            reversed(range(0, n_diffusion_steps)),
            desc="Resampled sampling loop time step",
            total=n_diffusion_steps,
            position=0,
            leave=True
        ):

            batched_times = torch.full((pred.shape[0],), self.ddim_timestep_seq[t], device=pred.device, dtype=torch.long)
            if t == 0:
                batched_times_prev = torch.full((pred.shape[0],), 0, device=device, dtype=torch.long)
                pred, x_0_pred = self.p_sample_ddim(
                    x=pred, 
                    pred_x_start_prev=x_0_pred,
                    t=batched_times, 
                    t_prev=batched_times_prev, 
                    batch=batch,
                    if_prev=True
                )
            else:
                batched_times_prev = torch.full((pred.shape[0],), self.ddim_timestep_seq[t-1], device=device, dtype=torch.long)
                pred, x_0_pred = self.p_sample_ddim(
                    x=pred,
                    pred_x_start_prev=x_0_pred, 
                    t=batched_times,
                    t_prev=batched_times_prev,
                    batch=batch
                )
        return pred, init_noise



    ''' Actual inference step '''
    def predict(
        self,
        x_0,
        obs,
        mask_past,
        masks_stages,
        *,
        n_samples=2,
        return_noise=False,
        n_diffusion_steps=-1,
        goal=None,gt_goal_one_hot=None,
    ):
        
        # Initialize observation
        
        obs = repeat(obs, "b t c -> (s b) t c", s=n_samples)
        x_0 = repeat(x_0, "b t c -> (s b) t c ", s=n_samples)
        mask_past = repeat(mask_past, "b t 1 -> (s b) t c", s=n_samples, c=obs.shape[-1])
        masks_stages = [repeat(mask.to(torch.bool), "b t c -> (s b) t c", s=n_samples) for mask in masks_stages]
        goal = repeat(goal, "b c -> (s b) c", s=n_samples)
        gt_goal_one_hot = repeat(gt_goal_one_hot, "b t c -> (s b) t c", s=n_samples)
        # Sample from the diffusion model
        
        x_out, init_noise = self.p_sample_loop_with_input(
            batch={
                "x_0": x_0,  # only used for shape
                "obs": obs,
                "mask_past": mask_past.to(torch.bool),
                "mask_all": masks_stages,
                "goal": goal,
                "goal_one_hot":gt_goal_one_hot,
            },
            init_rand=None,
            n_diffusion_steps=n_diffusion_steps
        )
          
        # Return
        init_noise = init_noise[0]
        if return_noise:
            assert n_samples == 1
            return x_out, init_noise
        return rearrange(x_out, "(s b) t c -> s b c t", s=n_samples)


