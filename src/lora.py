# lm_conditioner.py
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class LLMConditioner(nn.Module):
    def __init__(self, lm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 prefix_len=16, r=4, alpha=16, dropout=0.05, fourbit=True, device_map={"": "cuda:1"}):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(lm_name, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        if fourbit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.lm = AutoModelForCausalLM.from_pretrained(
                lm_name, quantization_config=bnb, device_map=device_map
            )
            self.lm = prepare_model_for_kbit_training(self.lm)
        else:
            #self.lm = AutoModelForCausalLM.from_pretrained(lm_name, device_map=device_map)
            self.lm = AutoModelForCausalLM.from_pretrained(lm_name, quantization_config=bnb, device_map={"": "cuda:1"})

        self.lm.gradient_checkpointing_enable()

        # LoRA: q_proj만
        lcfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=["q_proj"], bias="none", task_type="CAUSAL_LM"
        )
        self.lm = get_peft_model(self.lm, lcfg)

        self.d_model   = self.lm.config.hidden_size
        self.prefix_len = prefix_len
        self.sg2prefix = nn.Sequential(
            nn.Linear(512, 4*self.d_model), nn.GELU(),
            nn.Linear(4*self.d_model, prefix_len*self.d_model)
        )

        emb_device = self.lm.get_input_embeddings().weight.device
        self.sg2prefix.to(emb_device)

    def lm_loss_from_subgoals(self, s_bt, target_texts):
        """
        s_bt: (B_sel, 512)
        target_texts: list[str] with len == B_sel
        """
        emb_layer  = self.lm.get_input_embeddings()
        emb_device = emb_layer.weight.device

        # 1) subgoal → soft prefix (B, m, d)
        s_bt = s_bt.to(emb_device)
        P = self.sg2prefix(s_bt).view(s_bt.size(0), self.prefix_len, self.d_model)

        # 2) prompt 토큰 → 임베딩
        prompts   = ["Subgoal: "] * s_bt.size(0)
        tok_prompt = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        tok_prompt = {k: v.to(emb_device) for k, v in tok_prompt.items()}
        emb_prompt = emb_layer(tok_prompt["input_ids"])      # (B, Lp, d)
        labels_tok = self.tok(target_texts, return_tensors="pt", padding=True, add_special_tokens=False)
        labels_tok = {k: v.to(emb_device) for k, v in labels_tok.items()}
        emb_label  = emb_layer(labels_tok["input_ids"])           # (B, Ly, d)

        # 4) inputs_embeds = [prefix | prompt | label]
        inputs_embeds = torch.cat([P, emb_prompt, emb_label], dim=1)   # (B, m+Lp+Ly, d)

        # attention mask도 동일 길이로
        attn_prefix = torch.ones(s_bt.size(0), self.prefix_len, dtype=torch.long, device=emb_device)
        attn_mask   = torch.cat([attn_prefix, tok_prompt["attention_mask"], labels_tok["attention_mask"]], dim=1)

        # 5) labels도 동일 길이로: prefix+prompt 구간은 -100로 마스킹, label 구간만 실제 라벨
        B, total_len, _ = inputs_embeds.shape
        labels_full = torch.full((B, total_len), -100, dtype=torch.long, device=emb_device)
        # prefix+prompt 길이
        pp_len = self.prefix_len + tok_prompt["attention_mask"].size(1)
        labels_full[:, pp_len:pp_len + labels_tok["input_ids"].size(1)] = labels_tok["input_ids"]

        # 6) forward
        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels_full)
        return out.loss
