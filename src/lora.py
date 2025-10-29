# lm_conditioner.py
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

class LLMConditioner(nn.Module):
    def __init__(self, lm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 prefix_len=16, r=4, alpha=16, dropout=0.05, fourbit=True, device_map={"": "cuda:0"}):
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
            self.lm = AutoModelForCausalLM.from_pretrained(lm_name, quantization_config=bnb, device_map={"": "cuda:0"})

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

    # inside LLMConditioner class
    @torch.no_grad()
    def generate_from_subgoals(
        self,
        s_bt: torch.Tensor,                   # (B, 512) or (N, 512)
        prompt: str = "Subgoal: ",
        max_new_tokens: int = 32,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        stop_after_eos: bool = True,
    ) -> list[str]:
        """
        Returns a list of generated strings, one per row in s_bt.
        """
        self.eval()
        emb_layer  = self.lm.get_input_embeddings()
        emb_device = emb_layer.weight.device

        # (1) subgoal -> soft prefix
        s_bt = s_bt.to(emb_device)
        P = self.sg2prefix(s_bt).view(s_bt.size(0), self.prefix_len, self.d_model)   # (B, m, d)

        # (2) prompt → input embeddings
        prompts = [prompt] * s_bt.size(0)
        tok_prompt = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        tok_prompt = {k: v.to(emb_device) for k, v in tok_prompt.items()}
        emb_prompt = emb_layer(tok_prompt["input_ids"])                               # (B, Lp, d)

        # (3) build inputs_embeds & attn_mask
        inputs_embeds = torch.cat([P, emb_prompt], dim=1)                             # (B, m+Lp, d)
        attn_prefix   = torch.ones(s_bt.size(0), self.prefix_len, dtype=torch.long, device=emb_device)
        attn_mask     = torch.cat([attn_prefix, tok_prompt["attention_mask"]], dim=1) # (B, m+Lp)

        # (4) generate
        gen_ids = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
            use_cache=True,
            return_dict_in_generate=False,
        )
        # With inputs_embeds, HF returns only the newly generated token ids.
        texts = self.tok.batch_decode(gen_ids, skip_special_tokens=True)

        if stop_after_eos and self.tok.eos_token is not None:
            eos = self.tok.eos_token
            texts = [t.split(eos)[0] for t in texts]

        # optional cleanup
        texts = [t.strip() for t in texts]
        return texts

    @torch.no_grad()
    def generate_from_subgoal_seq(self, s_sbt: torch.Tensor, **gen_kwargs):
        """
        s_sbt: (S, 1, T, 512) subgoal vectors
        returns: list[list[str]] with shape (S, T)
        """
        emb_device = self.lm.get_input_embeddings().weight.device
        S, B, T, D = s_sbt.shape
        assert B == 1, "Expected batch=1 in (S,1,T,512)"

        # flatten to (S*T, 512)
        flat = s_sbt.reshape(S*T, D).to(emb_device)

        # call the base generator
        texts_flat = self.generate_from_subgoals(flat, **gen_kwargs)

        # regroup back to (S, T)
        texts = [texts_flat[i*T:(i+1)*T] for i in range(S)]
        return texts


    def save_subgoal_texts(self, s_sbt, out_dir, idx, **gen_kwargs):
        """
        s_sbt: (S,1,T,512) subgoal vectors
        out_dir: folder path to save .txt files
        idx: global index (e.g. video id)
        """
        os.makedirs(out_dir, exist_ok=True)
        texts_all = self.generate_from_subgoal_seq(s_sbt, **gen_kwargs)  # list of S × T strings

        S, _, T, _ = s_sbt.shape
        for s in range(S):
            fname = os.path.join(out_dir, f"{idx}_{s}.txt")
            with open(fname, "w", encoding="utf-8") as f:
                for t in range(T):
                    f.write(texts_all[s][t] + "\n")
            print(f"✅ wrote {fname}")
