from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
device = "cuda"
import time
import inspect
# -------------------------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        #Multi-headed Design
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # = (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # = (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # = (B, nh, T, hs)

        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        #att = F.softmax(att, dim=-1)
        #y = att @ v
        #Instead of Classic Attention we use Flash Attention to improve training speed
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        #output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) 

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 #max sequence length
    vocab_size: int = 50257 # number of tokens
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight


    def forward(self, idx, targets=None):
        #idx is shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        #forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        loss = None
        logits = self.lm_head(x) #(B, T, vocab_size)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Any parameters that is 2D will be weight decayed, otherwise no.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused_available)
        return optimizer
# -------------------------------------------------------------------------------------------------------------
import tiktoken

class DataLoader_Lite:
    def __init__(self, tokens, B, T, shuffle=True):
        self.B = B
        self.T = T
        self.tokens = tokens
        self.n = len(self.tokens)
        print(f"Loaded {self.n} tokens")
        self.max_pos = self.n - (B * T + 1)
        self.shuffle = shuffle
        self.order = torch.arange(0, self.max_pos, B * T)
        if self.shuffle:
            self.order = self.order[torch.randperm(len(self.order))]
        self.current_index = 0

    def next_batch(self):
        if self.current_index >= len(self.order):
            # Start a new epoch
            self.current_index = 0
            if self.shuffle:
                self.order = self.order[torch.randperm(len(self.order))]
            else:
                self.order = torch.arange(0, self.max_pos, self.B * self.T)
        start = self.order[self.current_index]
        buf = self.tokens[start : start + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_index += 1
        return x, y
#-------------------------------------------------------------------------------------------------------------------

# Control variable to switch between training and inference
TRAINING = False  # Set to True to train, False to perform inference without retraining

if TRAINING:
    # ---------------------------------------------------------------------------------------------------------
    # Load and Prepare the Dataset
    with open('tinyS.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    print(f"Loaded {len(tokens)} tokens")

    # Split into training and validation sets
    train_size = int(0.9 * len(tokens))
    train_tokens = tokens[:train_size]
    val_tokens = tokens[train_size:]

    # Create DataLoaders
    B = 2  # Batch size
    T = 1024  # Sequence length
    train_loader = DataLoader_Lite(train_tokens, B=B, T=T)
    val_loader = DataLoader_Lite(val_tokens, B=B, T=T)

    # ---------------------------------------------------------------------------------------------------------
    # Initialize the Model
    # Load the pre-trained GPT-2 model
    model = GPT.from_pretrained('gpt2')
    model.to(device)

    # ---------------------------------------------------------------------------------------------------------
    # Adjust the Optimizer and Learning Rate Scheduler
    learning_rate = 5e-5  # Smaller learning rate for fine-tuning
    optimizer = model.configure_optimizers(weight_decay=0.05, learning_rate=learning_rate, device=device)

    # Learning rate scheduler
    max_lr = learning_rate
    min_lr = max_lr * 0.1
    warmup_steps = 20
    max_steps = 150  # Adjust based on your training needs

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    # ---------------------------------------------------------------------------------------------------------
    # Training Loop with Validation
    total_batch_size = B * T * 8 
    grad_accum_steps = total_batch_size // (B * T)
    print(f"Total desired batch size: {total_batch_size}")
    print(f"=> Calculated gradient accumulation steps: {grad_accum_steps}")

    torch.set_float32_matmul_precision('high')

    for step in range(max_steps):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / dt
        print(f"Step {step:4d} | Loss: {loss_accum.item():.6f} | LR: {lr:.4e} | Norm: {norm:.4f} | Time: {dt*1000:.2f}ms | Tokens/sec: {tokens_per_sec:.2f}")

        # Validation
        if step % 5 == 0 or step == max_steps - 1:
            model.eval()
            val_loss = 0.0
            val_loader.current_index = 0
            with torch.no_grad():
                for _ in range(len(val_loader.order)):
                    x_val, y_val = val_loader.next_batch()
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits_val, loss_val = model(x_val, y_val)
                    val_loss += loss_val.item()
            avg_val_loss = val_loss / len(val_loader.order)
            print(f"Validation Loss: {avg_val_loss:.6f}")

    # ---------------------------------------------------------------------------------------------------------
    # Save the Fine-Tuned Model
    torch.save(model.state_dict(), 'fine_tuned_gpt2.pth')
    print("Model saved to fine_tuned_gpt2.pth")

else:
    # ---------------------------------------------------------------------------------------------------------
    # Inference Phase: Load the Saved Model and Generate Text
    enc = tiktoken.get_encoding('gpt2')

    # Initialize the model architecture
    config = GPTConfig(
        vocab_size=50257,  
        block_size=1024,   
        n_layer=12,
        n_head=12,
        n_embd=768
    )
    model = GPT(config)
    model.load_state_dict(torch.load('fine_tuned_gpt2.pth'))
    model.to(device)
    model.eval()

    def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, top_k=50):
        model.eval()
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            for _ in range(max_length):
                logits, _ = model(tokens)
                logits = logits[:, -1, :] / temperature
                filtered_logits = top_k_logits(logits, top_k=top_k)
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)
        generated_text = tokenizer.decode(tokens[0].cpu().numpy())
        return generated_text
    def top_k_logits(logits, top_k):
        v, ix = torch.topk(logits, top_k)
        out = logits.clone()
        out[out < v[:, -1, None]] = -float('Inf')
        return out

    # Example usage
    prompt = "Let me speak"
    generated_text = generate_text(model, enc, prompt, max_length=100)
    print(generated_text)