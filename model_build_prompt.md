# Task: Build the InfiniteRace World Model

## Context

You have been provided with two reference documents:

- **World Model Architecture Specification v0.3** — defines the model architecture, all tensor shapes, training strategy, and the full interface contract
- **DEMO_SPEC.md v1.0** — defines the CueEngine that is already implemented and produces all model inputs

The CueEngine (`demo/cue_engine.py`) is already built and running. Your task is to build the world model that consumes its output. Do not modify CueEngine or any demo code. Your deliverable is a standalone Python package `world_model/` that can be plugged into the demo in Phase 1 integration.

---

## What you are building

A real-time latent diffusion model that takes three inputs per frame and produces one output:

- **Input 1 — Warped previous frame** `(B, 3, 256, 256)` float32 `[-1, 1]`
- **Input 2a — Anchor crop** `(B, 3, 256, 256)` float32 `[-1, 1]`
- **Input 2b — Anchor pos_map** `(B, 256, 256, 2)` float32 radians
- **Input 3 — Action vector** `(B, 3)` float32 `[-1, 1]`
- **Output — Keyframe** `(B, 3, 256, 256)` float32 `[-1, 1]`

Target: **2-step LCM inference in under 50ms on an RTX 3070**. This is not an offline model. Every architectural and implementation decision must be made with this constraint as the primary filter.

---

## Package structure

```
world_model/
├── __init__.py
├── model.py          # InfiniteRaceWorldModel — full model class
├── vae.py            # Shared VAE encoder/decoder
├── unet.py           # Thin convolutional UNet backbone
├── transformer.py    # Transformer bottleneck (cross-attn + AdaLN)
├── pos_encoding.py   # Spherical sinusoidal positional encoding
├── action_mlp.py     # Action vector → AdaLN scale/shift
├── lcm.py            # LCM consistency distillation scheduler + sampler
├── train.py          # Training loop (Levels 2 and 3)
├── distill.py        # Level 4 LCM distillation
├── dataset.py        # Street view sequence dataset
├── losses.py         # L1 masked + LPIPS + distillation losses
├── inference.py      # Real-time inference runner (async thread)
├── queue_interface.py# wm_queue consumer — Phase 1 integration point
└── config.py         # All hyperparameters and constants
```

---

## 1. VAE — `vae.py`

### Architecture

A standard latent diffusion VAE with **8× spatial compression**:
- Input: `(B, 3, 256, 256)` → Output latent: `(B, 4, 32, 32)`
- Encoder: 4 downsampling stages, each doubling channels and halving spatial. Conv → GroupNorm → SiLU. Residual blocks at each scale.
- Bottleneck: 2 residual blocks with attention.
- Decoder: mirrors encoder with transposed convolutions.
- Latent channels: 4 (standard for SD-style VAE).

### Initialization

Initialize from **Stable Diffusion 1.5 VAE weights** (`stabilityai/sd-vae-ft-mse` from HuggingFace). The SD VAE already operates at 8× compression on 512×512 → 64×64. For 256×256 → 32×32, either:
- Load the full SD VAE and use it directly (it handles arbitrary input sizes), OR
- Load and strip to 3 downsampling stages if the 4th stage adds unnecessary compute at 256×256 input

The VAE is **frozen during all training phases**. Only the UNet, transformer, action MLP, and distillation components train.

### Interface

```python
class VAE(nn.Module):
    def encode(self, x: Tensor) -> Tensor:
        """x: (B, 3, H, W) float32 [-1,1] → z: (B, 4, H//8, W//8) float32"""

    def decode(self, z: Tensor) -> Tensor:
        """z: (B, 4, H//8, W//8) float32 → x: (B, 3, H, W) float32 [-1,1]"""
```

No reparameterization at inference — encode returns the mean directly.

---

## 2. Spherical Positional Encoding — `pos_encoding.py`

This is a critical component. Standard 2D sinusoidal encodings are insufficient because the anchor crop is a rectilinear projection of a sphere — the mapping from pixel position to angular position is non-linear. CueEngine provides the exact per-pixel world-space coordinates as `anchor_pos_map`.

### Input

`anchor_pos_map: (B, 256, 256, 2)` float32 — per-pixel `[azimuth_rad, elevation_rad]` in world space. After VAE encoding, this is downsampled to `(B, 32, 32, 2)` to match the latent resolution.

### Encoding

For each pixel position `(az, el)`, compute a sinusoidal embedding of dimension `d_model`:

```python
def spherical_sinusoidal(pos_map: Tensor, d_model: int) -> Tensor:
    """
    pos_map: (B, H, W, 2) float32 — [azimuth_rad, elevation_rad]
    Returns: (B, H*W, d_model) float32 — positional embeddings
    """
    B, H, W, _ = pos_map.shape
    az  = pos_map[..., 0]  # (B, H, W)
    el  = pos_map[..., 1]  # (B, H, W)

    # Encode azimuth and elevation separately, each using d_model//2 dimensions
    # Use the standard sinusoidal formula with geometric frequency spacing:
    #   PE[2i]   = sin(pos / 10000^(2i/d))
    #   PE[2i+1] = cos(pos / 10000^(2i/d))
    # Apply independently to az and el, then concatenate.

    # Flatten spatial dims, return (B, H*W, d_model)
```

The resulting embedding is **added** to the flattened anchor latent `z_anchor` before it is used as keys and values in cross-attention. This tells the transformer exactly where on the sphere each anchor latent token came from.

### Downsampling pos_map

Use `F.interpolate(pos_map.permute(0,3,1,2), size=(32,32), mode='bilinear', align_corners=False).permute(0,2,3,1)` to downsample from 256×256 to 32×32. Do not use nearest-neighbor — azimuth and elevation are continuous angular values and benefit from interpolation.

---

## 3. Action MLP — `action_mlp.py`

Encodes the 3D action vector into AdaLN scale and shift parameters for each transformer layer.

### Architecture

```python
class ActionMLP(nn.Module):
    """
    input:  (B, 3) float32 [-1, 1]  — [speed_norm, delta_heading_norm, steer]
    output: list of (scale, shift) tuples, one per transformer layer
            each scale/shift: (B, d_model) float32
    """
    def __init__(self, d_model: int, n_layers: int):
        # 3-layer MLP: 3 → 256 → 512 → n_layers * 2 * d_model
        # SiLU activations
        # Output split into n_layers pairs of (scale, shift)
        # Initialize final layer weights to zero — no action effect at start of training
```

Zero-initialize the final linear layer so the model starts training without any action conditioning influence, then learns it progressively. This prevents the action signal from destabilizing early training.

---

## 4. Transformer Bottleneck — `transformer.py`

The core reasoning module. Operates entirely at 32×32 latent resolution (1,024 tokens).

### Architecture

```python
class TransformerBottleneck(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 4, n_layers: int = 2):
        ...

    def forward(
        self,
        z_warp: Tensor,      # (B, 4, 32, 32) — warped frame latent
        z_anchor: Tensor,    # (B, 4, 32, 32) — anchor crop latent
        pos_enc: Tensor,     # (B, 1024, d_model) — spherical pos embedding
        adaln_params: list,  # [(scale, shift), ...] per layer from ActionMLP
    ) -> Tensor:             # (B, 4, 32, 32) — refined latent
```

### Implementation

```
Step 1: Project z_warp and z_anchor to d_model
  z_warp_flat   = flatten(z_warp)   → (B, 1024, d_model)  via linear proj
  z_anchor_flat = flatten(z_anchor) → (B, 1024, d_model)  via linear proj

Step 2: Add spherical positional encoding to anchor tokens only
  z_anchor_pos = z_anchor_flat + pos_enc  → (B, 1024, d_model)

Step 3: For each transformer layer i:
  a. Apply AdaLN to z_warp_flat using adaln_params[i]:
       z_warp_normed = layernorm(z_warp_flat)
       z_warp_flat   = z_warp_normed * (1 + scale_i) + shift_i

  b. Cross-attention (z_warp_flat attends to z_anchor_pos):
       query = z_warp_flat     (B, 1024, d_model)
       key   = z_anchor_pos    (B, 1024, d_model)
       value = z_anchor_pos    (B, 1024, d_model)
       z_warp_flat = z_warp_flat + cross_attn(query, key, value)

  c. Feed-forward:
       z_warp_flat = z_warp_flat + FFN(layernorm(z_warp_flat))

Step 4: Project back to latent channel dim and reshape
  z_out = unflatten(linear_proj(z_warp_flat)) → (B, 4, 32, 32)
```

**No self-attention** on the warped frame tokens. The residual correction task is spatially local — cross-attention to the anchor is sufficient. Self-attention across 1,024 tokens doubles compute with negligible benefit for this task.

**Attention implementation**: Use `torch.nn.MultiheadAttention` with `batch_first=True`. Set `kdim` and `vdim` to `d_model`. No causal masking — this is not autoregressive.

---

## 5. Thin UNet Backbone — `unet.py`

The convolutional encoder and decoder surrounding the transformer bottleneck. Handles spatial encoding and decoding. The transformer replaces the middle blocks of a standard UNet.

### Architecture

```
Encoder:
  256×256 input (3 channels)
  → conv 3×3, channels: 3→64
  → ResBlock(64) + Downsample(64→128)   # 128×128
  → ResBlock(128) + Downsample(128→256) # 64×64
  → ResBlock(256) + Downsample(256→512) # 32×32
  → linear projection: 512→d_model per spatial position
  → (B, d_model, 32, 32) → flatten → (B, 1024, d_model)

Bottleneck: TransformerBottleneck (d_model=512, n_heads=4, n_layers=2)

Decoder:
  (B, 1024, d_model) → unflatten → (B, d_model, 32, 32)
  → linear projection: d_model→512
  → ResBlock(512) + Upsample(512→256)   # 64×64
  → ResBlock(256) + Upsample(256→128)   # 128×128
  → ResBlock(128) + Upsample(128→64)    # 256×256
  → conv 3×3: 64→latent_channels (4)
```

**No skip connections** between encoder and decoder. Skip connections preserve fine spatial detail across the entire image — useful for full scene generation, wasteful when correcting a small residual. The model should only deviate from the warped frame where the anchor contradicts it.

**ResBlock**: Conv 3×3 → GroupNorm(8) → SiLU → Conv 3×3 → GroupNorm(8) → residual add. Standard pre-activation residual block.

**Downsample**: Conv 3×3 stride 2, or `nn.AvgPool2d(2)` followed by channel projection.

**Upsample**: `nn.Upsample(scale_factor=2, mode='nearest')` followed by Conv 3×3. Do not use transposed convolutions — they produce checkerboard artifacts.

### Initialization

Load encoder and decoder weights from a **pretrained Stable Diffusion 1.5 UNet** (`runwayml/stable-diffusion-v1-5`) where architecturally compatible. Specifically:
- The SD UNet input conv layer (handles 4 latent channels) is NOT used here — our UNet takes the image directly before VAE encoding. Use random initialization for the first conv.
- The SD UNet's down/up blocks use a similar ResBlock structure — load these weights where channel dimensions match, skip where they don't.
- The SD UNet's middle blocks (the transformer section) are NOT loaded — this is replaced by our custom TransformerBottleneck.

Provide a `load_pretrained_partial(sd_unet_state_dict)` method that performs this selective weight loading with logging of what was and was not loaded.

---

## 6. Full Model — `model.py`

```python
class InfiniteRaceWorldModel(nn.Module):
    def __init__(self, config: ModelConfig):
        self.vae        = VAE()          # frozen
        self.unet       = ThinUNet()     # trainable (partial SD init)
        self.action_mlp = ActionMLP()    # trainable from scratch
        self.pos_enc    = SphericalSinusoidal()  # no parameters

    def forward(
        self,
        cue1: Tensor,      # (B, 3, 256, 256) float32 [-1,1] — warped frame
        cue2: Tensor,      # (B, 3, 256, 256) float32 [-1,1] — anchor crop
        pos_map: Tensor,   # (B, 256, 256, 2) float32 radians
        action: Tensor,    # (B, 3) float32 [-1,1]
        noise_level: float = 0.0,  # for LCM training: amount of noise added to input
    ) -> Tensor:           # (B, 3, 256, 256) float32 [-1,1]

        # 1. Encode inputs to latent space (VAE frozen — no_grad)
        with torch.no_grad():
            z_warp   = self.vae.encode(cue1)   # (B, 4, 32, 32)
            z_anchor = self.vae.encode(cue2)   # (B, 4, 32, 32)

        # 2. Add noise to z_warp for training (simulates denoising task)
        if noise_level > 0:
            z_warp = z_warp + torch.randn_like(z_warp) * noise_level

        # 3. Compute spherical positional encoding for anchor
        pos_map_small = downsample_pos_map(pos_map, size=32)  # (B, 32, 32, 2)
        pos_enc = self.pos_enc(pos_map_small)  # (B, 1024, d_model)

        # 4. Compute AdaLN parameters from action vector
        adaln_params = self.action_mlp(action)  # [(scale, shift), ...]

        # 5. UNet forward: encode, bottleneck, decode
        z_out = self.unet(z_warp, z_anchor, pos_enc, adaln_params)  # (B, 4, 32, 32)

        # 6. Decode to pixel space
        with torch.no_grad():
            out = self.vae.decode(z_out)  # (B, 3, 256, 256) [-1,1]

        return out

    @torch.inference_mode()
    def infer(self, cue1, cue2, pos_map, action, steps: int = 2) -> Tensor:
        """Production inference path: LCM multi-step denoising."""
        # See lcm.py for implementation
        return self.lcm_sampler.sample(self, cue1, cue2, pos_map, action, steps)
```

---

## 7. LCM Consistency Distillation — `lcm.py`

This is what makes 2-step inference possible. Implement the Latent Consistency Model distillation procedure.

### LCM Sampler (inference)

```python
class LCMSampler:
    def __init__(self, num_train_timesteps: int = 1000, beta_schedule: str = "scaled_linear"):
        # Initialize noise schedule
        # Precompute alphas_cumprod, sigmas for all timesteps

    def sample(
        self,
        model: InfiniteRaceWorldModel,
        cue1: Tensor,
        cue2: Tensor,
        pos_map: Tensor,
        action: Tensor,
        steps: int = 2,
    ) -> Tensor:
        """
        LCM multi-step sampling.

        Steps:
        1. Start from z_warp (already encoded warped frame) plus small noise
        2. Map to LCM timestep schedule (steps=2 → timesteps e.g. [801, 401])
        3. For each timestep t:
           a. Model predicts clean latent x0_pred = model.forward(noisy_z, ...)
           b. Re-noise to next timestep using LCM consistency update
        4. Decode final latent
        """
```

**Key LCM insight for this use case**: Unlike text-to-image LCM where you start from pure noise, here you start from `z_warp` (the encoded warped previous frame, already ~85-90% correct) plus a small amount of noise calibrated to the timestep. The LCM's job is to find the consistent clean image in 2 steps from this informed starting point, not from pure noise. This dramatically reduces the number of steps needed.

### LCM Training Loss

```python
def lcm_consistency_loss(
    model: InfiniteRaceWorldModel,
    z_warp: Tensor,     # already-encoded warped frame
    z_target: Tensor,   # already-encoded target (anchor crop of next frame)
    cue2: Tensor,
    pos_map: Tensor,
    action: Tensor,
    t1: Tensor,         # current timestep
    t2: Tensor,         # next timestep (t2 < t1)
    w: float = 12.0,    # CFG guidance weight
) -> Tensor:
    """
    LCM consistency distillation loss.
    Forces model(noisy_z at t1) ≈ model(noisy_z at t2) in latent space.
    """
```

Implement following the LCM paper (Song et al., 2023). Use a **teacher model** (EMA copy of the student) for the t2 target. The consistency loss is the LPIPS distance in latent space between the two predictions.

---

## 8. Losses — `losses.py`

Three loss components applied during main training (Levels 2 and 3). The distillation loss is applied separately in Level 4.

### L1 Masked Residual Loss

```python
def masked_l1_loss(pred: Tensor, target: Tensor, warp: Tensor) -> Tensor:
    """
    pred:   (B, 3, 256, 256) predicted output
    target: (B, 3, 256, 256) ground truth (anchor crop of next frame)
    warp:   (B, 3, 256, 256) warped previous frame (the prior)

    Loss is weighted by per-pixel warp error:
    - Where warp ≈ target (reliable regions): low loss weight → don't punish the model for staying close to warp
    - Where warp ≠ target (error regions): high loss weight → force the model to correct

    weight = |warp - target| / (|warp - target|.mean() + eps)
    weight = clamp(weight, 0.1, 5.0)  # prevent extreme weighting
    return (weight * |pred - target|).mean()
    """
```

This masked formulation is important: it tells the model that staying close to the warped frame in reliable regions is correct behavior, and only forces correction where the warp demonstrably failed.

### Perceptual Loss (LPIPS)

```python
class PerceptualLoss(nn.Module):
    def __init__(self):
        # Use torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity
        # or implement using VGG16 feature layers: relu2_2, relu3_3, relu4_3
        # Network frozen — used as feature extractor only

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Both in [-1, 1]. Rescale to [0, 1] for VGG.
        # Return mean LPIPS distance
```

### Combined Training Loss

```python
def training_loss(pred, target, warp, lpips_fn, step):
    l1   = masked_l1_loss(pred, target, warp)
    perc = lpips_fn(pred, target)

    # Ramp up perceptual loss weight over first 10k steps
    # (start with L1 only to establish stable reconstruction, then add perceptual)
    perc_weight = min(0.5, 0.5 * step / 10000)

    return l1 + perc_weight * perc
```

---

## 9. Dataset — `dataset.py`

Training data is constructed by running CueEngine against recorded street view sequences with simulated player trajectories.

### Data Format

The training dataset consists of pre-extracted CueData tuples stored as numpy `.npz` files. Each file contains one training sample:

```python
{
    'warped_frame':       np.ndarray (256, 256, 3) uint8 BGR,  # Cue 1
    'anchor_crop':        np.ndarray (256, 256, 3) uint8 BGR,  # Cue 2a
    'anchor_pos_map':     np.ndarray (256, 256, 2) float32,    # Cue 2b
    'action_vector_norm': np.ndarray (3,) float32,             # Cue 3
    'target_frame':       np.ndarray (256, 256, 3) uint8 BGR,  # ground truth output
}
```

`target_frame` is the `anchor_crop` of the **next** frame in the sequence — what the model should produce given the current cues.

### Dataset Class

```python
class StreetViewDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train'):
        # Scan data_dir for all .npz files
        # Split 90/10 train/val deterministically by file hash

    def __getitem__(self, idx):
        sample = np.load(self.files[idx])
        # Normalize images from uint8 BGR to float32 RGB [-1, 1]:
        #   rgb = bgr[:, :, ::-1].astype(float32) / 127.5 - 1.0
        #   then permute to (3, 256, 256)
        # pos_map: permute from (256, 256, 2) to (2, 256, 256) for DataLoader,
        #   then re-permute back to (B, 256, 256, 2) in collate or model.forward
        return {
            'cue1':    torch.from_numpy(warped_rgb),    # (3, 256, 256)
            'cue2':    torch.from_numpy(anchor_rgb),    # (3, 256, 256)
            'pos_map': torch.from_numpy(pos_map),       # (2, 256, 256)
            'action':  torch.from_numpy(action_vec),    # (3,)
            'target':  torch.from_numpy(target_rgb),    # (3, 256, 256)
        }
```

### Data Generation Script

Provide `generate_dataset.py` — a script that runs CueEngine against a provided panorama dataset folder and a recorded or procedurally generated trajectory, saving `.npz` files to a specified output directory. This script imports `CueEngine` and `CueData` directly from `demo/cue_engine.py`.

```python
# generate_dataset.py
# Usage: python generate_dataset.py --panorama_dir gsv_data/ --output_dir training_data/ --n_samples 50000

from demo.cue_engine import CueEngine, CueData
from demo.loader import load_scene

# Simulate player trajectories across the node graph
# For each simulated step: call cue_engine.update(), save npz
```

---

## 10. Training Loop — `train.py`

### Training Phases

**Phase 1 — Not implemented here**: SD pretrained weight loading is handled in `vae.py` and `unet.py` at initialization.

**Phase 2 — Large video pretraining**: Train on nuScenes / Waymo driving video. Use optical flow between consecutive frames to derive warped_frame approximations. Action vectors from ego-motion estimation. This phase teaches temporal priors. Run for ~100k steps.

**Phase 3 — Street view fine-tuning**: Train on data from `generate_dataset.py` against real panorama sequences. Run for ~50k steps.

```python
def train(config: TrainConfig):
    model   = InfiniteRaceWorldModel(config.model)
    dataset = StreetViewDataset(config.data_dir)
    loader  = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                         num_workers=4, pin_memory=True)
    optim   = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optim, T_max=config.steps)
    lpips_fn  = PerceptualLoss().cuda().eval()
    scaler    = GradScaler()  # fp16 mixed precision

    for step, batch in enumerate(cycle(loader)):
        cue1    = batch['cue1'].cuda()     # (B, 3, 256, 256)
        cue2    = batch['cue2'].cuda()
        pos_map = batch['pos_map'].permute(0, 2, 3, 1).cuda()  # (B, 256, 256, 2)
        action  = batch['action'].cuda()
        target  = batch['target'].cuda()

        with autocast():
            pred = model(cue1, cue2, pos_map, action)
            loss = training_loss(pred, target, cue1, lpips_fn, step)

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        scheduler.step()

        if step % 500 == 0:
            log_metrics(step, loss, pred, target)
        if step % 5000 == 0:
            save_checkpoint(model, optim, step, config.checkpoint_dir)
```

### Logging

Log to Weights & Biases (`wandb`) if available, else TensorBoard:
- Scalar: total loss, L1 loss, perceptual loss, learning rate
- Images (every 1000 steps): `cue1` (warped frame), `cue2` (anchor crop), `pred` (model output), `target`, `residual = |pred - target| * 3` — all in a grid for visual inspection

---

## 11. LCM Distillation — `distill.py`

Applied after Level 3 training. Trains the model to perform accurate 2-step inference.

```python
def distill(config: DistillConfig):
    student = InfiniteRaceWorldModel.load(config.checkpoint)
    teacher = copy.deepcopy(student)  # EMA teacher
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad_(False)

    optim = AdamW(student.parameters(), lr=config.lr_distill)
    ema   = ExponentialMovingAverage(student.parameters(), decay=0.999)

    for step, batch in enumerate(cycle(loader)):
        # Sample two timesteps: t1 > t2
        t1 = sample_timestep(config.num_train_timesteps)
        t2 = t1 - config.skip_steps  # e.g. skip_steps = 500

        # Add noise at t1 and t2
        noisy_z_t1 = add_noise(z_warp, t1)
        noisy_z_t2 = add_noise(z_warp, t2)

        # Student prediction at t1
        with autocast():
            pred_student = student(noisy_z_t1, cue2, pos_map, action)

        # Teacher prediction at t2 (no grad)
        with torch.no_grad():
            pred_teacher = teacher(noisy_z_t2, cue2, pos_map, action)

        # Consistency loss: student at t1 ≈ teacher at t2
        loss_consistency = F.mse_loss(
            student.vae.encode(pred_student),
            student.vae.encode(pred_teacher).detach()
        )

        # Standard reconstruction loss to prevent collapse
        loss_recon = masked_l1_loss(pred_student, target, cue1)

        loss = loss_consistency + 0.5 * loss_recon
        # ... backward, step, ema update, teacher sync
```

---

## 12. Real-time Inference Runner — `inference.py`

Runs the model asynchronously in a background thread, consuming from `wm_queue` and pushing to `jitter_buffer`.

```python
class WorldModelRunner(threading.Thread):
    def __init__(self, model: InfiniteRaceWorldModel, wm_queue: Queue, jitter_buffer: JitterBuffer):
        self.model   = model.cuda().eval()
        self.queue   = wm_queue
        self.buffer  = jitter_buffer
        self.running = True

    def run(self):
        while self.running:
            packet = self.queue.get(block=True, timeout=0.1)
            if packet is None:
                continue

            with torch.inference_mode(), torch.cuda.amp.autocast():
                cue1    = normalize(packet['cue1']).cuda()  # (1,3,256,256)
                cue2    = normalize(packet['cue2']).cuda()
                pos_map = torch.from_numpy(packet['pos_map']).unsqueeze(0).cuda()
                action  = torch.from_numpy(packet['cue3']).unsqueeze(0).cuda()

                keyframe = self.model.infer(cue1, cue2, pos_map, action, steps=2)
                keyframe_uint8 = denormalize(keyframe).cpu().numpy()[0]  # (3,256,256) uint8

            self.buffer.push({
                'frame':     keyframe_uint8,
                'frame_idx': packet['frame_idx'],
            })
```

### JitterBuffer

```python
class JitterBuffer:
    """Thread-safe buffer holding the last N keyframes."""
    def __init__(self, maxlen: int = 4):
        self._buffer = collections.deque(maxlen=maxlen)
        self._lock   = threading.Lock()

    def push(self, item: dict):
        with self._lock:
            self._buffer.append(item)

    def get_two_latest(self) -> tuple[dict | None, dict | None]:
        with self._lock:
            if len(self._buffer) >= 2:
                return self._buffer[-2], self._buffer[-1]
            elif len(self._buffer) == 1:
                return self._buffer[-1], self._buffer[-1]
            return None, None
```

---

## 13. Queue Interface — Phase 1 Integration — `queue_interface.py`

This is the exact integration point that replaces `cue_panel.render()` in `demo/main.py`.

```python
# In demo/main.py, the integration replaces:
#   cue_panel.render(screen, cue_data)
# With:
#   world_model_interface.send(cue_data)

class WorldModelInterface:
    """
    Bridges CueData from demo CueEngine to world model runner.
    Drop-in replacement for cue_panel.render() in Phase 1 integration.
    """
    def __init__(self, model_checkpoint: str, display_surface=None):
        self.wm_queue     = Queue(maxsize=4)
        self.jitter_buffer = JitterBuffer(maxlen=4)
        self.runner       = WorldModelRunner(
            model    = InfiniteRaceWorldModel.load(model_checkpoint),
            wm_queue = self.wm_queue,
            jitter_buffer = self.jitter_buffer,
        )
        self.runner.daemon = True
        self.runner.start()
        self.display = display_surface  # optional: pygame surface to blit output

    def send(self, cue_data: 'CueData'):
        """Called once per game tick in place of cue_panel.render()."""
        if cue_data.warped_frame is None:
            return  # Skip until first warp is available

        packet = {
            'cue1':      cue_data.warped_frame,          # (256,256,3) uint8 BGR
            'cue2':      cue_data.anchor_crop,            # (256,256,3) uint8 BGR
            'pos_map':   cue_data.anchor_pos_map,         # (256,256,2) float32
            'cue3':      cue_data.action_vector_norm,     # (3,) float32
            'frame_idx': cue_data.frame_idx,
        }
        try:
            self.wm_queue.put_nowait(packet)
        except Full:
            pass  # Drop frame if queue is full — world model is running behind

        # Optionally display latest keyframe on pygame surface for monitoring
        if self.display is not None:
            kfA, kfB = self.jitter_buffer.get_two_latest()
            if kfB is not None:
                self._blit_keyframe(kfB['frame'])

    def stop(self):
        self.runner.running = False
        self.runner.join(timeout=2.0)
```

---

## 14. Configuration — `config.py`

```python
@dataclass
class ModelConfig:
    d_model:          int   = 512
    n_heads:          int   = 4
    n_layers:         int   = 2
    latent_channels:  int   = 4
    image_size:       int   = 256
    latent_size:      int   = 32   # image_size // 8
    vae_checkpoint:   str   = "stabilityai/sd-vae-ft-mse"
    unet_checkpoint:  str   = "runwayml/stable-diffusion-v1-5"

@dataclass
class TrainConfig:
    batch_size:       int   = 16
    lr:               float = 1e-4
    steps:            int   = 50_000
    warmup_steps:     int   = 1_000
    data_dir:         str   = "training_data/"
    checkpoint_dir:   str   = "checkpoints/"
    log_every:        int   = 100
    save_every:       int   = 5_000
    grad_clip:        float = 1.0
    mixed_precision:  bool  = True

@dataclass
class DistillConfig:
    checkpoint:       str   = "checkpoints/level3_final.pt"
    lr_distill:       float = 5e-6
    steps:            int   = 20_000
    skip_steps:       int   = 500    # t1 - t2
    num_train_timesteps: int = 1000
    ema_decay:        float = 0.999

@dataclass
class InferenceConfig:
    checkpoint:       str   = "checkpoints/lcm_final.pt"
    steps:            int   = 2
    queue_maxsize:    int   = 4
    jitter_maxlen:    int   = 4
    device:           str   = "cuda"
    dtype:            str   = "float16"  # Use fp16 at inference
```

---

## 15. Dependencies

```
# requirements.txt
torch>=2.1.0
torchvision>=0.16.0
diffusers>=0.25.0        # for SD VAE loading and LCM scheduler utilities
transformers>=4.36.0     # for loading pretrained model components
accelerate>=0.25.0       # for distributed training utilities
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
torchmetrics>=1.2.0      # for LPIPS
wandb>=0.16.0
tqdm>=4.65.0
safetensors>=0.4.0
einops>=0.7.0
```

---

## 16. Deliverable checklist

Build these in order. Each item is independently testable before moving to the next.

- [ ] `config.py` — all constants defined
- [ ] `vae.py` — VAE loads SD weights, encode/decode round-trip preserves images
- [ ] `pos_encoding.py` — spherical sinusoidal for arbitrary (az, el) inputs, unit tested
- [ ] `action_mlp.py` — 3→AdaLN params, zero-init final layer verified
- [ ] `transformer.py` — cross-attention forward pass, shape test: in=(B,4,32,32) out=(B,4,32,32)
- [ ] `unet.py` — full forward pass with random weights, shape test: in=(B,3,256,256) out=(B,4,32,32)
- [ ] `model.py` — full forward pass end-to-end, shape test all four inputs → (B,3,256,256) output
- [ ] `losses.py` — masked L1 and LPIPS, verify loss decreases on overfit test
- [ ] `dataset.py` — loads .npz files, normalization correct (pixel values in [-1,1])
- [ ] `generate_dataset.py` — imports CueEngine, generates .npz files from panorama data
- [ ] `train.py` — training loop runs without error for 100 steps on synthetic data
- [ ] `lcm.py` — LCM sampler produces valid output, distillation loss computable
- [ ] `distill.py` — distillation loop runs without error
- [ ] `inference.py` — WorldModelRunner starts, processes queue packets, inference under 100ms
- [ ] `queue_interface.py` — drop-in replacement for cue_panel.render() in demo/main.py

---

## 17. Critical constraints — do not compromise these

**Real-time inference is the primary constraint.** Every architecture decision flows from this. If an implementation choice makes the 50ms target harder to achieve, default to the simpler/smaller option.

- VAE is always frozen. Never train or fine-tune the VAE.
- No skip connections in the UNet. Do not add them even if reconstruction quality improves — they add compute and memory bandwidth at inference.
- No self-attention on warped frame tokens. Cross-attention to anchor only.
- Transformer operates at 32×32 latent, never at image resolution.
- The pos_map comes from CueEngine. Do not recompute it inside the model.
- fp16 at inference. All inference code must run in `torch.cuda.amp.autocast()`.
- Queue interface must be non-blocking (`put_nowait`, drop frames if full). The game loop cannot stall waiting for the world model.
- `WorldModelRunner` is a daemon thread. It must not prevent clean process exit.

---

## 18. Testing

Provide `tests/test_shapes.py` with pytest tests verifying all tensor shapes through the forward pass:

```python
def test_full_forward_pass():
    model = InfiniteRaceWorldModel(ModelConfig())
    B = 2
    cue1    = torch.randn(B, 3, 256, 256)
    cue2    = torch.randn(B, 3, 256, 256)
    pos_map = torch.randn(B, 256, 256, 2)
    action  = torch.randn(B, 3).clamp(-1, 1)
    out = model(cue1, cue2, pos_map, action)
    assert out.shape == (B, 3, 256, 256)
    assert out.dtype == torch.float32
    assert out.min() >= -1.05 and out.max() <= 1.05

def test_inference_speed():
    model = InfiniteRaceWorldModel(ModelConfig()).cuda().half().eval()
    cue1    = torch.randn(1, 3, 256, 256).cuda().half()
    cue2    = torch.randn(1, 3, 256, 256).cuda().half()
    pos_map = torch.randn(1, 256, 256, 2).cuda().half()
    action  = torch.randn(1, 3).cuda().half()
    # Warm up
    for _ in range(3):
        model.infer(cue1, cue2, pos_map, action, steps=2)
    # Time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        model.infer(cue1, cue2, pos_map, action, steps=2)
    torch.cuda.synchronize()
    ms_per_frame = (time.perf_counter() - t0) / 10 * 1000
    assert ms_per_frame < 50, f"Inference too slow: {ms_per_frame:.1f}ms"
```

Also provide `tests/test_integration.py` that imports from `demo.cue_engine` and verifies the queue interface receives correctly shaped packets.
