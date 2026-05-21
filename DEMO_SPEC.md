# InfiniteRace Model Input Demo — Technical Specification

**Version 1.0 · 2026-05-21**

---

## 1. Purpose

This demo is **partially production code**. `CueEngine` — the cue extraction module — will run in the product as a Python service sitting between the game engine and the AI models. It is not a prototype to be ported or discarded; it ships into the product with the visualization shell stripped away.

The demo wraps `CueEngine` in a Pygame environment that simulates the game engine's role (WASD navigation, mouse heading, GPS coordinate tracking from panorama data) so the cue extraction logic can be developed, debugged, and verified against real panorama data before the actual game engine is built.

The two components of the demo have completely different fates in production:

| Component | In demo | In product |
|---|---|---|
| `CueEngine` + `CueData` | Core, exercised by `_Player` simulation | Ships as-is — runs as Python service, fed by game engine telemetry via shared memory |
| `_Player`, `StreetViewer`, panels, minimap | Simulation shell and visualization | Stripped entirely — replaced by game engine layer |

The demo therefore serves three simultaneous purposes:
1. **Verification** — visually inspect all six model inputs in real time to catch parallax bugs, warp errors, and residual anomalies before they corrupt model training
2. **Development platform** — iterate on cue extraction algorithms (`_merged_crop`, `_lookahead_crop`, `_pos_map`) against real panorama data with immediate visual feedback
3. **Production pre-integration** — the exact Python module that the product AI pipeline will import and call once the game engine is ready to provide real telemetry

---

## 2. System Context

The InfiniteRace AI pipeline has three concurrent execution layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  GAME ENGINE LAYER  (sync, locked to display rate)              │
│  Physics · input · road mesh · HUD · skybox texture upload      │
│  Produces: player telemetry (lat, lon, heading, speed, steer)   │
│            game engine motion vectors (pixel-level, per frame)  │
└─────────────────┬───────────────────────┬───────────────────────┘
                  │                       │
                  ▼                       ▼
┌─────────────────────────┐   ┌──────────────────────────────────┐
│  WORLD MODEL LAYER      │   │  INTERPOLATION LAYER             │
│  (async, as fast as     │   │  (sync, inside render loop)      │
│   possible)             │   │                                  │
│  Inputs: Cue 1+2+3      │   │  Inputs: 2× keyframes from       │
│  Output: keyframe →     │──▶│    jitter buffer                 │
│    jitter buffer        │   │  + game engine motion vectors    │
└─────────────────────────┘   │  + look-ahead panorama crop      │
                               │  Output: display frame at 60fps  │
                               └──────────────────────────────────┘
```

**The CueEngine is the bridge.** It runs synchronously inside the game engine layer, once per game tick. It produces `CueData`, a single structured object that contains all data needed by both downstream AI layers.

---

## 3. Demo Architecture

### 3.1 File Structure

| File | Role |
|---|---|
| `demo/cue_engine.py` | Core cue extraction logic. `CueData` and `CueEngine` define the interface contract the product engine must satisfy. |
| `demo/cue_panel.py` | Pygame visualization of the six cue panels. Demo-only display layer. |
| `demo/main.py` | Pygame event loop and `_Player` — simulates the game engine's telemetry role for testing purposes only. |
| `demo/viewer.py` | Google Street View–style rectilinear panorama renderer and first-person navigation. |
| `demo/minimap.py` | Tile-map minimap with node positions and prismatoid frustums. |
| `demo/config.py` | All constants. `DATA_FOLDER` points to the panorama dataset. |
| `demo/loader.py` | Loads panorama nodes from a folder: JPEG + `coordinates.json`. |

### 3.2 Data Flow Per Frame

```
DEMO (development)                     PRODUCT (deployed)
─────────────────────────────          ──────────────────────────────────────
_Player.update(dt, keys, viewer)       Game engine (C++/Rust) physics tick
  simulated lat/lon/heading/speed  →     real lat/lon/heading/speed/steer
  via WASD + mouse drag                  via shared memory, sub-ms latency

cue_engine.update(player)           ←  IDENTICAL  →  cue_engine.update(telemetry)
  → CueData                                            → CueData

cue_panel.render(screen, cue_data)     wm_queue.put(cue1, cue2, pos_map, cue3)
  6-panel visualization                interp_engine.set_lookahead(lookahead_crop)
  (stripped in product)                skybox_texture.upload(interp_frame)
```

`CueEngine.update()` is the single call that is identical in both environments. Everything above it (telemetry source) and below it (consumer) differs between demo and product.

---

## 4. Cue Specifications

`CueEngine.update(player)` produces a `CueData` instance with the following fields. Fields are grouped by which downstream consumer uses them.

---

### 4.1 Cue 1 — Warped Previous Frame

**Consumer:** World model (UNet primary encoder input)

**Field:** `warped_frame: np.ndarray | None`  
**Shape:** `(256, 256, 3)` uint8, BGR channel order  
**Value:** `None` until the second frame after attaching to a new nearest node.

**How it is produced:**

When the nearest panorama node changes (i.e., the player has moved far enough that a different node is closest), the engine snapshots the current `anchor_crop` as the reference image `_ref_crop` and records `_ref_heading`.

Each subsequent frame, the reference is geometrically warped to the player's new heading via a pixel-column roll:

```
delta_az  = (current_heading − ref_heading) wrapped to (−180, 180)
pixel_shift = −round(delta_az × (256 / FOV_DEG))
warped_frame = np.roll(_ref_crop, pixel_shift, axis=1)
```

This models the dominant component of the pose change (yaw rotation) cheaply without optical flow. It is equivalent to the two-stage pose warp described in the architecture specification: the math stage. The learned optical-flow correction stage runs inside the world model, not here.

**Accuracy:** Approximately 85–90% of pixels are already correct. Errors appear at:
- Occlusion boundaries (foreground/background parallax)
- The left or right edges where new content has scrolled into view
- Regions with strong forward-motion parallax (close objects at low elevation)

**Role in the world model:** This tensor is the primary input to the UNet encoder. It establishes the temporal base. The model is trained to treat it as highly reliable and only deviate from it where the anchor crop contradicts it.

---

### 4.2 Cue 2 — Directional Anchor Crop

**Consumer:** World model (transformer bottleneck cross-attention K/V)

**Fields:**
- `anchor_crop: np.ndarray` — `(256, 256, 3)` uint8, BGR
- `anchor_pos_map: np.ndarray` — `(256, 256, 2)` float32

**How `anchor_crop` is produced:**

The anchor crop is a rectilinear perspective crop extracted from the nearest panorama database, centered on the player's current viewing direction (heading + elevation), using a 90° horizontal field of view.

The extraction uses `cv2.remap` with precomputed arctan2-based map tables — the same algorithm as `viewer.py`'s `_build_maps()` — giving correct rectilinear perspective (no barrel distortion, straight lines stay straight) from the equirectangular source panoramas.

Because the player is almost never standing exactly at a panorama node, a **parallax-corrected per-pixel blend** is used:

1. For each of the 256×256 output pixels, the player's ray is traced to a cylindrical scene depth of 50 m (horizontal distance, preserving vertical pillars correctly).
2. This gives a 3D world point for each pixel.
3. Each of the three nearest panorama nodes is sampled at the **exact bearing from that node's position** to that world point — accounting for the spatial offset between the player and the node.
4. The samples are blended by inverse distance from the player.

This parallax correction means that a vertical feature (building edge, pillar, pole) at 5 m will appear in the correct horizontal position in the anchor crop even if the nearest node is 8 m to the side — because each node's contribution is sampled from the bearing it would actually see that feature at.

**How `anchor_pos_map` is produced:**

For each pixel `(i, j)` in the 256×256 crop, the same ray tracing used to build the remap tables is applied, but instead of looking up a panorama pixel, the absolute world-space azimuth and elevation of that ray is recorded:

```
anchor_pos_map[i, j, 0] = azimuth_rad    ∈ (−π, π)
anchor_pos_map[i, j, 1] = elevation_rad  ∈ (−π/2, π/2)
```

Azimuth follows compass convention: 0 = North, π/2 = East, clockwise.

These values are the input to the world model's **spherical sinusoidal positional encoding**. Standard 2D learned positional encodings are insufficient because the anchor crop is a non-uniform projection of a sphere — the mapping from pixel position to angular position is non-linear and depends on the current heading and FOV. The pos_map provides the exact geometric coordinates per pixel so the transformer knows precisely where on the sphere each anchor pixel came from, enabling correct spatial placement of anchor content in the output frame.

**Role in the world model:** After VAE encoding (`anchor_crop` → 32×32 latent), the pos_map is used to add spherical sinusoidal position embeddings to the anchor latent tokens before they are used as keys and values in cross-attention. The warped frame latent (query) attends to anchor tokens at their geometrically correct positions.

---

### 4.3 Cue 3 — Action Vector

**Consumer:** World model (transformer bottleneck AdaLN)

**Fields:**
- `speed_mps: float` — raw forward speed in m/s (negative = reverse)
- `delta_heading_deg: float` — heading change in degrees since last frame
- `steer: float` — normalized steering in [−1, 1]
- `action_vector_norm: np.ndarray` — `(3,)` float32, ready for AdaLN injection

**Normalization:**

```
action_vector_norm[0] = speed_mps / MOVE_SPEED        # forward speed, clipped to [−1, 1]
action_vector_norm[1] = delta_heading_deg / 180.0      # heading rate, clipped to [−1, 1]
action_vector_norm[2] = steer                          # already [−1, 1]
```

`MOVE_SPEED` is the maximum speed in the demo (3.0 m/s). In the product, this normalization factor is updated to the vehicle's maximum speed for the road type.

`steer` is derived from `delta_heading_deg / MAX_STEER_DEG` where `MAX_STEER_DEG = 10.0` degrees/frame is the empirical full-steer threshold. In the product game engine, steer is provided directly as a physical steering angle from the physics simulation.

**Role in the world model:** The action vector is injected via Adaptive Layer Normalization (AdaLN) into the transformer bottleneck before cross-attention. It provides learned scale and shift parameters to the layer norm of each transformer layer, conditioning the entire latent representation on the magnitude and direction of player movement. Without it, the model cannot distinguish a forward translation (objects grow and move to periphery) from a yaw rotation (objects translate horizontally) from braking (scene decelerates).

---

### 4.4 Interpolation Grounding — Look-ahead Crop

**Consumer:** Interpolation engine (alongside game engine motion vectors)

**Field:** `lookahead_crop: np.ndarray`  
**Shape:** `(256, 256, 3)` uint8, BGR  
**FOV:** 120° horizontal (vs 90° for anchor crop)

**Purpose:**

The interpolation engine synthesizes intermediate frames between world model keyframes using motion vectors derived from the game engine's physics simulation. Motion vectors account for where existing pixels move — but they cannot provide pixel values for geometry that was not visible in the previous keyframe (newly occluded/disoccluded content entering from the frame edges as the car moves forward and turns).

The look-ahead crop solves this. It is a real-pixel panorama crop from a position projected ahead of the player along the current trajectory, showing the geometry the player will encounter during the next interpolation window.

**How it is produced:**

1. **Project ahead:** `lookahead_dist = min(|speed_mps| × 0.5s, 25m)`. At zero or low speed, a minimum of 0.5 m is used so the crop remains non-degenerate.
2. Compute the projected camera origin: `(proj_east, proj_north) = player + lookahead_dist × (sin(heading), cos(heading))`.
3. Re-rank the three nearest panorama nodes by their distance to the **projected position** (not the player position).
4. Call `_merged_crop` with the projected origin as camera offset and 120° FOV — same parallax-correct per-pixel blend, but the camera is shifted forward.

The 120° FOV is intentionally wider than the 90° anchor crop. During interpolation, the car is physically moving forward, so geometry at the lateral edges of the future frame (±45° to ±60°) needs to be available. The yellow boundary markers in Panel 2 of the demo visualize where the 90° world model FOV sits within the 120° look-ahead window.

**How the interpolation engine uses it:**

```
interp_input = {
    'keyframe_A':      jitter_buffer[-2],          # older keyframe
    'keyframe_B':      jitter_buffer[-1],           # newer keyframe
    'motion_vectors':  game_engine.motion_vectors,  # physics-derived pixel flow
    'lookahead_crop':  cue_data.lookahead_crop,     # new geometry source
    'frame_alpha':     t,                           # interpolation factor [0, 1]
}
```

The interpolation engine (FSR 3 or custom warp shader) uses `motion_vectors` for the warp. When a pixel in the interpolated frame falls outside both keyframes' coverage (i.e., it is new geometry), it samples from `lookahead_crop` at the corresponding azimuth/elevation. This eliminates the black-border or smearing artifacts that pure motion-vector interpolation produces at occlusion boundaries.

---

### 4.5 Diagnostic — Residual

**Consumer:** Development/debugging only. Not fed to any model.

**Field:** `residual: np.ndarray | None`  
**Shape:** `(256, 256, 3)` uint8  

**How it is produced:**
```
diff      = cv2.absdiff(anchor_crop, warped_frame)
amplified = clip(diff × 3.0, 0, 255)
residual  = grayscale(amplified) → BGR
```

**What it shows:** The actual information the world model must produce — the ~10–15% of pixels that the warp got wrong. In a healthy system, bright regions should appear only at:
- Occlusion boundaries (sharp depth discontinuities)
- Lateral edges (newly visible geometry after forward motion)
- Moving objects (not applicable with static panoramas)

If bright regions cover most of the frame, either the warp is broken (check `_ref_heading` reset logic) or the nearest node has changed without the reference being updated. If the residual is black everywhere, the player is stationary and the warp is exact — this is correct.

---

### 4.6 Temporal Field

**Field:** `frame_idx: int`

Monotonically incrementing counter, starts at 1 on the first `update()` call. Used by both model consumers to:
- Detect dropped frames (non-consecutive `frame_idx` values in the queue)
- Compute the keyframe latency: `latency_frames = current_frame_idx - keyframe.frame_idx`
- Maintain temporal ordering in the jitter buffer

---

## 5. Panel Display Reference

The demo renders six panels in a 3×2 grid to the right of the street view window.

| Panel | Title | Data source | Purpose |
|---|---|---|---|
| 0 (top-left) | Cue 2 — Anchor crop | `anchor_crop` + spherical grid overlay | Verify factual grounding; grid shows azimuth/elevation structure matching pos_map |
| 1 (top-right) | Cue 1 — Warped previous frame | `warped_frame` | Verify warp quality; should look like Panel 0 with a small lateral offset |
| 2 (mid-left) | Interp — Look-ahead | `lookahead_crop` (120° FOV) | Verify ahead geometry; yellow lines show 90° boundary within 120° window |
| 3 (mid-right) | Diagnostic — Residual | `residual` | Quantify model workload; should be sparse at occlusion edges only |
| 4 (bot-left) | Cue 3 — Action vector | `speed_mps`, `delta_heading_deg`, `steer` | Verify action conditioning; compass + numeric + steer bar |
| 5 (bot-right) | Node map | node offsets, `heading_deg` | Verify parallax geometry; top-down view of player + 3 nearest nodes |

---

## 6. Synchronous Data Flow and Timing

This section describes the exact timing relationship between the three execution layers and where `CueData` flows in each.

### 6.1 Per-Frame Tick (Sync, Target 60 fps)

The telemetry source differs between demo and product (see §3.2), but everything from `cue_engine.update()` onward is identical.

```
tick N (Δt ≈ 16.7 ms):

  [telemetry]
  demo:    _Player.update(dt, keys, viewer)
  product: read shared memory from game engine
  → lat, lon, heading, elevation, speed_mps, delta_heading, steer

  [cue extraction]   ← ~2–5 ms on CPU
  cue_data = cue_engine.update(player)
  → anchor_crop        (256×256, BGR)
  → anchor_pos_map     (256×256×2, float32)
  → warped_frame       (256×256, BGR)
  → action_vector_norm (3, float32)
  → lookahead_crop     (256×256, BGR)
  → residual           (256×256, BGR)
  → frame_idx          (int)

  [world model queue enqueue]   ← microseconds
  wm_queue.put({
      'cue1':       cue_data.warped_frame,          # (256,256,3) uint8
      'cue2':       cue_data.anchor_crop,            # (256,256,3) uint8
      'pos_map':    cue_data.anchor_pos_map,         # (256,256,2) float32
      'cue3':       cue_data.action_vector_norm,     # (3,) float32
      'frame_idx':  cue_data.frame_idx,
  })

  [interpolation]   ← sync, must complete within tick budget
  kf_A, kf_B = jitter_buffer.get_two_latest()
  interp_frame = interpolation_engine.synthesize(
      frame_A      = kf_A,
      frame_B      = kf_B,
      motion_vecs  = game_engine.motion_vectors,
      lookahead    = cue_data.lookahead_crop,
      alpha        = compute_alpha(kf_A.frame_idx, kf_B.frame_idx, cue_data.frame_idx),
  )
  skybox_texture.upload(interp_frame)  # CUDA-GL zero-copy

  [render]
  composite(road_mesh, car_model, skybox_texture, HUD) → display
```

### 6.2 World Model Thread (Async, Target 20+ keyframes/sec)

```
loop:
  packet = wm_queue.get(block=True)

  # Encode inputs
  z_warp   = vae_encoder(normalize(packet['cue1']))   # (1,4,32,32)
  z_anchor = vae_encoder(normalize(packet['cue2']))   # (1,4,32,32)

  # Add spherical positional encoding to anchor latent
  pos_enc  = spherical_sinusoidal(packet['pos_map'], d_model=512)  # (1,1024,512)
  z_anchor = z_anchor_flat + pos_enc                  # (1,1024,512)

  # Action conditioning
  adaln_params = action_mlp(packet['cue3'])           # scale+shift for each layer norm

  # Transformer bottleneck: query=warped, key/value=anchor
  z_out    = transformer(query=z_warp, kv=z_anchor, adaln=adaln_params)

  # 2-step LCM denoising → keyframe
  keyframe = vae_decoder(lcm_denoise(z_out, steps=2))  # (256,256,3)

  jitter_buffer.push({
      'frame': keyframe,
      'frame_idx': packet['frame_idx'],
  })
```

### 6.3 Latency Budget

At 60 fps game loop and 20 fps world model:

- One keyframe covers approximately 3 interpolated frames.
- The jitter buffer holds the 2 most recent keyframes.
- `frame_alpha` for interpolation = `(current_frame_idx − kf_A.frame_idx) / (kf_B.frame_idx − kf_A.frame_idx)`.
- If the world model runs faster than 20 fps, the buffer accumulates and alpha increments more smoothly.
- If the world model drops below 20 fps, the buffer holds the last valid pair and interpolation stretches the existing keyframes — visually acceptable because the look-ahead crop keeps the edges populated.

Typical `CueEngine.update()` wall-clock time on CPU: **2–5 ms** for the parallax-correct merge of three panoramas at 256×256. The bottleneck is the two `cv2.remap` calls. This is within the 16.7 ms tick budget. If the node density is low (player far from all nodes), the three nearest nodes may include nodes at 50+ m — at this distance the parallax correction has negligible effect and could be replaced by a single nearest-node crop with no perceptual difference.

---

## 7. Configuration Reference

All constants are in `demo/config.py`.

| Constant | Value | Used by |
|---|---|---|
| `ANCHOR_CROP_SIZE` | 256 | All crops — output resolution |
| `FOV_DEG` | 90.0 | Anchor crop and warped frame field of view |
| `LOOKAHEAD_FOV_DEG` | 120.0 | Look-ahead crop field of view |
| `LOOKAHEAD_TIME_S` | 0.5 | Seconds ahead to project for look-ahead origin |
| `MAX_STEER_DEG` | 10.0 | Delta-heading at which steer = ±1 |
| `MOVE_SPEED` | 3.0 m/s | Speed normalization denominator |
| `NEAREST_CACHE_DIST_M` | 5.0 m | Node ranking cache invalidation radius |
| `DATA_FOLDER` | `gsv_data/` | Panorama source directory |

---

## 8. CueData Field Reference

Complete field listing of the `CueData` dataclass returned by `CueEngine.update()`.

| Field | Type | Shape / Unit | Description |
|---|---|---|---|
| `anchor_crop` | uint8 ndarray | (256,256,3) BGR | Cue 2 image — factual grounding for world model |
| `anchor_pos_map` | float32 ndarray | (256,256,2) radians | Per-pixel [azimuth, elevation] for spherical pos encoding |
| `nearest_node_id` | str | — | ID of dominant anchor node |
| `nearest_node_dist_m` | float | metres | Distance from player to nearest node |
| `warped_frame` | uint8 ndarray \| None | (256,256,3) BGR | Cue 1 — pose-warped previous anchor |
| `speed_mps` | float | m/s | Raw forward speed |
| `delta_heading_deg` | float | degrees/frame | Heading change this frame |
| `steer` | float | [−1, 1] | Normalized steering |
| `action_vector_norm` | float32 ndarray | (3,) | [speed_norm, Δhdg_norm, steer] — Cue 3 |
| `lookahead_crop` | uint8 ndarray | (256,256,3) BGR | Interpolation grounding — 120° look-ahead |
| `residual` | uint8 ndarray \| None | (256,256,3) BGR | Diagnostic: |anchor − warp| ×3 |
| `frame_idx` | int | count | Monotonic frame counter |
| `heading_deg` | float | degrees | Compass heading (0=N, 90=E) |
| `elevation_deg` | float | degrees | Tilt angle (+ = up) |
| `dx_m` | float | metres | East displacement this frame |
| `dy_m` | float | metres | North displacement this frame |
| `second_node_id` | str | — | Second nearest node ID |
| `third_node_id` | str | — | Third nearest node ID |
| `nearest_east_m` | float | metres | Nearest node east offset from player |
| `nearest_north_m` | float | metres | Nearest node north offset from player |
| `second_east_m` | float | metres | Second node east offset |
| `second_north_m` | float | metres | Second node north offset |
| `third_east_m` | float | metres | Third node east offset |
| `third_north_m` | float | metres | Third node north offset |

---

## 9. Integration Path

`CueEngine` ships into the product without reimplementation. The integration has two sequential phases:

**Phase 1 — Model inference attached to demo (POC validation)**

Replace the visualization call in `main.py` with live model runners. The demo's `_Player` continues to supply simulated telemetry. This validates end-to-end inference quality against real panorama data before the game engine exists:

```python
# In main.py, replace:
cue_panel.render(screen, cue_data)

# With:
wm_queue.put_nowait({
    'cue1':      cue_data.warped_frame,
    'cue2':      cue_data.anchor_crop,
    'pos_map':   cue_data.anchor_pos_map,
    'cue3':      cue_data.action_vector_norm,
    'frame_idx': cue_data.frame_idx,
})
interp_engine.set_lookahead(cue_data.lookahead_crop)
```

**Phase 2 — Game engine integration**

When the game engine (C++/Rust) is ready, it exposes player telemetry to the Python AI pipeline via shared memory — the same sub-millisecond zero-copy mechanism used for skybox texture upload. `CueEngine` receives this telemetry in place of `_Player`. No changes to `CueEngine` are required. The game engine does not call `CueEngine` — it writes telemetry to a shared memory region that the Python AI process reads:

```
Game engine writes (shared memory):
  { lat, lon, heading_deg, elevation_deg, speed_mps, steer, motion_vectors[] }

CueEngine reads and processes → CueData → world model queue + interp engine
```

The product does not use the demo's `_Player`, `StreetViewer`, `Minimap`, or any Pygame code. Those are stripped when the game engine integration is live. `CueEngine`, `CueData`, `loader.py`, and the panorama database pipeline are unchanged.

**Input tensor preparation for the world model:**

```python
def prepare_world_model_inputs(packet):
    # Normalize images from uint8 [0, 255] to float32 [−1, 1]
    cue1_tensor = (packet['cue1'][..., ::-1].astype(np.float32) / 127.5 - 1.0)
    cue2_tensor = (packet['cue2'][..., ::-1].astype(np.float32) / 127.5 - 1.0)
    # pos_map already float32 in radians — pass directly
    pos_map     = packet['pos_map']
    # action vector already normalized float32
    action_vec  = packet['cue3']
    return cue1_tensor, cue2_tensor, pos_map, action_vec
```

**Tensor shapes expected by world model:**

```
cue1:       (B, 3, 256, 256)  float32, [−1, 1]   → VAE encoder → (B, 4, 32, 32) latent
cue2:       (B, 3, 256, 256)  float32, [−1, 1]   → VAE encoder → (B, 4, 32, 32) latent
pos_map:    (B, 256, 256, 2)  float32, radians    → spherical sinusoidal encoding
action_vec: (B, 3)            float32, [−1, 1]   → 3-layer MLP → AdaLN params
output:     (B, 3, 256, 256)  float32, [−1, 1]   → upscale to display resolution
```

---

## 10. Panorama Database Format

The demo loads panoramas from a folder containing:

- One JPEG per node, named by node ID (e.g., `node_000.jpg`)
- A `coordinates.json` file with the following schema per entry:

```json
{
    "id":            "node_000",
    "lat":           37.5665,
    "lon":           126.9780,
    "image":         "node_000.jpg",
    "compass_angle": 90.0
}
```

`compass_angle` is the compass bearing (degrees, 0=North) that maps to the **center** of the panorama image. Equivalently, the pixel at column 0 corresponds to `compass_angle + 180°`.

Panoramas must be equirectangular, 2:1 aspect ratio. Any resolution is supported; 4096×2048 is recommended for the demo. The production system uses Google Street View panoramas at 5760×2880.

Two datasets are included:

| Folder | Contents | Purpose |
|---|---|---|
| `gsv_data/` | Real Mapillary street view panoramas | Primary demo dataset |
| `calib_data/` | Synthetic panoramas with known pillar positions | Parallax calibration — see `gen_calib_panoramas.py` |

---

## 11. Calibration

`gen_calib_panoramas.py` generates a synthetic dataset with colored vertical pillars at known world positions for verifying the parallax correction in `_merged_crop`.

**Calibration procedure:**

1. Set `DATA_FOLDER = "calib_data"` in `config.py` and run the demo.
2. Open Panel 0 (Anchor crop). Each colored pillar should appear as a single sharp vertical stripe at its correct azimuth. If a pillar appears as a blurred or doubled stripe, the parallax blend in `_merged_crop` has a bug.
3. Open Panel 5 (Node map). Verify that the three nearest nodes are at the correct spatial positions relative to the player.
4. Move the player laterally (A/D keys) and confirm that the pillar stripes in Panel 0 shift correctly — closer pillars shift more than distant ones.
5. Set `DATA_FOLDER` back to `"gsv_data"` when calibration passes.

**Expected calibration result:** At position (0, 0) in the synthetic layout, looking North, the red pillar at (0, 10m) appears at azimuth 0°, the green pillar at (10, 10m) appears at azimuth ~45°, and the distant cyan pillar at (0, 80m) appears at azimuth 0° with negligible parallax across nodes.
