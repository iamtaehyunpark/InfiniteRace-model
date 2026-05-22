import math
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional

from config import (
    NEAREST_CACHE_DIST_M, ANCHOR_CROP_SIZE, FOV_DEG, EARTH_RADIUS_M,
    LOOKAHEAD_TIME_S, LOOKAHEAD_FOV_DEG, MOVE_SPEED,
)
from utils import haversine_distance

_SCENE_DEPTH_M = 50.0   # cylindrical scene depth for parallax projection


@dataclass
class CueData:
    # ── Cue 2 — directional anchor crop (world model cross-attention K/V) ──
    anchor_crop: np.ndarray              # (256, 256, 3) uint8 BGR
    anchor_pos_map: np.ndarray           # (256, 256, 2) float32 [azimuth_rad, elevation_rad]
    nearest_node_id: str
    nearest_node_dist_m: float

    # ── Cue 1 — warped previous frame (world model UNet primary input) ──
    warped_frame: Optional[np.ndarray]   # (256, 256, 3) uint8 BGR

    # ── Cue 3 — action vector (world model AdaLN) ──
    speed_mps: float
    delta_heading_deg: float
    steer: float                         # normalized –1 to 1
    action_vector_norm: np.ndarray       # (3,) float32 [speed_norm, delta_hdg_norm, steer]

    # ── Interpolation grounding — look-ahead panorama crop ──
    lookahead_crop: Optional[np.ndarray] # (256, 256, 3) uint8 BGR, 120° FOV

    # ── Diagnostic ──
    residual: Optional[np.ndarray]       # amplified |anchor − warp|

    # ── Temporal ──
    frame_idx: int                       # monotonic frame counter

    # ── Player state (visualization only) ──
    heading_deg: float
    elevation_deg: float
    dx_m: float
    dy_m: float

    # ── Nearest 3 nodes — ids and E/N offsets from player (metres) ──
    second_node_id: str
    third_node_id: str
    nearest_east_m: float
    nearest_north_m: float
    second_east_m: float
    second_north_m: float
    third_east_m: float
    third_north_m: float


class CueEngine:
    def __init__(self, nodes):
        self.nodes = nodes
        self._cache_lat: Optional[float] = None
        self._cache_lon: Optional[float] = None
        self._cache_sorted: list = []   # [(dist_m, node, east_m, north_m), ...]

        self._ref_node_id: Optional[str] = None
        self._ref_crop:    Optional[np.ndarray] = None
        self._ref_heading: float = 0.0
        self._frame_idx:   int   = 0

    # ------------------------------------------------------------------
    # Node cache
    # ------------------------------------------------------------------

    def _refresh_cache(self, lat, lon) -> None:
        if (self._cache_lat is not None
                and haversine_distance(lat, lon, self._cache_lat, self._cache_lon)
                < NEAREST_CACHE_DIST_M):
            return

        lat_rad = math.radians(lat)
        mpdlat  = EARTH_RADIUS_M * math.pi / 180.0
        mpdlon  = EARTH_RADIUS_M * math.cos(lat_rad) * math.pi / 180.0

        ranked = []
        for n in self.nodes:
            d   = haversine_distance(lat, lon, n.lat, n.lon)
            e_m = (n.lon - lon) * mpdlon
            n_m = (n.lat - lat) * mpdlat
            ranked.append((d, n, e_m, n_m))
        ranked.sort(key=lambda x: x[0])

        self._cache_sorted = ranked
        self._cache_lat, self._cache_lon = lat, lon

    # ------------------------------------------------------------------
    # Panorama cropping
    # ------------------------------------------------------------------

    def _crop_at_heading(self, img: np.ndarray, heading_deg: float,
                         compass_angle: float = 0.0, elevation_deg: float = 0.0,
                         fov: float = FOV_DEG) -> np.ndarray:
        """Rectilinear perspective crop from an equirectangular panorama (square output)."""
        S = ANCHOR_CROP_SIZE
        PH, PW = img.shape[:2]

        hr = math.radians(heading_deg)
        pr = math.radians(elevation_deg)
        ch, sh = math.cos(hr), math.sin(hr)
        cp, sp = math.cos(pr), math.sin(pr)

        right = np.array([ ch,      -sh,      0.0], dtype=np.float32)
        fwd   = np.array([ sh * cp,  ch * cp, sp ], dtype=np.float32)
        up    = np.array([-sh * sp, -ch * sp, cp ], dtype=np.float32)

        tan_h = math.tan(math.radians(fov / 2.0))
        xs = np.linspace(-tan_h,  tan_h, S, dtype=np.float32)
        ys = np.linspace( tan_h, -tan_h, S, dtype=np.float32)
        cx, cy = np.meshgrid(xs, ys)

        wx = cx * right[0] + cy * up[0] + fwd[0]
        wy = cx * right[1] + cy * up[1] + fwd[1]
        wz = cx * right[2] + cy * up[2] + fwd[2]

        azimuth   = np.arctan2(wx.astype(np.float64), wy.astype(np.float64))
        elevation = np.arctan2(wz.astype(np.float64),
                               np.sqrt(wx.astype(np.float64)**2 + wy.astype(np.float64)**2))

        c_off = math.radians(compass_angle + 180.0)
        map_x = (((azimuth - c_off) / (2.0 * math.pi)) % 1.0 * PW).astype(np.float32)
        map_y = ((0.5 - elevation / math.pi) * PH).astype(np.float32)

        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def _pos_map(self, heading_deg: float, elevation_deg: float,
                 fov: float = FOV_DEG) -> np.ndarray:
        """
        Returns (S, S, 2) float32 array of absolute world-space
        [azimuth_rad, elevation_rad] for each pixel of a rectilinear crop
        at the given heading and elevation.

        This is the per-pixel geometric metadata needed for spherical
        sinusoidal positional encoding in the world model's cross-attention.
        azimuth_rad  ∈ (−π, π)     — compass bearing from North, CW
        elevation_rad ∈ (−π/2, π/2) — angle above/below horizon
        """
        S = ANCHOR_CROP_SIZE
        hr = math.radians(heading_deg)
        pr = math.radians(elevation_deg)
        ch, sh = math.cos(hr), math.sin(hr)
        cp, sp = math.cos(pr), math.sin(pr)

        right = np.array([ ch,      -sh,      0.0], dtype=np.float64)
        fwd   = np.array([ sh * cp,  ch * cp, sp ], dtype=np.float64)
        up    = np.array([-sh * sp, -ch * sp, cp ], dtype=np.float64)

        tan_h = math.tan(math.radians(fov / 2.0))
        xs = np.linspace(-tan_h,  tan_h, S, dtype=np.float64)
        ys = np.linspace( tan_h, -tan_h, S, dtype=np.float64)
        cx, cy = np.meshgrid(xs, ys)

        wx = cx * right[0] + cy * up[0] + fwd[0]
        wy = cx * right[1] + cy * up[1] + fwd[1]
        wz = cx * right[2] + cy * up[2] + fwd[2]

        azimuth   = np.arctan2(wx, wy).astype(np.float32)
        elevation = np.arctan2(wz, np.sqrt(wx**2 + wy**2)).astype(np.float32)

        return np.stack([azimuth, elevation], axis=-1)   # (S, S, 2)

    def _merged_crop(self,
                     nodes_dists: list,          # [(dist_m, node, east_m, north_m), ...]
                     heading_deg: float,
                     elevation_deg: float,
                     fov_deg: float = FOV_DEG,
                     origin_east_m: float = 0.0,  # camera origin offset from player
                     origin_north_m: float = 0.0,
                     scene_depth_m: float = _SCENE_DEPTH_M) -> np.ndarray:
        """
        Parallax-correct per-pixel blend.

        For each output pixel the player's ray is traced to scene_depth_m (cylindrical).
        The resulting world point is then sampled from each node's panorama at the
        exact bearing from THAT NODE'S POSITION.  Blended by 1/distance.

        origin_east_m / origin_north_m shift the camera origin (used for look-ahead
        so the crop is grounded at a projected-ahead position rather than the player).
        """
        S = ANCHOR_CROP_SIZE
        hr = math.radians(heading_deg)
        pr = math.radians(elevation_deg)
        ch, sh = math.cos(hr), math.sin(hr)
        cp, sp = math.cos(pr), math.sin(pr)

        right = np.array([ ch,      -sh,      0.0], dtype=np.float64)
        fwd   = np.array([ sh * cp,  ch * cp, sp ], dtype=np.float64)
        up    = np.array([-sh * sp, -ch * sp, cp ], dtype=np.float64)

        tan_h = math.tan(math.radians(fov_deg / 2.0))
        xs = np.linspace(-tan_h,  tan_h, S, dtype=np.float64)
        ys = np.linspace( tan_h, -tan_h, S, dtype=np.float64)
        cx, cy_g = np.meshgrid(xs, ys)

        dx = cx * right[0] + cy_g * up[0] + fwd[0]
        dy = cx * right[1] + cy_g * up[1] + fwd[1]
        dz = cx * right[2] + cy_g * up[2] + fwd[2]

        # Cylindrical depth — preserves vertical pillars correctly
        cyl_dist = np.maximum(np.sqrt(dx**2 + dy**2), 1e-6)
        t  = scene_depth_m / cyl_dist
        wx = dx * t + origin_east_m
        wy = dy * t + origin_north_m
        wz = dz * t

        accum = np.zeros((S, S, 3), dtype=np.float64)
        w_sum = np.zeros((S, S),    dtype=np.float64)

        for dist_m, node, east_m, north_m in nodes_dists:
            PH, PW = node.image_bgr.shape[:2]
            w = 1.0 / max(dist_m, 0.1)

            rx = wx - east_m
            ry = wy - north_m
            rz = wz

            az  = np.arctan2(rx, ry)
            el  = np.arctan2(rz, np.sqrt(rx * rx + ry * ry))

            c_off = math.radians(node.compass_angle + 180.0)
            map_x = (((az - c_off) / (2.0 * math.pi)) % 1.0 * PW).astype(np.float32)
            map_y = ((0.5 - el / math.pi) * PH).astype(np.float32)

            sample = cv2.remap(node.image_bgr, map_x, map_y,
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            accum += sample.astype(np.float64) * w
            w_sum += w

        w_sum = np.maximum(w_sum[:, :, np.newaxis], 1e-9)
        return np.clip(accum / w_sum, 0, 255).astype(np.uint8)

    def _lookahead_crop(self,
                        heading_deg: float, elevation_deg: float,
                        speed_mps: float) -> np.ndarray:
        """
        120° crop projected ahead along player trajectory for interpolation grounding.
        Camera origin is shifted forward by lookahead_dist so the crop is grounded
        in the panoramas that cover the geometry the player will reach next.
        """
        lookahead_m = min(max(abs(speed_mps), 1.0) * LOOKAHEAD_TIME_S, 25.0)
        h_rad = math.radians(heading_deg)
        proj_east_m  = lookahead_m * math.sin(h_rad)
        proj_north_m = lookahead_m * math.cos(h_rad)

        # Re-rank cached nodes by distance to projected position
        proj_ranked = sorted(
            self._cache_sorted,
            key=lambda x: math.hypot(x[2] - proj_east_m, x[3] - proj_north_m),
        )[:3]

        # Distances from projected origin for inverse-distance blending
        nodes_dists = [
            (max(math.hypot(e - proj_east_m, n - proj_north_m), 0.1), node, e, n)
            for _, node, e, n in proj_ranked
        ]

        return self._merged_crop(
            nodes_dists, heading_deg, elevation_deg,
            fov_deg=LOOKAHEAD_FOV_DEG,
            origin_east_m=proj_east_m,
            origin_north_m=proj_north_m,
        )

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, player) -> CueData:
        lat, lon  = player.lat, player.lon
        heading   = player.heading_deg
        elevation = player.elevation_deg
        speed     = player.speed_mps

        self._refresh_cache(lat, lon)

        dist_near, nearest, n1_e, n1_n = self._cache_sorted[0]
        dist_2nd,  second,  n2_e, n2_n = self._cache_sorted[1] if len(self._cache_sorted) > 1 else self._cache_sorted[0]
        dist_3rd,  third,   n3_e, n3_n = self._cache_sorted[2] if len(self._cache_sorted) > 2 else self._cache_sorted[-1]

        # ── Cue 2 — anchor crop + spherical position map ──
        anchor_crop = self._merged_crop(
            [(dist_near, nearest, n1_e, n1_n),
             (dist_2nd,  second,  n2_e, n2_n),
             (dist_3rd,  third,   n3_e, n3_n)],
            heading, elevation,
        )

        anchor_pos_map = self._pos_map(heading, elevation)

        # ── Interpolation grounding — look-ahead crop ──
        lookahead_crop = self._lookahead_crop(heading, elevation, speed)

        # ── Cue 1 — warped previous frame ──
        if nearest.id != self._ref_node_id:
            self._ref_node_id = nearest.id
            self._ref_crop    = anchor_crop.copy()
            self._ref_heading = heading

        warped_frame: Optional[np.ndarray] = None
        if self._ref_crop is not None:
            ppd_256  = ANCHOR_CROP_SIZE / FOV_DEG
            delta_az = (heading - self._ref_heading + 360.0) % 360.0
            if delta_az > 180.0:
                delta_az -= 360.0
            warped_frame = np.roll(self._ref_crop, -int(delta_az * ppd_256), axis=1)

        # ── Diagnostic — residual ──
        residual: Optional[np.ndarray] = None
        if warped_frame is not None:
            diff      = cv2.absdiff(anchor_crop, warped_frame)
            amplified = np.clip(diff.astype(np.float32) * 3.0, 0, 255).astype(np.uint8)
            gray      = cv2.cvtColor(amplified, cv2.COLOR_BGR2GRAY)
            residual  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # ── Cue 3 — action vector, raw + normalized ──
        steer = getattr(player, 'steer', 0.0)
        action_vector_norm = np.array([
            player.speed_mps / max(MOVE_SPEED, 1e-3),   # [−1, 1] forward/reverse
            player.delta_heading / 180.0,                # [−1, 1] heading change
            steer,                                       # [−1, 1] steer
        ], dtype=np.float32)

        self._frame_idx += 1

        return CueData(
            anchor_crop=anchor_crop,
            anchor_pos_map=anchor_pos_map,
            nearest_node_id=nearest.id,
            nearest_node_dist_m=dist_near,
            warped_frame=warped_frame,
            speed_mps=speed,
            delta_heading_deg=player.delta_heading,
            steer=steer,
            action_vector_norm=action_vector_norm,
            lookahead_crop=lookahead_crop,
            residual=residual,
            frame_idx=self._frame_idx,
            heading_deg=heading,
            elevation_deg=elevation,
            dx_m=player.dx_m,
            dy_m=player.dy_m,
            second_node_id=second.id,
            third_node_id=third.id,
            nearest_east_m=n1_e,
            nearest_north_m=n1_n,
            second_east_m=n2_e,
            second_north_m=n2_n,
            third_east_m=n3_e,
            third_north_m=n3_n,
        )
