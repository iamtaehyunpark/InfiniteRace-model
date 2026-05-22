import math
import numpy as np
import cv2
import pygame

from config import (
    DEFAULT_FOV, MIN_FOV, MAX_FOV, PITCH_LIMIT,
    MAX_NAV_NODES, MAX_NAV_DIST_M, ARROW_PITCH, ARROW_HIT_PX,
)
from utils import haversine_distance, bearing_between


class StreetViewer:
    def __init__(self, nodes: list, vw: int, vh: int):
        self.nodes = nodes
        self.vw, self.vh = vw, vh

        self.current_node = nodes[0]
        self.heading_deg  = 0.0
        self.pitch_deg    = 0.0
        self.fov_deg      = DEFAULT_FOV

        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None
        self._dirty = True

        pygame.font.init()
        self._font    = pygame.font.SysFont("monospace", 14, bold=True)
        self._font_sm = pygame.font.SysFont("monospace", 11)

    # ------------------------------------------------------------------
    # State setters
    # ------------------------------------------------------------------

    def set_node(self, node) -> None:
        self.current_node = node
        self._dirty = True

    def rotate(self, dyaw: float, dpitch: float) -> None:
        self.heading_deg = (self.heading_deg + dyaw) % 360.0
        self.pitch_deg   = max(-PITCH_LIMIT, min(PITCH_LIMIT, self.pitch_deg + dpitch))
        self._dirty = True

    def zoom(self, delta: float) -> None:
        self.fov_deg = max(MIN_FOV, min(MAX_FOV, self.fov_deg + delta))
        self._dirty = True

    def navigate_to(self, node) -> None:
        bear = bearing_between(
            self.current_node.lat, self.current_node.lon,
            node.lat, node.lon,
        )
        self.heading_deg = bear
        self.pitch_deg   = 0.0
        self.set_node(node)

    # ------------------------------------------------------------------
    # Perspective remap tables
    # ------------------------------------------------------------------

    def _camera_axes(self):
        hr = math.radians(self.heading_deg)
        pr = math.radians(self.pitch_deg)
        ch, sh = math.cos(hr), math.sin(hr)
        cp, sp = math.cos(pr), math.sin(pr)
        # World frame: East=X, North=Y, Up=Z
        right   = (ch,       -sh,        0.0)
        forward = (sh * cp,   ch * cp,   sp)
        # up = right × forward
        up      = (-sh * sp, -ch * sp,   cp)
        return right, up, forward

    def _build_maps(self) -> None:
        H, W = self.vh, self.vw
        fov_h = math.radians(self.fov_deg)
        fov_v = 2.0 * math.atan(math.tan(fov_h / 2.0) * H / W)
        tan_h = math.tan(fov_h / 2.0)
        tan_v = math.tan(fov_v / 2.0)

        # Camera-space rays: linspace so centre pixel maps to exact heading
        xs = np.linspace(-tan_h, tan_h, W, dtype=np.float32)
        ys = np.linspace( tan_v, -tan_v, H, dtype=np.float32)   # Y flipped (top=up)
        cx, cy = np.meshgrid(xs, ys)                              # (H, W)
        cz = np.ones((H, W), dtype=np.float32)

        right, up, fwd = self._camera_axes()

        # World-space ray for each pixel
        wx = cx * right[0] + cy * up[0] + cz * fwd[0]
        wy = cx * right[1] + cy * up[1] + cz * fwd[1]
        wz = cx * right[2] + cy * up[2] + cz * fwd[2]

        # Spherical → equirectangular
        azimuth   = np.arctan2(wx.astype(np.float64), wy.astype(np.float64))
        elevation = np.arctan2(wz.astype(np.float64),
                               np.sqrt(wx.astype(np.float64)**2 + wy.astype(np.float64)**2))

        PH, PW = self.current_node.image_bgr.shape[:2]
        c_off = math.radians(self.current_node.compass_angle + 180.0)

        self._map_x = (((azimuth - c_off) / (2.0 * math.pi)) % 1.0 * PW).astype(np.float32)
        self._map_y = ((0.5 - elevation / math.pi) * PH).astype(np.float32)
        self._dirty = False

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def nav_nodes(self) -> list:
        dists = [
            (haversine_distance(self.current_node.lat, self.current_node.lon, n.lat, n.lon), n)
            for n in self.nodes if n is not self.current_node
        ]
        dists.sort(key=lambda x: x[0])
        return [n for d, n in dists[:MAX_NAV_NODES] if d <= MAX_NAV_DIST_M]

    def _project(self, bearing_deg: float, pitch_deg: float) -> tuple[int, int] | None:
        br = math.radians(bearing_deg)
        pr = math.radians(pitch_deg)
        wx = math.sin(br) * math.cos(pr)
        wy = math.cos(br) * math.cos(pr)
        wz = math.sin(pr)

        right, up, fwd = self._camera_axes()
        cam_x = wx * right[0] + wy * right[1] + wz * right[2]
        cam_y = wx * up[0]    + wy * up[1]    + wz * up[2]
        cam_z = wx * fwd[0]   + wy * fwd[1]   + wz * fwd[2]

        if cam_z <= 1e-6:
            return None

        tan_h = math.tan(math.radians(self.fov_deg / 2.0))
        tan_v = math.tan(math.radians(self.fov_deg / 2.0 * self.vh / self.vw))

        sx = int(( cam_x / cam_z / tan_h * 0.5 + 0.5) * self.vw)
        sy = int((-cam_y / cam_z / tan_v * 0.5 + 0.5) * self.vh)

        # Allow a margin so arrows near edges are still hittable
        M = ARROW_HIT_PX + 4
        if -M <= sx < self.vw + M and -M <= sy < self.vh + M:
            return sx, sy
        return None

    def node_at_screen(self, mx: int, my: int):
        for node in self.nav_nodes():
            bear = bearing_between(
                self.current_node.lat, self.current_node.lon, node.lat, node.lon,
            )
            pos = self._project(bear, ARROW_PITCH)
            if pos and math.hypot(pos[0] - mx, pos[1] - my) < ARROW_HIT_PX:
                return node
        return None

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, surface: pygame.Surface, mouse_pos: tuple[int, int]) -> None:
        # ── Panorama ──────────────────────────────────────────────────
        if self._dirty:
            self._build_maps()

        frame_bgr = cv2.remap(
            self.current_node.image_bgr, self._map_x, self._map_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP,
        )
        # BGR (H×W×3) → RGB transposed (W×H×3) for pygame
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1].transpose(1, 0, 2))
        pygame.surfarray.blit_array(surface, frame_rgb)

        # ── Navigation arrows ─────────────────────────────────────────
        overlay = pygame.Surface((self.vw, self.vh), pygame.SRCALPHA)
        mx, my  = mouse_pos

        for node in self.nav_nodes():
            bear    = bearing_between(
                self.current_node.lat, self.current_node.lon, node.lat, node.lon,
            )
            pos     = self._project(bear, ARROW_PITCH)
            if pos is None:
                continue

            hover   = math.hypot(pos[0] - mx, pos[1] - my) < ARROW_HIT_PX
            circ_c  = (255, 220, 50, 220) if hover else (255, 255, 255, 180)
            border_c = (0, 0, 0, 200)

            pygame.draw.circle(overlay, circ_c,   pos, 20)
            pygame.draw.circle(overlay, border_c, pos, 20, 2)

            # Arrow direction: project bearing at horizon for screen direction
            pos_h = self._project(bear, 0.0)
            if pos_h:
                adx = pos_h[0] - pos[0]
                ady = pos_h[1] - pos[1]
                n   = math.hypot(adx, ady)
                if n > 1:
                    adx /= n; ady /= n
                else:
                    adx, ady = 0.0, -1.0
            else:
                adx, ady = 0.0, -1.0

            # Chevron inside the circle
            tip = (int(pos[0] + adx * 11), int(pos[1] + ady * 11))
            bl  = (int(pos[0] - adx * 5 - ady * 8), int(pos[1] - ady * 5 + adx * 8))
            br  = (int(pos[0] - adx * 5 + ady * 8), int(pos[1] - ady * 5 - adx * 8))
            pygame.draw.polygon(overlay, (20, 20, 20, 240), [tip, bl, br])

            # Node label on hover
            if hover:
                lbl = self._font_sm.render(node.id, True, (255, 255, 255))
                lx  = pos[0] + 26
                ly  = pos[1] - lbl.get_height() // 2
                bg  = pygame.Surface((lbl.get_width() + 8, lbl.get_height() + 4), pygame.SRCALPHA)
                bg.fill((0, 0, 0, 160))
                overlay.blit(bg,  (lx - 4, ly - 2))
                overlay.blit(lbl, (lx, ly))

        surface.blit(overlay, (0, 0))

        # ── HUD ───────────────────────────────────────────────────────
        self._draw_hud(surface)

    def _draw_hud(self, surface: pygame.Surface) -> None:
        # Node ID — top left
        txt = self._font.render(self.current_node.id, True, (255, 255, 255))
        bg  = pygame.Surface((txt.get_width() + 12, txt.get_height() + 8), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 150))
        surface.blit(bg,  (10, 10))
        surface.blit(txt, (16, 14))

        # Heading — top right
        dirs  = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        card  = dirs[round(self.heading_deg / 45) % 8]
        htxt  = self._font.render(f"{self.heading_deg:05.1f}°  {card}", True, (255, 255, 255))
        hbg   = pygame.Surface((htxt.get_width() + 12, htxt.get_height() + 8), pygame.SRCALPHA)
        hbg.fill((0, 0, 0, 150))
        surface.blit(hbg,  (self.vw - htxt.get_width() - 22, 10))
        surface.blit(htxt, (self.vw - htxt.get_width() - 16, 14))

        # Pitch indicator — if not near horizon
        if abs(self.pitch_deg) > 2:
            ptxt = self._font_sm.render(
                f"{'↑' if self.pitch_deg > 0 else '↓'}{abs(self.pitch_deg):.0f}°",
                True, (200, 200, 200),
            )
            surface.blit(ptxt, (self.vw // 2 - ptxt.get_width() // 2, 14))
