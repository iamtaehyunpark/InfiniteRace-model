import math
import numpy as np
import cv2
import pygame

from config import (
    PANEL_PADDING, PANEL_GAP, PANEL_W, PANEL_H,
    FOV_DEG, LOOKAHEAD_FOV_DEG, PLAYER_COLOR,
)

_HEADER_H   = 22
_BORDER_COL = (55, 55, 75)
_BG_COL     = (18, 18, 30)
_TEXT_COL   = (230, 230, 230)
_DIM_COL    = (110, 110, 130)
_NONE_COL   = (80, 80, 100)

# Node colours (nearest=yellow, second=green, third=blue)
_NODE_COLS = [(255, 200, 50), (60, 200, 140), (60, 140, 220)]


def _bgr_to_surface(bgr: np.ndarray) -> pygame.Surface:
    arr = np.ascontiguousarray(bgr[:, :, ::-1].transpose(1, 0, 2))
    return pygame.surfarray.make_surface(arr)


class CuePanel:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self._panel_origins: list[tuple[int, int]] = []
        for row in range(3):
            for col in range(2):
                px = PANEL_PADDING + col * (PANEL_W + PANEL_GAP)
                py = PANEL_PADDING + row * (PANEL_H + PANEL_GAP)
                self._panel_origins.append((px, py))

        pygame.font.init()
        self._f_label = pygame.font.SysFont("monospace", 11, bold=True)
        self._f_data  = pygame.font.SysFont("monospace", 11)
        self._f_tag   = pygame.font.SysFont("monospace", 10)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _panel_rect(self, idx: int) -> tuple[int, int, int, int]:
        rx, ry = self._panel_origins[idx]
        return self.x + rx, self.y + ry, PANEL_W, PANEL_H

    def _draw_bg(self, surf, rx, ry) -> None:
        pygame.draw.rect(surf, _BG_COL, (rx, ry, PANEL_W, PANEL_H))
        pygame.draw.rect(surf, _BORDER_COL, (rx, ry, PANEL_W, PANEL_H), 1)

    def _header(self, surf, rx, ry, title: str) -> int:
        pygame.draw.rect(surf, (28, 28, 44), (rx, ry, PANEL_W, _HEADER_H))
        pygame.draw.line(surf, _BORDER_COL,
                         (rx, ry + _HEADER_H), (rx + PANEL_W, ry + _HEADER_H), 1)
        t = self._f_label.render(title, True, _TEXT_COL)
        surf.blit(t, (rx + 6, ry + (_HEADER_H - t.get_height()) // 2))
        return ry + _HEADER_H + 2

    def _blit_image(self, surf, img_bgr: np.ndarray, rx, ry, w, h) -> None:
        if img_bgr is None or img_bgr.size == 0 or w <= 0 or h <= 0:
            return
        scaled = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
        surf.blit(_bgr_to_surface(scaled), (rx, ry))

    def _tag(self, surf, text: str, rx, ry, w, h) -> None:
        t  = self._f_tag.render(text, True, (255, 255, 255))
        bg = pygame.Surface((t.get_width() + 6, t.get_height() + 2), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        bx = rx + w - bg.get_width() - 3
        by = ry + h - bg.get_height() - 3
        surf.blit(bg, (bx, by))
        surf.blit(t,  (bx + 3, by + 1))

    def _no_data(self, surf, rx, ry, msg="No data") -> None:
        t  = self._f_tag.render(msg, True, _NONE_COL)
        cx = rx + PANEL_W // 2 - t.get_width() // 2
        cy = ry + PANEL_H // 2 - t.get_height() // 2
        surf.blit(t, (cx, cy))

    # ------------------------------------------------------------------
    # Panel 0 — Anchor crop  (Cue 2 — world model cross-attention K/V)
    # ------------------------------------------------------------------

    def _draw_spherical_grid(self, surf, rx, cy, iw, ih, fov_deg: float) -> None:
        """Overlay azimuth/elevation grid on a rectilinear crop."""
        ov = pygame.Surface((iw, ih), pygame.SRCALPHA)
        tan_h = math.tan(math.radians(fov_deg / 2.0))

        def az_to_px(az_off_deg: float) -> int:
            return int((math.tan(math.radians(az_off_deg)) / tan_h * 0.5 + 0.5) * iw)

        def el_to_py(el_off_deg: float) -> int:
            return int((-math.tan(math.radians(el_off_deg)) / tan_h * 0.5 + 0.5) * ih)

        grid_color    = (80, 220, 120, 55)
        horizon_color = (80, 220, 120, 100)

        for az in range(-60, 61, 15):
            x = az_to_px(az)
            if 0 <= x < iw:
                col = horizon_color if az == 0 else grid_color
                pygame.draw.line(ov, col, (x, 0), (x, ih), 1)
                if az != 0:
                    lbl = self._f_tag.render(f"{az:+d}", True, (80, 200, 100, 180))
                    ov.blit(lbl, (x + 2, 2))

        for el in range(-45, 46, 15):
            y = el_to_py(el)
            if 0 <= y < ih:
                col = horizon_color if el == 0 else grid_color
                pygame.draw.line(ov, col, (0, y), (iw, y), 1)
                if el != 0:
                    lbl = self._f_tag.render(f"{el:+d}", True, (80, 200, 100, 180))
                    ov.blit(lbl, (2, y + 2))

        surf.blit(ov, (rx, cy))

    def _panel0(self, surf, rx, ry, cd) -> None:
        cy = self._header(surf, rx, ry,
                          f"Cue 2 — Anchor crop  ·  {cd.nearest_node_id}  ·  {cd.nearest_node_dist_m:.1f} m")
        iw = PANEL_W - 4
        ih = ry + PANEL_H - cy - 2
        self._blit_image(surf, cd.anchor_crop, rx + 2, cy, iw, ih)
        self._draw_spherical_grid(surf, rx + 2, cy, iw, ih, FOV_DEG)

    # ------------------------------------------------------------------
    # Panel 1 — Warped frame  (Cue 1 — world model UNet primary input)
    # ------------------------------------------------------------------

    def _panel1(self, surf, rx, ry, cd) -> None:
        cy = self._header(surf, rx, ry, "Cue 1 — Warped previous frame")
        iw = PANEL_W - 4
        ih = ry + PANEL_H - cy - 2
        if cd.warped_frame is None:
            self._no_data(surf, rx, ry + _HEADER_H, "Waiting…")
            return
        self._blit_image(surf, cd.warped_frame, rx + 2, cy, iw, ih)

    # ------------------------------------------------------------------
    # Panel 2 — Look-ahead crop  (interpolation engine grounding)
    # ------------------------------------------------------------------

    def _panel2(self, surf, rx, ry, cd) -> None:
        cy = self._header(surf, rx, ry,
                          f"Interp — Look-ahead  ·  {LOOKAHEAD_FOV_DEG:.0f}° FOV")
        iw = PANEL_W - 4
        ih = ry + PANEL_H - cy - 2
        if cd.lookahead_crop is None:
            self._no_data(surf, rx, ry + _HEADER_H)
            return
        self._blit_image(surf, cd.lookahead_crop, rx + 2, cy, iw, ih)
        # Thin vertical line showing the 90° FOV boundary within the 120° crop
        fov_frac = math.tan(math.radians(FOV_DEG / 2.0)) / math.tan(math.radians(LOOKAHEAD_FOV_DEG / 2.0))
        x_left  = int((0.5 - fov_frac * 0.5) * iw)
        x_right = int((0.5 + fov_frac * 0.5) * iw)
        ov = pygame.Surface((iw, ih), pygame.SRCALPHA)
        pygame.draw.line(ov, (255, 200, 50, 160), (x_left,  0), (x_left,  ih), 1)
        pygame.draw.line(ov, (255, 200, 50, 160), (x_right, 0), (x_right, ih), 1)
        surf.blit(ov, (rx + 2, cy))
        spd_tag = f"{abs(cd.speed_mps):.1f} m/s"
        self._tag(surf, spd_tag, rx + 2, cy, iw, ih)

    # ------------------------------------------------------------------
    # Panel 3 — Residual  (diagnostic: what model must correct)
    # ------------------------------------------------------------------

    def _panel3(self, surf, rx, ry, cd) -> None:
        cy = self._header(surf, rx, ry, "Diagnostic — Residual  ·  |anchor − warp| ×3")
        iw = PANEL_W - 4
        ih = ry + PANEL_H - cy - 2
        if cd.residual is None:
            self._no_data(surf, rx, ry + _HEADER_H, "Waiting…")
            return
        self._blit_image(surf, cd.residual, rx + 2, cy, iw, ih)

    # ------------------------------------------------------------------
    # Panel 4 — Action vector  (Cue 3 — world model AdaLN)
    # ------------------------------------------------------------------

    def _panel4(self, surf, rx, ry, cd) -> None:
        cy = self._header(surf, rx, ry, "Cue 3 — Action vector")

        # Left: compass
        r     = min((PANEL_H - _HEADER_H) // 2 - 8, PANEL_W // 4 - 4)
        cx_c  = rx + PANEL_W // 4
        cy_c  = ry + _HEADER_H + r + 6
        pygame.draw.circle(surf, (28, 28, 44), (cx_c, cy_c), r)
        pygame.draw.circle(surf, _BORDER_COL,  (cx_c, cy_c), r, 1)
        for label, az in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
            rad = math.radians(az - 90)
            tx  = cx_c + int((r - 10) * math.cos(rad))
            ty  = cy_c + int((r - 10) * math.sin(rad))
            t   = self._f_tag.render(label, True, _DIM_COL)
            surf.blit(t, (tx - t.get_width() // 2, ty - t.get_height() // 2))
        arrow_rad = math.radians(cd.heading_deg - 90)
        ax = cx_c + int((r - 6) * math.cos(arrow_rad))
        ay = cy_c + int((r - 6) * math.sin(arrow_rad))
        pygame.draw.line(surf, PLAYER_COLOR, (cx_c, cy_c), (ax, ay), 2)
        pygame.draw.circle(surf, PLAYER_COLOR, (ax, ay), 3)

        # Right: numeric action vector + steer bar
        tx = rx + PANEL_W // 2 + 4
        ty = cy + 4
        lines = [
            ("speed", f"{cd.speed_mps:+.2f} m/s"),
            ("Δhdg",  f"{cd.delta_heading_deg:+.2f}°"),
            ("steer", None),   # drawn as bar below
        ]
        for label, val in lines[:2]:
            lbl = self._f_data.render(f"{label:<5} {val}", True, _TEXT_COL)
            surf.blit(lbl, (tx, ty))
            ty += 14

        # Steer bar
        steer_lbl = self._f_data.render("steer", True, _TEXT_COL)
        surf.blit(steer_lbl, (tx, ty))
        ty += 14
        bar_w = PANEL_W - (PANEL_W // 2 + 4) - 6
        bar_h = 8
        bar_x = tx
        bar_y = ty
        pygame.draw.rect(surf, (40, 40, 60), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(surf, _BORDER_COL,  (bar_x, bar_y, bar_w, bar_h), 1)
        center_x = bar_x + bar_w // 2
        fill_w   = int(abs(cd.steer) * bar_w / 2)
        col      = (255, 200, 50) if abs(cd.steer) > 0.05 else (60, 60, 80)
        if cd.steer >= 0:
            pygame.draw.rect(surf, col, (center_x, bar_y + 1, fill_w, bar_h - 2))
        else:
            pygame.draw.rect(surf, col, (center_x - fill_w, bar_y + 1, fill_w, bar_h - 2))
        pygame.draw.line(surf, _DIM_COL, (center_x, bar_y), (center_x, bar_y + bar_h), 1)

    # ------------------------------------------------------------------
    # Panel 5 — Local node map  (parallax context)
    # ------------------------------------------------------------------

    def _panel5(self, surf, rx, ry, cd) -> None:
        cy = self._header(surf, rx, ry, "Node map  ·  parallax context")

        area_h = ry + PANEL_H - cy
        area_w = PANEL_W
        ctr_x  = rx + area_w // 2
        ctr_y  = cy + area_h // 2

        # Scale: fit all 3 nodes with margin
        offsets = [
            (cd.nearest_east_m,  cd.nearest_north_m),
            (cd.second_east_m,   cd.second_north_m),
            (cd.third_east_m,    cd.third_north_m),
        ]
        max_dist = max(math.hypot(e, n) for e, n in offsets)
        max_dist = max(max_dist, 5.0)
        margin   = 28
        scale    = (min(area_w, area_h) / 2 - margin) / max_dist  # px/m

        def world_to_px(east_m, north_m):
            return (int(ctr_x + east_m * scale),
                    int(ctr_y - north_m * scale))  # Y flipped

        # North arrow
        na = self._f_tag.render("N", True, _DIM_COL)
        surf.blit(na, (rx + area_w - 14, cy + 4))
        pygame.draw.line(surf, _DIM_COL,
                         (rx + area_w - 10, cy + 16),
                         (rx + area_w - 10, cy + 6), 1)
        pygame.draw.polygon(surf, _DIM_COL, [
            (rx + area_w - 10, cy + 6),
            (rx + area_w - 13, cy + 10),
            (rx + area_w - 7,  cy + 10),
        ])

        # Player FOV wedge
        fov_half = math.radians(FOV_DEG / 2.0)
        h_rad    = math.radians(cd.heading_deg)
        depth_px = int(min(max_dist * 0.8, max_dist) * scale)
        depth_px = max(depth_px, 20)
        l_ang    = h_rad - fov_half - math.pi / 2
        r_ang    = h_rad + fov_half - math.pi / 2
        wedge_l  = (int(ctr_x + depth_px * math.cos(l_ang)),
                    int(ctr_y + depth_px * math.sin(l_ang)))
        wedge_r  = (int(ctr_x + depth_px * math.cos(r_ang)),
                    int(ctr_y + depth_px * math.sin(r_ang)))
        fov_ov   = pygame.Surface((area_w, area_h), pygame.SRCALPHA)
        pygame.draw.polygon(fov_ov, (255, 200, 50, 25),
                            [(ctr_x - rx, ctr_y - cy), wedge_l, wedge_r])
        surf.blit(fov_ov, (rx, cy))
        pygame.draw.line(surf, (255, 200, 50, 120), (ctr_x, ctr_y), wedge_l, 1)
        pygame.draw.line(surf, (255, 200, 50, 120), (ctr_x, ctr_y), wedge_r, 1)

        # Node dots + connecting lines
        node_ids = [cd.nearest_node_id, cd.second_node_id, cd.third_node_id]
        for i, ((e, n), col) in enumerate(zip(offsets, _NODE_COLS)):
            npx, npy = world_to_px(e, n)
            pygame.draw.line(surf, (50, 50, 70), (ctr_x, ctr_y), (npx, npy), 1)
            pygame.draw.circle(surf, col, (npx, npy), 4)
            dist_m = math.hypot(e, n)
            lbl = self._f_tag.render(f"{node_ids[i]}  {dist_m:.0f}m", True, col)
            surf.blit(lbl, (npx + 6, npy - lbl.get_height() // 2))

        # Player dot
        pygame.draw.circle(surf, PLAYER_COLOR, (ctr_x, ctr_y), 5)
        h_end = (int(ctr_x + 12 * math.sin(h_rad)),
                 int(ctr_y - 12 * math.cos(h_rad)))
        pygame.draw.line(surf, PLAYER_COLOR, (ctr_x, ctr_y), h_end, 2)

        # Scale bar
        scale_m  = 10
        scale_px = int(scale_m * scale)
        if scale_px > 5:
            bar_x = rx + 6
            bar_y = ry + PANEL_H - 8
            pygame.draw.line(surf, _DIM_COL, (bar_x, bar_y), (bar_x + scale_px, bar_y), 2)
            lbl = self._f_tag.render(f"{scale_m}m", True, _DIM_COL)
            surf.blit(lbl, (bar_x + scale_px + 3, bar_y - 5))

    # ------------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------------

    def render(self, surface, cue_data) -> None:
        pygame.draw.rect(surface, (12, 12, 22),
                         (self.x, self.y, self.width, self.height))

        renderers = [
            self._panel0,   # Cue 2 — anchor crop
            self._panel1,   # Cue 1 — warped frame
            self._panel2,   # Interp — look-ahead crop
            self._panel3,   # Diagnostic — residual
            self._panel4,   # Cue 3 — action vector
            self._panel5,   # Node map
        ]

        for idx, renderer in enumerate(renderers):
            rx, ry, _, _ = self._panel_rect(idx)
            self._draw_bg(surface, rx, ry)
            renderer(surface, rx, ry, cue_data)
