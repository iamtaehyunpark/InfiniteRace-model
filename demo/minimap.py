import math
import pygame

from config import MINIMAP_SIZE, MINIMAP_MARGIN, WINDOW_WIDTH, WINDOW_HEIGHT


class Minimap:
    _SIZE    = MINIMAP_SIZE
    _MARGIN  = MINIMAP_MARGIN
    _FOV_LEN = 24   # px — calibration: at pitch=0 the bottom of the view hits ground here

    def __init__(self, tile_map, all_nodes: list):
        self.tile_map  = tile_map
        self.all_nodes = all_nodes
        pygame.font.init()

    # ------------------------------------------------------------------
    # Ground projection
    # ------------------------------------------------------------------

    def _virtual_h(self, fov_h_deg: float) -> float:
        fov_v_deg = math.degrees(
            2.0 * math.atan(math.tan(math.radians(fov_h_deg / 2.0)) * WINDOW_HEIGHT / WINDOW_WIDTH)
        )
        return self._FOV_LEN * math.tan(math.radians(fov_v_deg / 2.0))

    def _ground_pts(self, cx: int, cy: int,
                    heading_deg: float, pitch_deg: float,
                    fov_h_deg: float, virtual_h: float) -> list:
        """Return [BL, BR, TR, TL] minimap pixel positions for the 4 frustum corners."""
        fov_v_deg = math.degrees(
            2.0 * math.atan(math.tan(math.radians(fov_h_deg / 2.0)) * WINDOW_HEIGHT / WINDOW_WIDTH)
        )
        H  = math.radians(heading_deg)
        P  = math.radians(pitch_deg)
        ch, sh = math.cos(H), math.sin(H)
        cp, sp = math.cos(P), math.sin(P)

        right = ( ch,    -sh,    0.0)
        fwd   = ( sh*cp,  ch*cp, sp)
        up    = (-sh*sp, -ch*sp, cp)

        tan_h = math.tan(math.radians(fov_h_deg / 2.0))
        tan_v = math.tan(math.radians(fov_v_deg / 2.0))
        max_d = self._FOV_LEN * 4.0

        pts = []
        for ccx, ccy in [(-tan_h, -tan_v), (+tan_h, -tan_v),
                          (+tan_h, +tan_v), (-tan_h, +tan_v)]:
            wx = ccx*right[0] + ccy*up[0] + fwd[0]
            wy = ccx*right[1] + ccy*up[1] + fwd[1]
            wz = ccx*right[2] + ccy*up[2] + fwd[2]

            horiz = math.sqrt(wx*wx + wy*wy)
            if wz < -1e-6:
                t = -virtual_h / wz
            else:
                t = max_d / max(horiz, 1e-6)
            t = min(t, max_d / max(horiz, 1e-6))

            pts.append((cx + int(t * wx), cy - int(t * wy)))
        return pts  # BL, BR, TR, TL

    # ------------------------------------------------------------------
    # Drawing helper
    # ------------------------------------------------------------------

    def _draw_prismatoid(self, target: pygame.Surface,
                         cx: int, cy: int,
                         heading_deg: float, pitch_deg: float,
                         fov_deg: float, virtual_h: float,
                         fill_rgba: tuple, line_rgba: tuple,
                         near_rgba: tuple) -> None:
        BL, BR, TR, TL = self._ground_pts(cx, cy, heading_deg, pitch_deg, fov_deg, virtual_h)
        S    = target.get_width()
        surf = pygame.Surface((S, S), pygame.SRCALPHA)
        pygame.draw.polygon(surf, fill_rgba, [BL, BR, TR, TL])
        pygame.draw.line(surf, line_rgba, (cx, cy), TL, 1)
        pygame.draw.line(surf, line_rgba, (cx, cy), TR, 1)
        pygame.draw.line(surf, near_rgba,  BL, BR, 1)
        pygame.draw.line(surf, line_rgba,  TL, TR, 1)
        target.blit(surf, (0, 0))

    # ------------------------------------------------------------------
    # Public render
    # ------------------------------------------------------------------

    def render(self, surface: pygame.Surface,
               player_lat: float, player_lon: float, current_node,
               heading_deg: float, fov_deg: float = 90.0, pitch_deg: float = 0.0,
               anchors: list = None) -> None:
        """anchors: list of (node, crop_heading_deg, crop_fov_deg) tuples."""
        S = self._SIZE
        W, H = surface.get_size()
        x0 = W - S - self._MARGIN
        y0 = H - S - self._MARGIN

        mini = pygame.Surface((S, S))
        mini.fill((35, 35, 45))

        # All positions are relative to the player's tile pixel (minimap centers on player)
        if self.tile_map.available:
            ppx, ppy = self.tile_map.latlon_to_pane_px(player_lat, player_lon)
        else:
            ppx, ppy = 0, 0

        # ── Tile background (centred on player) ───────────────────────
        if self.tile_map.available:
            half = S // 2
            tw, th = self.tile_map.surface.get_size()
            src_x = max(0, min(tw - S, ppx - half))
            src_y = max(0, min(th - S, ppy - half))
            dst_x = max(0, half - ppx)
            dst_y = max(0, half - ppy)
            mini.blit(self.tile_map.surface, (dst_x, dst_y), pygame.Rect(src_x, src_y, S, S))

        # ── Node dots ─────────────────────────────────────────────────
        if self.tile_map.available:
            for node in self.all_nodes:
                nx, ny = self.tile_map.latlon_to_pane_px(node.lat, node.lon)
                mx = S // 2 + (nx - ppx)
                my = S // 2 + (ny - ppy)
                if 0 <= mx < S and 0 <= my < S:
                    col = (60, 200, 140) if node is current_node else (80, 120, 210)
                    r   = 5 if node is current_node else 3
                    pygame.draw.circle(mini, col, (mx, my), r)
        else:
            lats = [n.lat for n in self.all_nodes]
            lons = [n.lon for n in self.all_nodes]
            lat_r = max(lats) - min(lats) or 1e-5
            lon_r = max(lons) - min(lons) or 1e-5
            scale = S * 0.8
            for node in self.all_nodes:
                mx = int(S * 0.1 + (node.lon - min(lons)) / lon_r * scale)
                my = int(S * 0.9 - (node.lat - min(lats)) / lat_r * scale)
                if 0 <= mx < S and 0 <= my < S:
                    col = (60, 200, 140) if node is current_node else (80, 120, 210)
                    r   = 5 if node is current_node else 2
                    pygame.draw.circle(mini, col, (mx, my), r)

        cx, cy = S // 2, S // 2
        vh     = self._virtual_h(fov_deg)

        # ── Anchor 3D prismatoid cones (pitch=0 — ground-level cameras) ──
        _anchor_styles = [
            ((60, 200, 140, 30), (60, 200, 140, 150), (60, 200, 140, 100)),
            ((60, 140, 220, 30), (60, 140, 220, 150), (60, 140, 220, 100)),
            ((200, 80, 220, 30), (200, 80, 220, 150), (200, 80, 220, 100)),
        ]
        if self.tile_map.available and anchors:
            for i, (anchor, anchor_hdg, anchor_fov) in enumerate(anchors):
                if anchor is None:
                    continue
                fill, line, near = _anchor_styles[i % len(_anchor_styles)]
                nx, ny = self.tile_map.latlon_to_pane_px(anchor.lat, anchor.lon)
                ax = S // 2 + (nx - ppx)
                ay = S // 2 + (ny - ppy)
                avh = self._virtual_h(anchor_fov)
                self._draw_prismatoid(mini, ax, ay, heading_deg, pitch_deg, anchor_fov, avh,
                                      fill, line, near)

        # ── Player 3D frustum ground footprint ────────────────────────
        self._draw_prismatoid(mini, cx, cy, heading_deg, pitch_deg, fov_deg, vh,
                              (255, 200, 50, 40), (255, 200, 50, 180), (255, 200, 50, 120))

        pygame.draw.circle(mini, (255, 200, 50), (cx, cy), 4)

        # ── Border ────────────────────────────────────────────────────
        pygame.draw.rect(mini, (160, 160, 160), (0, 0, S, S), 1)
        surface.blit(mini, (x0, y0))
