"""
Downloads and caches OpenStreetMap-compatible raster tiles at startup,
stitches them into a single pygame Surface, and provides Mercator-correct
GPS → pane-pixel conversion so nodes land on actual streets.
"""
import os
import math
import ssl
import urllib.request
import numpy as np
import cv2
import pygame
from pathlib import Path

# CartoDB Dark Matter matches the app's dark theme perfectly.
# Swap to OSM_URL for the default light map.
CARTO_DARK_URL = "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
OSM_URL        = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

TILE_SIZE  = 256
USER_AGENT = "WorldModelInputDemo/1.0 (educational use)"


class TileMap:
    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        pane_w: int,
        pane_h: int,
        cache_dir: str,
        nodes=None,          # used for auto-zoom if provided
        tile_url: str = CARTO_DARK_URL,
    ):
        self.pane_w   = pane_w
        self.pane_h   = pane_h
        self.tile_url = tile_url
        self.available = False
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._cache = cache_dir
        self._ssl   = ssl._create_unverified_context()

        # Pick zoom so the full node cluster fits in ~70 % of the pane
        self.zoom = self._pick_zoom(center_lat, center_lon, nodes)

        # Scene centre in Web-Mercator world pixels at chosen zoom
        self.cx_wpx, self.cy_wpy = self._to_world(center_lat, center_lon)

        print(f"  Fetching map tiles (zoom {self.zoom}, "
              f"~{self._metres_per_pixel():.1f} m/px)…")
        self.surface = self._build_surface()

    # ------------------------------------------------------------------
    # Coordinate maths
    # ------------------------------------------------------------------

    def _to_world(self, lat: float, lon: float) -> tuple[float, float]:
        """GPS → Web-Mercator world-pixel coordinates at self.zoom."""
        n   = (2 ** self.zoom) * TILE_SIZE
        wpx = (lon + 180.0) / 360.0 * n
        lr  = math.radians(lat)
        wpy = (1.0 - math.log(math.tan(lr) + 1.0 / math.cos(lr)) / math.pi) / 2.0 * n
        return wpx, wpy

    def latlon_to_pane_px(self, lat: float, lon: float) -> tuple[int, int]:
        """Mercator-correct GPS → pixel position within the map pane."""
        wpx, wpy = self._to_world(lat, lon)
        return (
            int(wpx - self.cx_wpx + self.pane_w / 2),
            int(wpy - self.cy_wpy + self.pane_h / 2),
        )

    def _metres_per_pixel(self) -> float:
        # At the Equator: 40075016 m / (2^zoom * 256 px). Corrected for latitude
        # via centre lat (not stored — approximate from world coords).
        equator_mpp = 40_075_016.0 / ((2 ** self.zoom) * TILE_SIZE)
        return equator_mpp  # caller uses this for display only

    def _scene_px_span(self, nodes, zoom: int) -> tuple[float, float]:
        """Pixel span of the node bounding box at the given zoom."""
        if not nodes:
            return 0.0, 0.0
        lats = [n.lat for n in nodes]
        lons = [n.lon for n in nodes]
        ts   = TILE_SIZE
        n2   = (2 ** zoom) * ts

        def wx(lon):  return (lon + 180) / 360 * n2
        def wy(lat):
            lr = math.radians(lat)
            return (1 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi) / 2 * n2

        return abs(wx(max(lons)) - wx(min(lons))), abs(wy(min(lats)) - wy(max(lats)))

    def _pick_zoom(self, center_lat, center_lon, nodes) -> int:
        if nodes is None:
            return 16
        for z in range(18, 12, -1):
            span_x, span_y = self._scene_px_span(nodes, z)
            # Add padding: aim for scene to occupy ≤ 65 % of pane
            if span_x <= self.pane_w * 0.65 and span_y <= self.pane_h * 0.65:
                return z
        return 14

    # ------------------------------------------------------------------
    # Tile fetching & stitching
    # ------------------------------------------------------------------

    def _fetch_tile(self, tx: int, ty: int) -> np.ndarray | None:
        n_tiles = 2 ** self.zoom
        tx = tx % n_tiles
        if ty < 0 or ty >= n_tiles:
            return None

        path = os.path.join(self._cache, f"{self.zoom}_{tx}_{ty}.png")
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return img

        try:
            url = self.tile_url.format(z=self.zoom, x=tx, y=ty)
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=15, context=self._ssl) as r:
                data = r.read()
            Path(path).write_bytes(data)
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"    tile ({tx},{ty}) failed: {e}")
            return None

    def _build_surface(self) -> pygame.Surface:
        ts = TILE_SIZE
        # Tile indices that cover the pane
        tx0 = int((self.cx_wpx - self.pane_w / 2) / ts)
        tx1 = int((self.cx_wpx + self.pane_w / 2) / ts) + 1
        ty0 = int((self.cy_wpy - self.pane_h / 2) / ts)
        ty1 = int((self.cy_wpy + self.pane_h / 2) / ts) + 1

        cols, rows = tx1 - tx0 + 1, ty1 - ty0 + 1
        canvas = np.full((rows * ts, cols * ts, 3), 28, dtype=np.uint8)

        ok = 0
        total = cols * rows
        for row, ty in enumerate(range(ty0, ty1 + 1)):
            for col, tx in enumerate(range(tx0, tx1 + 1)):
                tile = self._fetch_tile(tx, ty)
                if tile is not None:
                    h, w = tile.shape[:2]
                    dy, dx = row * ts, col * ts
                    canvas[dy:dy + h, dx:dx + w] = tile[:h, :w]
                    ok += 1

        print(f"  {ok}/{total} tiles loaded.")
        self.available = ok > 0

        # Crop canvas to pane dimensions, centred on cx/cy
        crop_x = max(0, int(self.cx_wpx - tx0 * ts - self.pane_w / 2))
        crop_y = max(0, int(self.cy_wpy - ty0 * ts - self.pane_h / 2))
        cropped = canvas[crop_y:crop_y + self.pane_h, crop_x:crop_x + self.pane_w]

        # Pad if world edge was hit
        if cropped.shape[:2] != (self.pane_h, self.pane_w):
            padded = np.full((self.pane_h, self.pane_w, 3), 28, dtype=np.uint8)
            h, w = cropped.shape[:2]
            padded[:h, :w] = cropped
            cropped = padded

        arr = np.ascontiguousarray(cropped[:, :, ::-1].transpose(1, 0, 2))
        return pygame.surfarray.make_surface(arr)
