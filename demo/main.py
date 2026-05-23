import argparse
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pygame

from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, PANEL_SECTION_W,
    YAW_SENS, PITCH_SENS, PITCH_LIMIT,
    ELEVATION_RATE, MOVE_SPEED,
    EARTH_RADIUS_M, TARGET_FPS, DATA_FOLDER,
    MAX_STEER_DEG, FOV_DEG,
)
from loader import load_scene, build_graph
from tile_map import TileMap
from viewer import StreetViewer
from minimap import Minimap
from cue_engine import CueEngine
from cue_panel import CuePanel
from utils import bearing_between, haversine_distance


class _Player:
    """Tracks free-roaming position; heading/pitch are owned by the viewer."""
    def __init__(self, lat: float, lon: float):
        self.lat           = lat
        self.lon           = lon
        self.heading_deg   = 0.0
        self.elevation_deg = 0.0
        self.speed_mps     = 0.0
        self.dx_m          = 0.0
        self.dy_m          = 0.0
        self.delta_heading = 0.0
        self.steer         = 0.0   # normalized –1 to 1 (Cue 3)
        self._prev_heading = 0.0

    def update(self, dt: float, keys, viewer) -> None:
        # UP/DOWN arrows — tilt
        if keys[pygame.K_UP]:
            viewer.pitch_deg = min(viewer.pitch_deg + ELEVATION_RATE * dt,  PITCH_LIMIT)
            viewer._dirty = True
        if keys[pygame.K_DOWN]:
            viewer.pitch_deg = max(viewer.pitch_deg - ELEVATION_RATE * dt, -PITCH_LIMIT)
            viewer._dirty = True

        # W/S — forward/backward,  A/D — strafe left/right
        fwd_spd    = 0.0
        strafe_spd = 0.0
        if keys[pygame.K_w]:  fwd_spd    =  MOVE_SPEED
        if keys[pygame.K_s]:  fwd_spd    = -MOVE_SPEED
        if keys[pygame.K_a]:  strafe_spd = -MOVE_SPEED   # left
        if keys[pygame.K_d]:  strafe_spd =  MOVE_SPEED   # right

        self.speed_mps = math.hypot(fwd_spd, strafe_spd)

        h_rad      = math.radians(viewer.heading_deg)
        s_rad      = math.radians(viewer.heading_deg + 90.0)   # perpendicular right
        lat_rad    = math.radians(self.lat)
        cos_lat    = math.cos(lat_rad)

        fwd_dist    = fwd_spd    * dt
        strafe_dist = strafe_spd * dt

        dlat = (fwd_dist * math.cos(h_rad) + strafe_dist * math.cos(s_rad))
        dlon = (fwd_dist * math.sin(h_rad) + strafe_dist * math.sin(s_rad))

        self.lat  += dlat / EARTH_RADIUS_M * (180.0 / math.pi)
        if cos_lat > 1e-9:
            self.lon += dlon / (EARTH_RADIUS_M * cos_lat) * (180.0 / math.pi)

        self.dx_m          = dlon
        self.dy_m          = dlat
        self.heading_deg   = viewer.heading_deg
        self.elevation_deg = viewer.pitch_deg
        self.delta_heading = (self.heading_deg - self._prev_heading + 180.0) % 360.0 - 180.0
        self.steer         = max(-1.0, min(1.0, self.delta_heading / MAX_STEER_DEG))
        self._prev_heading = self.heading_deg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Path to lcm_final.pt. If omitted, shows cue panel only.")
    parser.add_argument("--steps", type=int, default=1,
                        help="LCM inference steps (default 1 for CPU speed)")
    args = parser.parse_args()

    # Auto-detect checkpoint in repo root if not specified
    if args.model is None:
        default_ckpt = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "lcm_final.pt"
        )
        if os.path.isfile(default_ckpt):
            args.model = default_ckpt

    folder_path = DATA_FOLDER
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory")
        sys.exit(1)

    print("Loading scene…")
    nodes = load_scene(folder_path)
    build_graph(nodes)
    print(f"  {len(nodes)} nodes loaded")

    center_lat = sum(n.lat for n in nodes) / len(nodes)
    center_lon = sum(n.lon for n in nodes) / len(nodes)

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".tile_cache")
    tile_map  = TileMap(center_lat, center_lon, WINDOW_WIDTH, WINDOW_HEIGHT,
                        cache_dir=cache_dir, nodes=nodes)

    total_w = WINDOW_WIDTH + PANEL_SECTION_W

    pygame.init()
    screen   = pygame.display.set_mode((total_w, WINDOW_HEIGHT))
    pygame.display.set_caption("Street View")
    clock    = pygame.time.Clock()
    fps_font = pygame.font.SysFont("monospace", 11)

    viewer     = StreetViewer(nodes, WINDOW_WIDTH, WINDOW_HEIGHT)
    minimap    = Minimap(tile_map, nodes)
    cue_engine = CueEngine(nodes)
    cue_panel  = CuePanel(WINDOW_WIDTH, 0, PANEL_SECTION_W, WINDOW_HEIGHT)
    player     = _Player(nodes[0].lat, nodes[0].lon)
    gsv_surf   = screen.subsurface(pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))

    wm_interface = None
    if args.model:
        try:
            from world_model.queue_interface import WorldModelInterface
            wm_output_surf = screen.subsurface(
                pygame.Rect(WINDOW_WIDTH - 256, 0, 256, 256)  # top-right, clear of minimap
            )
            wm_interface = WorldModelInterface(
                model_checkpoint=args.model,
                display_surface=wm_output_surf,
                steps=args.steps,
            )
            print(f"World model loaded: {args.model}  (steps={args.steps})")
        except Exception as e:
            print(f"World model load failed: {e} — running without model")

    if len(nodes) >= 2:
        viewer.heading_deg = bearing_between(
            nodes[0].lat, nodes[0].lon, nodes[1].lat, nodes[1].lon,
        )

    try:
     _run_loop(screen, clock, fps_font, viewer, minimap, cue_engine, cue_panel,
               player, gsv_surf, nodes, wm_interface)
    finally:
        if wm_interface is not None:
            wm_interface.stop()
        pygame.quit()


def _run_loop(screen, clock, fps_font, viewer, minimap, cue_engine, cue_panel,
              player, gsv_surf, nodes, wm_interface):
    from config import (
        WINDOW_WIDTH, WINDOW_HEIGHT, YAW_SENS, PITCH_SENS, PITCH_LIMIT,
        ELEVATION_RATE, FOV_DEG, TARGET_FPS,
    )
    dragging   = False
    drag_start = (0, 0)
    drag_moved = False
    while True:
        dt        = clock.tick(TARGET_FPS) / 1000.0
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
                elif event.key == pygame.K_r:
                    viewer.heading_deg = 0.0
                    viewer.pitch_deg   = 0.0
                    viewer.fov_deg     = 90.0
                    viewer._dirty      = True

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if mouse_pos[0] < WINDOW_WIDTH:
                    dragging   = True
                    drag_start = event.pos
                    drag_moved = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx = event.pos[0] - drag_start[0]
                    dy = event.pos[1] - drag_start[1]
                    if abs(dx) > 2 or abs(dy) > 2:
                        drag_moved = True
                    viewer.rotate(dx * YAW_SENS, -dy * PITCH_SENS)
                    drag_start = event.pos

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
                if not drag_moved and mouse_pos[0] < WINDOW_WIDTH:
                    node = viewer.node_at_screen(*event.pos)
                    if node:
                        viewer.navigate_to(node)
                        player.lat = node.lat
                        player.lon = node.lon

            elif event.type == pygame.MOUSEWHEEL:
                if mouse_pos[0] < WINDOW_WIDTH:
                    viewer.zoom(-event.y * 5.0)

        # WASD update
        keys = pygame.key.get_pressed()
        player.update(dt, keys, viewer)

        # Auto-snap viewer to nearest node as player moves
        nearest = min(nodes, key=lambda n: haversine_distance(player.lat, player.lon, n.lat, n.lon))
        if nearest is not viewer.current_node:
            viewer.set_node(nearest)

        # Cue data
        cue_data  = cue_engine.update(player)
        node_map  = {n.id: n for n in nodes}
        n1 = node_map.get(cue_data.nearest_node_id)
        n2 = node_map.get(cue_data.second_node_id)
        n3 = node_map.get(cue_data.third_node_id)
        # Anchors for minimap prismatoids — all share player heading (anchor crops are at player direction)
        anchors = [
            (n1, player.heading_deg, FOV_DEG),
            (n2, player.heading_deg, FOV_DEG),
            (n3, player.heading_deg, FOV_DEG),
        ]

        # GSV side
        viewer.render(gsv_surf, mouse_pos)
        minimap.render(gsv_surf, player.lat, player.lon, viewer.current_node,
                       viewer.heading_deg, viewer.fov_deg, viewer.pitch_deg,
                       anchors)

        # Cue panel side
        cue_panel.render(screen, cue_data)

        # World model — send cues and blit latest keyframe (bottom-right overlay)
        if wm_interface is not None:
            wm_interface.send(cue_data)

        # Divider
        pygame.draw.line(screen, (60, 60, 80),
                         (WINDOW_WIDTH, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), 2)

        fps = clock.get_fps()
        fps_surf = fps_font.render(f"{fps:.0f} fps", True, (160, 160, 160))
        screen.blit(fps_surf, (WINDOW_WIDTH - fps_surf.get_width() - 10,
                               WINDOW_HEIGHT - fps_surf.get_height() - 8))

        pygame.display.flip()



if __name__ == "__main__":
    main()
