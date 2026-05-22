WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720

# Cue panel section (appended to the right of the GSV viewer)
PANEL_W         = 256
PANEL_H         = 228
PANEL_PADDING   = 10
PANEL_GAP       = 8
PANEL_SECTION_W = PANEL_PADDING + PANEL_W + PANEL_GAP + PANEL_W + PANEL_PADDING  # 540

# Viewer
DEFAULT_FOV  = 90.0
MIN_FOV      = 20.0
MAX_FOV      = 120.0
PITCH_LIMIT  = 85.0

# Mouse sensitivity
YAW_SENS   = 0.20   # deg per pixel drag
PITCH_SENS = 0.20

# WASD movement
TURN_RATE      = 90.0   # deg/s  (A/D)
ELEVATION_RATE = 60.0   # deg/s  (UP/DOWN arrows)
MOVE_SPEED     = 3.0    # m/s    (W/S)

# Navigation arrows
MAX_NAV_NODES  = 5
MAX_NAV_DIST_M = 80.0
ARROW_PITCH    = -20.0   # elevation angle where nav arrows appear
ARROW_HIT_PX   = 36      # click hit radius in pixels

# Minimap overlay
MINIMAP_SIZE   = 200
MINIMAP_MARGIN = 14

# Cue engine
NEAREST_CACHE_DIST_M = 5.0
ANCHOR_CROP_SIZE     = 256
FOV_DEG              = 90.0
LOOKAHEAD_TIME_S     = 0.5    # seconds ahead for interpolation grounding crop
LOOKAHEAD_FOV_DEG    = 120.0  # wider FOV for look-ahead crop
MAX_STEER_DEG        = 10.0   # delta_heading (deg/frame) that maps to steer = ±1

# Colors
PLAYER_COLOR = (255, 200, 50)

# Data source
DATA_FOLDER = "/Users/a/GitHub/InfiniteRace-model-input-demo/gsv_data"

# Kept for loader.py compatibility
MAP_PADDING    = 40
EARTH_RADIUS_M = 6_371_000
TARGET_FPS     = 60
