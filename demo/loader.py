import json
import os
import numpy as np
import cv2
from dataclasses import dataclass, field
from config import MAP_PADDING


@dataclass
class Node:
    id: str
    lat: float
    lon: float
    image_path: str
    image_bgr: np.ndarray
    compass_angle: float = 0.0   # direction camera faced at capture (degrees CW from North)
    screen_pos: tuple = field(default=(0, 0))


def load_scene(folder_path: str) -> list[Node]:
    coords_path = os.path.join(folder_path, "coordinates.json")
    if not os.path.exists(coords_path):
        print(f"Error: coordinates.json not found in '{folder_path}'")
        raise SystemExit(1)

    try:
        with open(coords_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse coordinates.json: {e}")
        raise SystemExit(1)

    nodes: list[Node] = []
    for item in data:
        node_id = str(item.get("id", "?"))
        img_filename = item.get("image", "")
        img_path = os.path.join(folder_path, img_filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load '{img_path}', skipping node '{node_id}'")
            continue

        h, w = img.shape[:2]
        if h > 0 and abs(w / h - 2.0) > 0.1:
            print(f"Warning: '{img_filename}' aspect ratio {w/h:.2f} is not 2:1 — sphere mapping will be inaccurate")

        nodes.append(Node(
            id=node_id,
            lat=float(item["lat"]),
            lon=float(item["lon"]),
            image_path=img_path,
            image_bgr=img,
            compass_angle=float(item.get("compass_angle", 0.0)),
        ))

    if len(nodes) < 2:
        print(f"Error: Need at least 2 valid nodes, only {len(nodes)} loaded")
        raise SystemExit(1)

    return nodes


def compute_scene_bounds(nodes: list[Node]) -> tuple[float, float, float, float]:
    lats = [n.lat for n in nodes]
    lons = [n.lon for n in nodes]
    return min(lats), max(lats), min(lons), max(lons)


def latlon_to_screen(lat, lon, bounds, map_width, map_height, padding=MAP_PADDING):
    min_lat, max_lat, min_lon, max_lon = bounds
    uw = map_width - 2 * padding
    uh = map_height - 2 * padding

    px = padding + uw // 2 if max_lon == min_lon else int((lon - min_lon) / (max_lon - min_lon) * uw) + padding
    py = padding + uh // 2 if max_lat == min_lat else int((1.0 - (lat - min_lat) / (max_lat - min_lat)) * uh) + padding
    return px, py


def compute_screen_positions(nodes: list[Node], map_width, map_height) -> tuple[list[Node], tuple]:
    bounds = compute_scene_bounds(nodes)
    for node in nodes:
        node.screen_pos = latlon_to_screen(node.lat, node.lon, bounds, map_width, map_height)
    return nodes, bounds


def build_graph(nodes: list[Node]) -> dict[int, list[int]]:
    """Bidirectional nearest-neighbor graph. Each node connects to its closest peer."""
    from utils import haversine_distance
    graph: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for i, a in enumerate(nodes):
        others = [j for j in range(len(nodes)) if j != i]
        best_j = min(others, key=lambda j: haversine_distance(a.lat, a.lon, nodes[j].lat, nodes[j].lon), default=-1)
        if best_j >= 0:
            if best_j not in graph[i]:
                graph[i].append(best_j)
            if i not in graph[best_j]:
                graph[best_j].append(i)
    return graph
