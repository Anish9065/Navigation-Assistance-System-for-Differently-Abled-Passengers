"""
Navigator — graph-based station navigation for railway stations.

Zones are mapped to regions of the video frame.
The graph defines walkable connections between zones.
"""

from config.settings import Settings


# ── Station Graph ─────────────────────────────────────────────────
#  Each edge: (from_zone, to_zone, instruction, direction_arrow)
STATION_GRAPH = {
    'entrance': [
        ('main_hall',      'Walk straight into the main hall',               'down'),
        ('ticket_counter', 'Turn right to reach the ticket counter',         'right'),
        ('restroom',       'Turn left for the restroom',                     'left'),
    ],
    'main_hall': [
        ('entrance',       'Walk towards the main entrance',                  'up'),
        ('platform_1',     'Turn left and proceed to Platform 1',            'left'),
        ('platform_2',     'Turn right and proceed to Platform 2',           'right'),
        ('ticket_counter', 'Move right to reach the ticket counter',         'up-right'),
        ('waiting_area',   'Walk forward to the waiting area',               'down'),
        ('elevator',       'Walk to the center for the elevator',            'down'),
        ('restroom',       'Move to the upper-left for the restroom',        'up-left'),
        ('exit',           'Walk straight ahead to the exit',                'down'),
    ],
    'ticket_counter': [
        ('main_hall',      'Move left into the main hall',                   'left'),
        ('entrance',       'Walk towards the entrance',                      'down-left'),
    ],
    'platform_1': [
        ('main_hall',      'Walk right into the main hall',                  'right'),
        ('elevator',       'Use the elevator in the center of the hall',     'right'),
        ('waiting_area',   'Walk right to the waiting area',                 'right'),
    ],
    'platform_2': [
        ('main_hall',      'Walk left into the main hall',                   'left'),
        ('elevator',       'Use the elevator in the center of the hall',     'left'),
        ('waiting_area',   'Walk left to the waiting area',                  'left'),
    ],
    'restroom': [
        ('entrance',       'Walk towards the entrance',                      'right'),
        ('main_hall',      'Walk right into the main hall',                  'right'),
    ],
    'elevator': [
        ('platform_1',     'Take the elevator then walk left to Platform 1', 'left'),
        ('platform_2',     'Take the elevator then walk right to Platform 2','right'),
        ('main_hall',      'Walk forward into the main hall',                'up'),
    ],
    'waiting_area': [
        ('main_hall',      'Walk forward into the main hall',                'up'),
        ('platform_1',     'Walk left to Platform 1',                        'left'),
        ('platform_2',     'Walk right to Platform 2',                       'right'),
        ('exit',           'Walk forward to the exit',                       'down'),
    ],
    'exit': [
        ('main_hall',      'Walk back into the main hall',                   'up'),
        ('waiting_area',   'Walk forward to the waiting area',               'up'),
    ],
}

ZONE_DISPLAY_NAMES = {
    'entrance':       'Entrance',
    'main_hall':      'Main Hall',
    'ticket_counter': 'Ticket Counter',
    'platform_1':     'Platform 1',
    'platform_2':     'Platform 2',
    'restroom':       'Restroom / Washroom',
    'elevator':       'Elevator / Lift',
    'waiting_area':   'Waiting Area',
    'exit':           'Exit',
}


class Navigator:

    # ── Zone from position ────────────────────────────────────────
    def get_zone(self, fx: float, fy: float) -> str:
        """Return zone name for a normalised (fx, fy) position."""
        for zone, (x1, y1, x2, y2) in Settings.ZONES.items():
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                return zone
        return 'main_hall'

    # ── Arrow for one step ────────────────────────────────────────
    def get_arrow_direction(self, current_zone: str, destination: str) -> str:
        if current_zone == destination:
            return 'arrived'
        for (nxt, _, arrow) in STATION_GRAPH.get(current_zone, []):
            if nxt == destination:
                return arrow
        # multi-hop: try one intermediate step
        path, _ = self.get_route(current_zone, destination)
        if len(path) >= 2:
            for (nxt, _, arrow) in STATION_GRAPH.get(current_zone, []):
                if nxt == path[1]:
                    return arrow
        return 'up'

    # ── BFS Route ────────────────────────────────────────────────
    def get_route(self, src: str, dst: str):
        if src == dst:
            return [src], ["You have arrived at your destination."]
        from collections import deque
        visited = {src: None}
        queue   = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                break
            for (nxt, _, _) in STATION_GRAPH.get(node, []):
                if nxt not in visited:
                    visited[nxt] = node
                    queue.append(nxt)

        # Reconstruct path
        path = []
        cur  = dst
        while cur is not None:
            path.append(cur)
            cur = visited.get(cur)
        path.reverse()

        if path[0] != src:
            return [src, dst], [f"Direct route not found. Head towards {ZONE_DISPLAY_NAMES.get(dst, dst)}."]

        # Build step instructions
        steps = []
        for i in range(len(path) - 1):
            for (nxt, instr, _) in STATION_GRAPH.get(path[i], []):
                if nxt == path[i + 1]:
                    steps.append(instr)
                    break

        if not steps:
            steps = [f"Head towards {ZONE_DISPLAY_NAMES.get(dst, dst)}."]

        return path, steps

    # ── Directions from detections ────────────────────────────────
    def get_directions_from_detections(self, detections: list, frame_shape: tuple,
                                       destination: str = 'platform_1') -> list:
        h, w = frame_shape[:2]
        directions = []
        for det in detections:
            cx, cy = det['center']
            fx, fy = cx / w, cy / h
            zone   = self.get_zone(fx, fy)
            _, steps = self.get_route(zone, destination)
            directions.append({
                'class':       det['class_name'],
                'zone':        ZONE_DISPLAY_NAMES.get(zone, zone),
                'instruction': steps[0] if steps else 'Proceed to destination',
                'confidence':  det['confidence'],
            })
        return directions

    # ── Station map for frontend ──────────────────────────────────
    def get_station_map(self) -> dict:
        nodes = []
        for zone, name in ZONE_DISPLAY_NAMES.items():
            x1, y1, x2, y2 = Settings.ZONES[zone]
            nodes.append({
                'id':   zone,
                'name': name,
                'x':    (x1 + x2) / 2,
                'y':    (y1 + y2) / 2,
            })
        edges = []
        for src, connections in STATION_GRAPH.items():
            for (dst, instr, _) in connections:
                edges.append({'from': src, 'to': dst, 'instruction': instr})
        return {'nodes': nodes, 'edges': edges, 'zone_names': ZONE_DISPLAY_NAMES}
