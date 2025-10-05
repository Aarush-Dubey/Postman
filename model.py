import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from heapq import heappush, heappop

ROBOFLOW_API_KEY = "PQA0gaPVZE9Op07R5IB7"

class LunarPathPlanner:
    def __init__(self, image_path, sun_azimuth=45):
        self.image_path = image_path
        self.sun_azimuth = sun_azimuth
        self.sun_direction = np.array([np.cos(np.radians(sun_azimuth)), 
                                       -np.sin(np.radians(sun_azimuth))])
        self.image = None
        self.craters = []
        self.path = []
        self.stops = []
        self.shadows = []
        
    def detect_craters(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print("Error loading image")
            return
        
        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
        
        print("Detecting craters...")
        result = client.infer(self.image_path, model_id="chandrayaan-2-ohrc-lunar-crater-dataset-begao/2")
        
        for pred in result.get('predictions', []):
            x = int(pred['x'])
            y = int(pred['y'])
            w = int(pred['width'])
            h = int(pred['height'])
            radius = int((w + h) / 4)
            
            self.craters.append({
                'center': (x, y),
                'radius': radius
            })
        
        print(f"Found {len(self.craters)} craters")
        self.calculate_shadows()
        
    def calculate_shadows(self):
        for crater in self.craters:
            cx, cy = crater['center']
            r = crater['radius']
            
            shadow_len = r * 1.5
            shadow_x = int(cx - self.sun_direction[0] * shadow_len)
            shadow_y = int(cy - self.sun_direction[1] * shadow_len)
            
            self.shadows.append({
                'crater': (cx, cy),
                'end': (shadow_x, shadow_y),
                'radius': r,
                'width': r * 0.6
            })
    
    def in_shadow(self, point):
        px, py = point
        for shadow in self.shadows:
            cx, cy = shadow['crater']
            sx, sy = shadow['end']
            
            vec_point = np.array([px - cx, py - cy])
            vec_shadow = np.array([sx - cx, sy - cy])
            
            if np.linalg.norm(vec_shadow) > 0:
                proj = np.dot(vec_point, vec_shadow) / np.linalg.norm(vec_shadow)
                
                if 0 < proj < np.linalg.norm(vec_shadow):
                    perp = np.abs(np.cross(vec_point, vec_shadow / np.linalg.norm(vec_shadow)))
                    if perp < shadow['width']:
                        return True
        return False
    
    def is_safe(self, point):
        px, py = point
        h, w = self.image.shape[:2]
        
        if px < 0 or px >= w or py < 0 or py >= h:
            return False
        
        # check craters
        for crater in self.craters:
            cx, cy = crater['center']
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            if dist < crater['radius'] + 20:
                return False
        
        if self.in_shadow(point):
            return False
        
        return True
    
    def find_path(self, start, goal):
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        
        step = 15
        directions = [
            (step, 0), (step, step), (0, step), 
            (-step, step), (-step, 0), (-step, -step),
            (0, -step), (step, -step)
        ]
        
        visited = set()
        max_iter = 15000
        iterations = 0
        
        while open_set and iterations < max_iter:
            iterations += 1
            _, current = heappop(open_set)
            
            if current in visited:
                continue
            visited.add(current)
            
            # check if reached goal
            dist = np.sqrt((current[0] - goal[0])**2 + (current[1] - goal[1])**2)
            if dist < step * 3:
                path = [goal]
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                return path[::-1]
            
            # check neighbors
            for dx, dy in directions:
                nx = current[0] + dx
                ny = current[1] + dy
                neighbor = (nx, ny)
                
                if neighbor in visited:
                    continue
                
                if not self.is_safe(neighbor):
                    continue
                
                move_cost = np.sqrt(dx**2 + dy**2)
                heuristic = np.sqrt((nx - goal[0])**2 + (ny - goal[1])**2)
                
                penalty = 0
                if self.in_shadow(neighbor):
                    penalty = 500
                
                # check nearby craters
                for crater in self.craters[:5]:
                    cx, cy = crater['center']
                    d = np.sqrt((nx - cx)**2 + (ny - cy)**2)
                    if d < crater['radius'] + 50:
                        penalty += 30
                        break
                
                tentative_g = g_score[current] + move_cost + penalty
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic
                    heappush(open_set, (f, neighbor))
        
        print("Path not found")
        return None
    
    def identify_stops(self, num_stops=10):
        if len(self.path) < num_stops:
            num_stops = max(1, len(self.path) // 2)
        
        path_len = len(self.path)
        indices = np.linspace(int(path_len * 0.1), int(path_len * 0.95), num_stops, dtype=int)
        
        for i, idx in enumerate(indices):
            point = self.path[idx]
            stop_type = self.get_stop_type(point, i, num_stops)
            
            self.stops.append({
                'pos': point,
                'type': stop_type
            })
        
        print(f"Identified {len(self.stops)} stops")
    
    def get_stop_type(self, point, num, total):
        px, py = point
        
        # find nearest crater
        min_dist = float('inf')
        nearest_crater = None
        for crater in self.craters:
            cx, cy = crater['center']
            d = np.sqrt((px - cx)**2 + (py - cy)**2)
            if d < min_dist:
                min_dist = d
                nearest_crater = crater
        
        if num == 0:
            return "Landing Site"
        elif nearest_crater and min_dist < nearest_crater['radius'] + 30:
            return "Crater Rim"
        elif num == total - 1:
            return "Final Stop"
        elif num % 3 == 0:
            return "Regolith Sample"
        else:
            return "Surface Study"
    
    def plan(self):
        h, w = self.image.shape[:2]
        
        # find start point
        start = None
        for y in range(h // 4, h - h // 4):
            if self.is_safe((60, y)):
                start = (60, y)
                break
        
        # find end point
        end = None
        for y in range(h // 4, h - h // 4):
            if self.is_safe((w - 60, y)):
                end = (w - 60, y)
                break
        
        if not start or not end:
            print("Could not find safe points")
            return
        
        print(f"Planning path from {start} to {end}")
        self.path = self.find_path(start, end)
        
        if not self.path:
            print("No path found")
            return
        
        self.identify_stops()
    
    def save_map(self, filename="lunar_path_map_1.jpg"):
        output = self.image.copy()
        
        # draw shadows
        for shadow in self.shadows:
            cx, cy = shadow['crater']
            sx, sy = shadow['end']
            pts = np.array([
                [cx, cy],
                [sx + shadow['width'], sy + shadow['width']/2],
                [sx - shadow['width'], sy - shadow['width']/2]
            ], np.int32)
            cv2.fillPoly(output, [pts], (180, 180, 180))
        
        # draw craters
        for crater in self.craters:
            cx, cy = crater['center']
            cv2.circle(output, (cx, cy), crater['radius'], (0, 0, 255), 2)
        
        # draw path
        if len(self.path) > 1:
            for i in range(len(self.path) - 1):
                cv2.line(output, self.path[i], self.path[i+1], (0, 255, 255), 3)
        
        # draw stops
        for i, stop in enumerate(self.stops):
            pos = stop['pos']
            cv2.circle(output, pos, 12, (0, 255, 0), -1)
            cv2.circle(output, pos, 15, (255, 255, 255), 2)
            cv2.putText(output, f"S{i+1}", (pos[0] - 8, pos[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # mark start and end
        start = self.path[0]
        end = self.path[-1]
        cv2.circle(output, start, 15, (0, 255, 0), -1)
        cv2.putText(output, "LANDING", (start[0] + 20, start[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.circle(output, end, 15, (255, 0, 0), -1)
        
        # sun direction
        cv2.arrowedLine(output, (50, 50), 
                       (int(50 + 40 * self.sun_direction[0]), 
                        int(50 + 40 * self.sun_direction[1])), 
                       (0, 255, 255), 3)
        cv2.putText(output, "SUN", (55, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imwrite(filename, output)
        print(f"Map saved to {filename}")


def main():
    planner = LunarPathPlanner("luna_1.jpg", sun_azimuth=45)
    
    planner.detect_craters()
    planner.plan()
    planner.save_map()
    
    print("Done!")
    
    # show result
    img = cv2.imread("lunar_path_map_1.jpg")
    cv2.imshow("Lunar Path", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()