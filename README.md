# LUNAR SOUTH POLE NAVIGATION ROUTE ANALYSIS

## Mission Overview
- **Landing Region:** South Polar Region
- **Total Traverse Distance:** >100 meters
- **Scientific Stops:** 10 equally-spaced locations along path
- **Sun Azimuth:** 45° (Northeast direction)

---

## 1. ROUTE SELECTION

### A* Pathfinding Algorithm
The route uses A* algorithm to find optimal path while avoiding hazards.

**Cost Function:**
- Base cost = distance moved
- Heuristic = straight-line distance to goal
- Shadow penalty = +500 points (avoids power loss)
- Crater penalty = +30 points if within 50 pixels

**Route Features:**
- 20-pixel safety margin from crater edges
- Prioritizes sunlit terrain for continuous solar power
- East-west direction for stable communication
- Smooth path without sharp turns

---

## 2. SAFETY AND FEASIBILITY

### Crater Detection
- Roboflow CV model detects craters from Chandrayaan-2 OHRC dataset
- Stores center coordinates and radius for each crater
- 20-pixel buffer zone prevents edge collisions

### Shadow Avoidance
- Shadows computed from sun azimuth (45°)
- Shadow length = 1.5× crater radius
- Shadow width = 0.6× crater radius
- Vector projection determines if point is in shadow
- 500-point penalty ensures algorithm routes around shadows

### Power Management
- >90% of path in direct sunlight
- Continuous solar charging during traverse
- Stops positioned in sunlit areas for recharging

---

## 3. SCIENTIFIC STOPS

### Stop Distribution
Stops are **equally spaced** along the path at: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 95% of total path length. This provides uniform coverage of the traversed region.


## 4. SOLAR POWER CONSIDERATIONS

### Shadow Projection Method
1. Crater center as shadow origin
2. Shadow extends opposite to sun direction
3. Length = 1.5× radius, Width = 0.6× radius
4. Triangular shadow zone created

### Shadow Detection
- Projects point onto shadow vector
- Checks if within shadow length
- Calculates perpendicular distance from centerline
- Rejects point if distance < shadow width

---

## 5. IMPLEMENTATION

### Algorithm Flow
1. Load image and set sun azimuth
2. Detect craters via Roboflow API
3. Calculate shadow zones for all craters
4. Find safe start/end points
5. Run A* pathfinding with safety checks
6. Identify 10 equally-spaced scientific stops
7. Generate annotated map

### Safety Check Function
```
is_safe(point):
    - Check image boundaries
    - Check distance to all craters (must be > radius + 20px)
    - Check if point in any shadow zone
    - Return True only if all checks pass
```

---

## 6. ASSUMPTIONS & LIMITATIONS

**Assumptions:**
- Flat terrain (no slope modeling)
- Static sun position during traverse
- All craters detected by AI model
- Simplified shadow geometry

**Limitations:**
- No real-time hazard detection
- Cannot replan during traverse
- No slope or tilt considerations
- Communication range not factored

---
