"""
ROB521_assignment1.py

This assignment will introduce you to the idea of motion planning for  
holonomic robots that can move in any direction and change direction of 
motion instantaneously.  Although unrealistic, it can work quite well for
complex large scale planning.  You will generate mazes to plan through 
and employ the PRM algorithm presented in lecture as well as any 
variations you can invent in the later sections.

There are three questions to complete (5 marks each):

    Question 1: implement the PRM algorithm to construct a graph
    connecting start to finish nodes.
    Question 2: find the shortest path over the graph by implementing the
    Dijkstra's or A* algorithm.
    Question 3: identify sampling, connection or collision checking 
    strategies that can reduce runtime for mazes.

Three helper functions are provided for you to use in your motion planning 
solution: min_dist_to_edges, distance_point_to_segment, and check_collision.
The first two are used to determine if a point is at least a minimum distance
from all walls in the maze, and the third checks if a line segment intersects
any walls in the maze.  You may modify these functions if you wish or use them
as is.
 
To complete the assignment, fill in the required sections of this script with 
your code, run it to generate the requested plots, then paste the plots into 
a short report that includes a few comments about what you've observed.  
Append your version of this script to the report.  Hand in the report as a 
PDF file.

requires: numpy, matplotlib

S L Waslander, revised January 2026 - Converted to Python
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from math import sqrt, inf
import heapq
import json

# set random seed for repeatability if desired
SEED = 1024
Navy = "#1E3765"
Sky = "#6FC7EA"
Taupe = "#8F8174"

np.random.seed(SEED)

# ==========================
# Maze Generation
# ==========================
#
# The maze function returns a map object with all of the edges in the maze.
# Each row of the map structure draws a single line of the maze.  The
# function returns the lines with coordinates [x1 y1 x2 y2].
# Bottom left corner of maze is [0.5 0.5], 
# Top right corner is [col+0.5 row+0.5]
# Each wall is [start_col start_row end_col end_row] and goes from bottom/left to top/right.

def maze(rows, cols, seed=SEED):
    """
    Generate a random maze using iterative depth-first search algorithm.
    Returns a list of line segments representing walls.
    """
    np.random.seed(seed)
    # Initialize grid with all walls
    walls = []
    
    # Create walls list, outer walls first
    for i in range(cols):
        walls.append([i + 0.5, 0.5, i + 1.5, 0.5])  
        for j in range(rows):
            if i == 0:
                walls.append([0.5, j + 0.5, 0.5, j + 1.5])  
            walls.append([i + 0.5, j + 1.5, i + 1.5, j + 1.5])  # horizontal walls
            walls.append([i + 1.5, j + 0.5, i + 1.5,  j + 1.5])  # vertical walls
  
    visited = np.zeros((cols, rows), dtype=bool)
    
    #Remove start and end walls
    walls.remove([0.5, 0.5, 0.5, 1.5])  # Remove entrance wall
    walls.remove([cols + 0.5, rows - 0.5, cols + 0.5, rows + 0.5])  # Remove exit wall

    # Iterative depth-first search using a stack (avoids recursion limit)
    stack = [(0, 0)]
    visited[0, 0] = True
    
    while stack:
        x, y = stack[-1]
        
        # Directions: right, down, left, up
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        np.random.shuffle(directions)
        
        found_unvisited = False
        for (dx, dy) in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and not visited[nx, ny]:
                # Remove wall between current and next cell
                if dx == 1:  # right
                    wall_to_remove = [x + 1.5, y + 0.5, x + 1.5, y + 1.5]
                    if wall_to_remove in walls:
                        walls.remove(wall_to_remove)
                elif dx == -1:  # left
                    wall_to_remove = [x + 0.5, y + 0.5, x + 0.5, y + 1.5]
                    if wall_to_remove in walls:
                        walls.remove(wall_to_remove)
                elif dy == 1:  # up
                    wall_to_remove = [x + 0.5, y + 1.5, x + 1.5, y + 1.5]
                    if wall_to_remove in walls:
                        walls.remove(wall_to_remove)
                elif dy == -1:  # down
                    wall_to_remove = [x + 0.5, y + 0.5, x + 1.5, y + 0.5]
                    if wall_to_remove in walls:
                        walls.remove(wall_to_remove)
                
                visited[nx, ny] = True
                stack.append((nx, ny))
                found_unvisited = True
                break
        
        if not found_unvisited:
            stack.pop()
    
    return walls

def show_maze(walls, rows, cols, ax):
    """Draw the maze on the given matplotlib axis."""
    walls = np.array(walls)
    for wall in walls:
        ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=1)
    ax.set_xlim(0, cols + 1)
    ax.set_ylim(0, rows + 1)
    ax.set_aspect('equal')
    ax.grid(False)

def min_dist_to_edges(point, walls, min_dist=0.1):
    """
    Check if a point is at least min_dist away from all walls.
    Returns True if the point is valid (far enough from all walls).
    """
    px, py = point
    for wall in walls:
        x1, y1, x2, y2 = wall
        # Distance from point to line segment
        dist = distance_point_to_segment(px, py, x1, y1, x2, y2)
        if dist < min_dist:
            return False
    return True

def distance_point_to_segment(px, py, x1, y1, x2, y2):
    """Calculate minimum distance from point (px, py) to line segment."""
    # Vector from start to end of segment
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        return sqrt((px - x1)**2 + (py - y1)**2)
    
    # Parameter t for closest point on line
    t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / length_sq))
    
    closest_x = x1 + t*dx
    closest_y = y1 + t*dy
    
    return sqrt((px - closest_x)**2 + (py - closest_y)**2)

def check_collision(x1, y1, x2, y2, walls, min_dist=0.1):
    """
    Check if the line segment from (x1,y1) to (x2,y2) collides with any walls.
    Returns True if collision-free.
    """
    # Check multiple points along the path
    steps = int(sqrt((x2-x1)**2 + (y2-y1)**2) * 10) + 2
    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0
        x = x1 + t*(x2 - x1)
        y = y1 + t*(y2 - y1)
        if not min_dist_to_edges([x, y], walls, min_dist):
            return False
    return True

# ======================================================
# Question 1: construct a PRM connecting start and finish
# ======================================================
#
# Using 500 samples, construct a PRM graph whose milestones stay at least 
# 0.1 units away from all walls, using the min_dist_to_edges function provided for 
# collision detection.  Use a nearest neighbour connection strategy and the 
# check_collision function provided for collision checking, and find an 
# appropriate number of connections to ensure a connection from  start to 
# finish with high probability.
row = 5
col = 7
walls = maze(row, col)
start = np.array([0.5, 1.0])
finish = np.array([col + 0.5, row])
# variables to store PRM components
nS = 500  # number of samples to try for milestone creation
milestones = [start, finish]  # each row is a point [x y] in feasible space
edges = []  # each row is should be an edge of the form [x1 y1 x2 y2]
print("Time to create PRM graph")
t0 = time()
# ------insert your PRM generation code here-------
# --- Sampling strategies ---
def sample_uniform(nS, walls):
    nodes = []
    for _ in range(nS):
        x = np.random.uniform(0.5, col + 0.5)
        y = np.random.uniform(0.5, row + 0.5)
        if min_dist_to_edges([x, y], walls):
            nodes.append(np.array([x, y]))
    return nodes

def sample_gaussian_lavalle(nS, walls, sigma=0.7):
    nodes = []
    for _ in range(nS):
        qx = np.random.uniform(0.5, col + 0.5)
        qy = np.random.uniform(0.5, row + 0.5)
        qx_p = qx + np.random.normal(0, sigma)
        qy_p = qy + np.random.normal(0, sigma)
        q_free = min_dist_to_edges([qx, qy], walls)
        if 0.5 <= qx_p <= col + 0.5 and 0.5 <= qy_p <= row + 0.5:
            qp_free = min_dist_to_edges([qx_p, qy_p], walls)
        else:
            qp_free = False
        if q_free and not qp_free:
            nodes.append(np.array([qx, qy]))
        elif qp_free and not q_free:
            nodes.append(np.array([qx_p, qy_p]))
    return nodes

def sample_gaussian_latombe(nS, walls, sigma=0.7):
    nodes = []
    for _ in range(nS):
        qx = np.random.uniform(0.5, col + 0.5)
        qy = np.random.uniform(0.5, row + 0.5)
        d = abs(np.random.normal(0, sigma))
        theta = np.random.uniform(0, 2 * np.pi)
        qx_p = qx + d * np.cos(theta)
        qy_p = qy + d * np.sin(theta)
        q_free = min_dist_to_edges([qx, qy], walls)
        if 0.5 <= qx_p <= col + 0.5 and 0.5 <= qy_p <= row + 0.5:
            qp_free = min_dist_to_edges([qx_p, qy_p], walls)
        else:
            qp_free = False
        if q_free and not qp_free:
            nodes.append(np.array([qx, qy]))
        elif qp_free and not q_free:
            nodes.append(np.array([qx_p, qy_p]))
    return nodes

def sample_bridge(nS, walls, sigma=0.7):
    nodes = []
    for _ in range(nS):
        qx = np.random.uniform(0.5, col + 0.5)
        qy = np.random.uniform(0.5, row + 0.5)
        if min_dist_to_edges([qx, qy], walls):
            continue
        qx_p = qx + np.random.normal(0, sigma)
        qy_p = qy + np.random.normal(0, sigma)
        if not (0.5 <= qx_p <= col + 0.5 and 0.5 <= qy_p <= row + 0.5):
            continue
        if min_dist_to_edges([qx_p, qy_p], walls):
            continue
        mx, my = (qx + qx_p) / 2, (qy + qy_p) / 2
        if min_dist_to_edges([mx, my], walls):
            nodes.append(np.array([mx, my]))
    return nodes

sample_methods = [
    ("Uniform",               sample_uniform,          dict()),
    ("Gauss-Lavalle σ=0.5",   sample_gaussian_lavalle, dict(sigma=0.5)),
    ("Gauss-Latombe σ=0.5",   sample_gaussian_latombe, dict(sigma=0.5)),
    ("Bridge σ=0.5",          sample_bridge,           dict(sigma=0.5)),
]

# --- nearest neighbour connection ---
def find_k_nearest(milestones, walls, k=20):

    nodes = np.asarray(milestones, float)
    edge_set = set()
    n = len(nodes)
    k = min(k, n - 1)

    for i in range(n):
        list_distances = np.linalg.norm(nodes - nodes[i], axis=1)
        sorted_distances = np.argsort(list_distances)[1:k+1]
        xi, yi = nodes[i]

        for j in sorted_distances:
            xj, yj = nodes[j]
            a, b = (i, j) if i < j else (j, i)

            if (a, b) in edge_set: continue

            if check_collision(xi, yi, xj, yj, walls, min_dist=0.1):
                edge_set.add((a, b))

    return list(edge_set)

# --- construct PRM ---
def construct_prm(start, finish, walls, nS, sample_fn, k=13, seed=SEED):
    np.random.seed(seed)

    t0 = time()
    sampled = sample_fn(nS, walls)
    milestones = [start, finish] + sampled
    edge_pairs = find_k_nearest(milestones, walls, k=k)
    dt = time() - t0

    accept = (len(sampled) / nS * 100.0) if nS else 0.0
    stats = {
        "dt": dt,
        "accept": accept,
        "n_nodes": len(milestones),
        "n_edges": len(edge_pairs),
    }
    return milestones, edge_pairs, stats

# --- run ---
def run_sampling(sample_specs, *, n_trials=100, base_seed=SEED, out_json="q1_sampling_mc_stats.json"):
    data = {
        "meta": {
            "n_trials": n_trials,
            "base_seed": base_seed,
            "row": row,
            "col": col,
            "nS": nS,
            "k": 13,
        },
        "methods": {}
    }

    for name, fn, kwargs in sample_specs:
        trials = []

        for trial in range(n_trials):
            seed = base_seed + trial  

            sample_fn = (lambda nS_, walls_, fn=fn, kwargs=kwargs: fn(nS_, walls_, **kwargs))

            milestones, edge_pairs, stats = construct_prm(
                start=start,
                finish=finish,
                walls=walls,
                nS=nS,
                sample_fn=sample_fn,
                k=13,
                seed=seed,
            )

            trials.append({
                "trial": trial,
                "seed": seed,
                "dt": float(stats["dt"]),
                "n_nodes": int(stats["n_nodes"]),
                "n_edges": int(stats["n_edges"]),
                "accept": float(stats["accept"]), 
            })

        # averages over 100 runs
        dt_mean = float(np.mean([t["dt"] for t in trials]))
        n_nodes_mean = float(np.mean([t["n_nodes"] for t in trials]))
        n_edges_mean = float(np.mean([t["n_edges"] for t in trials]))
        accept_mean = float(np.mean([t["accept"] for t in trials]))

        data["methods"][name] = {
            "kwargs": kwargs,
            "summary": {
                "dt_mean": dt_mean,
                "n_nodes_mean": n_nodes_mean,
                "n_edges_mean": n_edges_mean,
                "accept_mean": accept_mean,
            },
            "trials": trials,
        }

        print(f"{name}: dt={dt_mean:.4f}s, nodes={n_nodes_mean:.1f}, edges={n_edges_mean:.1f}, accept={accept_mean:.1f}%")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

run_sampling(sample_methods, n_trials=100, base_seed=SEED, out_json="q1_sampling_method_stats.json")

# ------end of your PRM generation code -------
def plot_prm(ax, milestones, edge_pairs, walls, row, col, start, finish, title="", subtitle="", node_marksize=2.5):
    m = np.asarray(milestones, float)

    # edges (as segments)
    if edge_pairs and len(edge_pairs) > 0:
        e = np.array([[m[i, 0], m[i, 1], m[j, 0], m[j, 1]] for i, j in edge_pairs], float)
        ax.plot([e[:, 0], e[:, 2]], [e[:, 1], e[:, 3]], color=Sky, alpha=0.35, linewidth=0.6)

    # nodes
    ax.plot(m[:, 0], m[:, 1], ".", markersize=node_marksize, color=Sky)

    # start/finish
    ax.plot(start[0], start[1], "o", markersize=7, color = Navy)
    ax.plot(finish[0], finish[1], "x", markersize=7, color = Taupe)

    show_maze(walls, row, col, ax)

    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    else:
        ax.set_title(title, fontsize=11)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

for ax, (name, fn, kwargs) in zip(axes.ravel(), sample_methods):
    nodes, edge_pairs, stats = construct_prm(
        start=start,
        finish=finish,
        walls=walls,
        nS=nS,
        sample_fn=fn,
        k=13,
        seed=SEED,
    )
    if name == 'Uniform':
        edge_indices_q1 = edge_pairs
        milestones_q1 = nodes

    subtitle = f"{stats['n_nodes']} nodes | {stats['n_edges']} edges"
    plot_prm(
        ax=ax,
        milestones=nodes,
        edge_pairs=edge_pairs,
        walls=walls,
        row=row,
        col=col,
        start=start,
        finish=finish,
        title=name,
        subtitle=subtitle,
        node_marksize=2.2,
    )
plt.savefig("q1_sampling_comparison.png", dpi=200)
plt.show()
plt.close()

# =================================================================
# Question 2: Find the shortest path over the PRM graph
# =================================================================
#
# Using an optimal graph search method (Dijkstra's or A*), find the 
# shortest path across the graph generated.  Please code your own 
# implementation instead of using any built in functions.
print('Time to find shortest path')
t0 = time()
spath = []  # list of milestone indices that form the shortest path
# ------insert your shortest path finding code here-------

def compute_euclidean_distance(milestone1, milestone2):
    """calculate straight-line distance to goal"""
    distance = sqrt(
        (milestone1[0] - milestone2[0])**2 
        + (milestone1[1] - milestone2[1])**2
        )
    return distance

def build_adjacency_list(milestones, edge_pairs):
    nodes = np.asarray(milestones, float)
    adj_list = [[] for _ in range(len(milestones))]
    sqrt_ = sqrt
    for i, j in edge_pairs:
        dx = nodes[i, 0] - nodes[j, 0]
        dy = nodes[i, 1] - nodes[j, 1]
        cost = sqrt_(dx*dx + dy*dy)
        adj_list[i].append((j, cost))
        adj_list[j].append((i, cost))
    return adj_list

def astar(start_index, goal_index, milestones, adj_list):
    """a straight forward implementation for understanding"""
    # define heuristic function
    def heuristic(node_index):
        return compute_euclidean_distance(
            milestones[node_index], 
            milestones[goal_index]
        )
    
    # node class encapsulates all info
    class Node:
        def __init__(self, index):
            self.index = index
            self.parent = None
            self.g = inf 
            self.h = heuristic(index)
            self.f = self.g + self.h

    # dictionary to map node index to node
    nodes = {i : Node(i) for i in range(len(milestones))}
    nodes[start_index].g = 0
    nodes[start_index].f = heuristic(start_index)

    # priority queue: list of index of nodes
    Q = [start_index]
    # nodes already expanded 
    dead = set()

    while Q:
        # sort by f-score and pop the minimum
        Q.sort(key=lambda index: nodes[index].f)
        current_index = Q.pop(0)

        # skip if the node has been expanded
        if current_index in dead: continue
        dead.add(current_index)

        # if reach goal node, reconstruct the path
        if current_index == goal_index:
            path = []
            node = nodes[current_index]
            while node.parent:
                path.append(node.index)
                node = node.parent
            path.append(start_index)
            return path[::-1]
        
        # explore neighbours
        for neighbour_index, edge_cost in adj_list[current_index]:
            if neighbour_index in dead: continue

            # calculate tentative cost to come if entering the neighbour
            tentative_g = nodes[current_index].g + edge_cost
            # if this tentative g is lower than the neighbour's current g
            # set current node as the parent of the neighbour
            if tentative_g < nodes[neighbour_index].g:
                nodes[neighbour_index].g = tentative_g
                nodes[neighbour_index].f = tentative_g + nodes[neighbour_index].h
                nodes[neighbour_index].parent = nodes[current_index]

                # add to the queue if the neighbour is not in
                if neighbour_index not in Q:
                    Q.append(neighbour_index)

    return []

adj_list = build_adjacency_list(milestones_q1, edge_indices_q1)
spath = astar(start_index=0, goal_index=1, milestones=milestones_q1, adj_list=adj_list)

# ------end of your shortest path finding code -------
elapsed = time() - t0
print(f"Time elapsed: {elapsed:.4f} seconds")

def edge_pairs_to_segments(milestones, edge_pairs):
    m = np.asarray(milestones, float)
    ep = np.asarray(edge_pairs, int)
    edges_coord = np.empty((len(ep), 4), dtype=float)
    edges_coord[:, 0:2] = m[ep[:, 0], :]
    edges_coord[:, 2:4] = m[ep[:, 1], :]
    return edges_coord

edges_coord = edge_pairs_to_segments(milestones_q1, edge_indices_q1) 
# plot the shortest path
fig, ax = plt.subplots(figsize=(10, 8))
m = np.asarray(milestones_q1, float)
ax.plot(m[:, 0], m[:, 1], ".", markersize=4, color=Sky)
if edges_coord is not None and len(edges_coord) > 0:
    e = np.asarray(edges_coord, float)
    ax.plot([e[:, 0], e[:, 2]], [e[:, 1], e[:, 3]],
            c=Sky, alpha=0.5, linewidth=0.5)
ax.plot(start[0], start[1], "o", markersize=8, color=Navy)
ax.plot(finish[0], finish[1], "x", markersize=8, color=Taupe)
if spath is not None and len(spath) > 1:
    sp = np.asarray(spath, dtype=int)
    path_points = m[sp]
    ax.plot(path_points[:, 0], path_points[:, 1], "o-",
            linewidth=3, markersize=6, color=Navy)
show_maze(walls, row, col, ax)
ax.set_title(f"Q2 - {row} X {col} Maze Shortest Path")
plt.tight_layout()
plt.savefig("assignment1_q2.png", dpi=150)
plt.show()
plt.close()

# ================================================================
# Question 3: find a faster way
# ================================================================
#
# Modify your milestone generation, edge connection, collision detection 
# and/or shortest path methods to reduce runtime.  What is the largest maze 
# for which you can find a shortest path from start to goal in under 20 
# seconds on your computer? (Anything larger than 40x40 will suffice for 
# full marks)

row = 41
col = row
walls = maze(row, col)
start = np.array([0.5, 1.0])
finish = np.array([col + 0.5, row])
milestones = [list(start), list(finish)]
edges = []
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(start[0], start[1], 'o', markersize=7, color=Navy)
ax.plot(finish[0], finish[1], 'x', markersize=7, color=Taupe)
show_maze(walls, row, col, ax)
plt.draw()
plt.pause(0.1)
print(f"Attempting large {row} X {col} maze...")
t0 = time()
# ------insert your optimized algorithm here------
np.random.seed(SEED)  

def compute_path_length_from_indices(milestones, path_indices):
    if path_indices is None or len(path_indices) < 2:
        return 0.0
    path_points = np.asarray([milestones[node_index] for node_index in path_indices], dtype=float)
    deltas = np.diff(path_points, axis=0)
    segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    return float(segment_lengths.sum())

def build_wall_grid(walls, padding=0.1):
    """
    spatial hash using grid index for wall segments
    so later queries only check nearby walls
    """
    grid = {}
    for i, (x1, y1, x2, y2) in enumerate(walls):
        for grid_x in range(int(min(x1,x2)-padding), int(max(x1,x2)+padding)+1):
            for grid_y in range(int(min(y1,y2)-padding), int(max(y1,y2)+padding)+1):
                grid.setdefault((grid_x,grid_y), []).append(i)
    return grid

def build_node_grid(nodes, cell_size):
    """
    spatial hash for sampled milestones
    for retrieving nearby candidate neighbours
    """
    grid = {}
    for i, (x, y) in enumerate(nodes):
        grid_x, grid_y = int(x / cell_size), int(y / cell_size)
        grid.setdefault((grid_x, grid_y), []).append(i)
    return grid

def min_dist_to_edges_q3(node, walls, wall_grid, min_dist=0.1):
    """
    check if the node is at least min_dist away from nearby wall segments
    (candidates from the 3x3 grid-cell neighbourhood around grid_x, grid_y)
    """
    x, y = node
    grid_x, grid_y = int(x), int(y)
    checked = set()
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            for index in wall_grid.get((grid_x+dx, grid_y+dy), []):
                if index in checked: continue
                checked.add(index)
                if distance_point_to_segment(x, y, *walls[index]) < min_dist:
                    return False
    return True

def check_collision_q3(x1, y1, x2, y2, walls, wall_grid, min_dist=0.1, step=5):
    """
    check collision on the line segment (x1,y1)->(x2,y2) 
    by sampling intermediate points
    """
    dist = sqrt((x2-x1)**2 + (y2-y1)**2)
    # number of substeps along the segment
    substeps = max(3, int(dist / step))
    for i in range(substeps + 1):
        t = i / substeps        # interpolation ratio 

        # point on segment at fraction t (linear interpolation)
        x = x1 + t*(x2-x1)
        y = y1 + t*(y2-y1)
        
        # collision test at this sampled point (only checks walls in neighbouring grid cells)
        if not min_dist_to_edges_q3((x, y), walls, wall_grid, min_dist):
            return False

    return True

def find_k_nearest_q3(milestones, walls, wall_grid, k=15, radius_max=7.0, min_dist=0.2, step=0.35):
    """
    build PRM edges by connecting each milestone to up to k milestones within radius_max
    """
    # float array for vectorisation
    nodes = np.asarray(milestones, float)
    # spatial hash for candidate neighbours
    grid = build_node_grid(nodes, radius_max)
    # compare squared distances to avoid sqrt inside loops
    radius_squared = radius_max * radius_max

    edges = set()

    # iterate over all milestones
    for i, (x, y) in enumerate(nodes):
        # convert world coordinate to grid 
        grid_x, grid_y = int(x / radius_max), int(y / radius_max)
        
        # only check nearby grid cells
        candidates = []
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                candidates.extend(grid.get((grid_x+dx, grid_y+dy), []))
        # remove self index
        candidates = [j for j in candidates if j != i]
        if not candidates: continue
        
        # keep only candidates within radius_max
        dist_squared = np.sum((nodes[candidates] - nodes[i])**2, axis=1)
        mask = dist_squared < radius_squared
        candidates = np.array(candidates)[mask]
        dist_squared = dist_squared[mask]
        
        # if too many candidates, keep the k smallest distances
        # argpartition gives "k smallest (unordered)" in O(M) 
        # rather than full sort O(M log M)
        if len(candidates) > k:
            idx = np.argpartition(dist_squared, k-1)[:k]
            candidates, dist_squared = candidates[idx], dist_squared[idx]
        
        # sort the reduced set
        for j in candidates[np.argsort(dist_squared)]:
            # canonicalise undirected edge key (a<b) to avoid duplicates (i,j) vs (j,i)
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            # if not in edges and collision-free
            # check collision and add to edges 
            if (a,b) not in edges and check_collision_q3(*nodes[a], *nodes[b], walls, wall_grid, min_dist, step):
                edges.add((a, b))
    return list(edges)

# --- bidirectional A* with heap ---
def astar_bidirectional(start_index, goal_index, milestones, adj_list):
    start_x, start_y = milestones[start_index]
    goal_x, goal_y = milestones[goal_index]

    def heuristic_to_goal(node_index):
        x, y = milestones[node_index]
        dx = x - goal_x
        dy = y - goal_y
        return sqrt(dx * dx + dy * dy)

    def heuristic_to_start(node_index):
        x, y = milestones[node_index]
        dx = x - start_x
        dy = y - start_y
        return sqrt(dx * dx + dy * dy)

    # forward search states
    g_forward = {start_index: 0.0}
    parent_forward = {}
    dead_forward = set()
    Q_forward = [(heuristic_to_goal(start_index), start_index)]  

    # backward search states
    g_backward = {goal_index: 0.0}  
    parent_backward = {}
    dead_backward = set()
    Q_backward = [(heuristic_to_start(goal_index), goal_index)]  

    best_path_cost = inf
    meet_index = None
    expanded = 0

    def expand_one(open_heap, g_this, parent_this, dead_this, g_other, heuristic_this):
        nonlocal best_path_cost, meet_index, expanded

        if not open_heap:
            return

        _, current_index = heapq.heappop(open_heap)
        if current_index in dead_this:
            return

        dead_this.add(current_index)
        expanded += 1

        # meeting check
        if current_index in g_other:
            candidate_cost = g_this[current_index] + g_other[current_index]
            if candidate_cost < best_path_cost:
                best_path_cost = candidate_cost
                meet_index = current_index

        # relax neighbours
        for neighbour_index, edge_cost in adj_list[current_index]:
            if neighbour_index in dead_this:
                continue

            tentative_g = g_this[current_index] + edge_cost
            if tentative_g < g_this.get(neighbour_index, inf):
                g_this[neighbour_index] = tentative_g
                parent_this[neighbour_index] = current_index
                heapq.heappush(open_heap, (tentative_g + heuristic_this(neighbour_index), neighbour_index))
    
    # main loop
    while Q_forward or Q_backward:
        best_forward_key = Q_forward[0][0] if Q_forward else inf
        best_backward_key = Q_backward[0][0] if Q_backward else inf

        # termination rule (same style as your previous bidir code)
        if best_forward_key + best_backward_key >= best_path_cost:
            break

        expand_one(Q_forward, g_forward, parent_forward, dead_forward,
                   g_backward, heuristic_to_goal)
        expand_one(Q_backward, g_backward, parent_backward, dead_backward,
                   g_forward, heuristic_to_start)

    if meet_index is None:
        return [], {"expanded": expanded, "path_cost": inf, "meet_index": None}

    # reconstruct: start -> meet
    path_left = []
    node_index = meet_index
    while node_index in parent_forward:
        path_left.append(node_index)
        node_index = parent_forward[node_index]
    path_left.append(start_index)
    path_left.reverse()

    # reconstruct: meet -> goal 
    path_right = []
    node_index = meet_index
    while node_index in parent_backward:
        node_index = parent_backward[node_index]
        path_right.append(node_index)

    path = path_left + path_right
    return path, {"expanded": expanded, "path_cost": best_path_cost, "meet_index": meet_index}

n_samples = int(6 * row * col)  # Dense sampling for robustness
wall_grid = build_wall_grid(walls)
milestones = [start, finish]

for _ in range(n_samples):
    milestone = (np.random.uniform(0.5, col+0.5), np.random.uniform(0.5, row+0.5))
    if min_dist_to_edges_q3(milestone, walls, wall_grid):
        milestones.append(list(milestone))

edge_list = find_k_nearest_q3(milestones, walls, wall_grid, k=47, radius_max=7, min_dist=0.1, step=0.15)
adj_list = build_adjacency_list(milestones, edge_list)
spath, stats = astar_bidirectional(0, 1, milestones, adj_list)
edges = [[*milestones[i], *milestones[j]] for i, j in edge_list]

# ------end of your optimized algorithm-------
dt = time() - t0
path_length_q3 = compute_path_length_from_indices(milestones, spath)
print(f"Q3 path length: {path_length_q3:.3f}")
ax.plot(np.array(milestones)[:, 0], np.array(milestones)[:, 1], '.', markersize=1, color=Sky, alpha=0.1)
if len(edges) > 0:
    edges = np.array(edges)
    ax.plot([edges[:, 0], edges[:, 2]], [edges[:, 1], edges[:, 3]], c=Sky, alpha=0.1, linewidth=0.1)

if len(spath) > 1:
    path_points = np.array([milestones[i] for i in spath], float)

    ax.plot(path_points[:, 0], path_points[:, 1], 'o-', linewidth=1.7, markersize=2, color=Navy)
ax.set_title(f'Q3 - {row} X {col} Maze solved in {dt:.4f} seconds')
plt.tight_layout()
plt.savefig('assignment1_q3.png', dpi=250)
plt.show()  
plt.close()
print(f"Q3 completed in {dt:.4f} seconds")

# ================================================================
# Question 3e: find a faster way (grid)
# ================================================================
row = 41
col = row
walls = maze(row, col)
start = np.array([0.5, 1.0])
finish = np.array([col + 0.5, row])
milestones = [list(start), list(finish)]
edges = []
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(start[0], start[1], 'o', markersize=7, color=Navy)
ax.plot(finish[0], finish[1], 'x', markersize=7, color=Taupe)
show_maze(walls, row, col, ax)
plt.draw()
plt.pause(0.1)
print(f"Attempting large {row} X {col} maze...")
t0 = time()

# ------insert your optimized algorithm here------
# --- convert maze to grid ---
def build_adjacency_list_grid(walls, row, col):
    walls_np = np.asarray(walls, dtype=float)

    # blocked walls between adjacent cells
    # right_wall_blocked[r,c] blocks (c,r) <-> (c+1,r), c in [0..col-2]
    # top_wall_blocked[r,c]   blocks (c,r) <-> (c,r+1), r in [0..row-2]
    right_wall_blocked = np.zeros((row, col - 1), dtype=bool)
    top_wall_blocked   = np.zeros((row - 1, col), dtype=bool)

    # vertical walls: x1==x2 and length 1 in y
    is_vertical = (walls_np[:, 0] == walls_np[:, 2]) & ((walls_np[:, 3] - walls_np[:, 1]) == 1.0)
    vertical_walls = walls_np[is_vertical]
    if vertical_walls.size > 0:
        cell_col = (vertical_walls[:, 0] - 1.5).astype(int)  # x = cell_col + 1.5
        cell_row = (vertical_walls[:, 1] - 0.5).astype(int)  # y1 = cell_row + 0.5
        in_range = (
            (0 <= cell_row) & (cell_row < row) &
            (0 <= cell_col) & (cell_col < col - 1)
        )
        right_wall_blocked[cell_row[in_range], cell_col[in_range]] = True

    # horizontal walls: y1==y2 and length 1 in x
    is_horizontal = (walls_np[:, 1] == walls_np[:, 3]) & ((walls_np[:, 2] - walls_np[:, 0]) == 1.0)
    horizontal_walls = walls_np[is_horizontal]
    if horizontal_walls.size > 0:
        cell_col = (horizontal_walls[:, 0] - 0.5).astype(int)  # x1 = cell_col + 0.5
        cell_row = (horizontal_walls[:, 1] - 1.5).astype(int)  # y  = cell_row + 1.5
        in_range = (
            (0 <= cell_row) & (cell_row < row - 1) &
            (0 <= cell_col) & (cell_col < col)
        )
        top_wall_blocked[cell_row[in_range], cell_col[in_range]] = True

    # open edges (vectorized)
    right_open = ~right_wall_blocked
    right_rows, right_cols = np.where(right_open)
    right_a = right_rows * col + right_cols
    right_b = right_a + 1

    top_open = ~top_wall_blocked
    top_rows, top_cols = np.where(top_open)
    top_a = top_rows * col + top_cols
    top_b = top_a + col

    edge_pairs = np.vstack((
        np.column_stack((right_a, right_b)),
        np.column_stack((top_a, top_b)),
    ))
    edge_pairs = np.sort(edge_pairs, axis=1)  # ensure (a<b)

    # adjacency list
    node_count = row * col
    adj_list = [[] for _ in range(node_count)]
    grid_edge_cost = 1.0  # center-to-center spacing

    for node_a, node_b in edge_pairs:
        node_a = int(node_a); node_b = int(node_b)
        adj_list[node_a].append((node_b, grid_edge_cost))
        adj_list[node_b].append((node_a, grid_edge_cost))

    return adj_list, edge_pairs

def attach_start_goal_to_grid(adj_list, row, col, start, finish):
    _sqrt = sqrt

    def grid_cell_index(cell_col, cell_row, col_count):
        return cell_row * col_count + cell_col
    
    def euclidean_cost(p, q):
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        return _sqrt(dx * dx + dy * dy)

    # milestones_world for cell centers
    cell_columns = np.arange(col, dtype=float)
    cell_rows = np.arange(row, dtype=float)

    grid_cell_columns, grid_cell_rows = np.meshgrid(cell_columns, cell_rows)  # shapes (row, col)

    cell_centers_world = np.column_stack((
        grid_cell_columns.ravel(order="C") + 1.0,
        grid_cell_rows.ravel(order="C") + 1.0,
    ))  # shape (row*col, 2)

    milestones_world = cell_centers_world.tolist()  # list[[x,y], ...] in row-major order

    start_index = row * col
    goal_index = start_index + 1

    milestones_world.append((float(start[0]), float(start[1])))
    milestones_world.append((float(finish[0]), float(finish[1])))

    # grow adjacency
    adj_list.append([])
    adj_list.append([])

    start_cell_index = grid_cell_index(0, 0, col)
    goal_cell_index  = grid_cell_index(col - 1, row - 1, col)

    start_cost = euclidean_cost(milestones_world[start_index], milestones_world[start_cell_index])
    goal_cost  = euclidean_cost(milestones_world[goal_index],  milestones_world[goal_cell_index])

    # undirected attachments
    adj_list[start_index].append((start_cell_index, start_cost))
    adj_list[start_cell_index].append((start_index, start_cost))

    adj_list[goal_index].append((goal_cell_index, goal_cost))
    adj_list[goal_cell_index].append((goal_index, goal_cost))

    return milestones_world, start_index, goal_index

# --- run ---
adj_list_grid, edge_pairs_grid = build_adjacency_list_grid(walls, row, col)
milestones_world, start_index, goal_index = attach_start_goal_to_grid(adj_list_grid, row, col, start, finish)

spath_indices, stats = astar_bidirectional(start_index, goal_index, milestones_world, adj_list_grid)
path = [milestones_world[node_index] for node_index in spath_indices]  
milestones = [list(point_xy) for point_xy in path]
spath = list(range(len(milestones)))
edges = [[*milestones[i], *milestones[i + 1]] for i in range(len(milestones) - 1)] if len(milestones) > 1 else []

# ------end of your optimized algorithm-------
dt = time() - t0
path_length_q3e = compute_path_length_from_indices(milestones_world, spath_indices)
print(f"Q3e path length: {path_length_q3e:.3f}")
ax.plot(np.array(milestones)[:, 0], np.array(milestones)[:, 1], '.', markersize=1, color=Sky)
if len(spath) > 1:
    path_points = np.array([milestones[i] for i in spath], float)
    ax.plot(path_points[:, 0], path_points[:, 1], 'o-', linewidth=1.7, markersize=2, color=Navy)
ax.set_title(f'Q3 - {row} X {col} Maze solved in {dt:.4f} seconds')
plt.tight_layout()
plt.savefig(f'assignment1_q3_grid {row}X{col} .png', dpi=250)
plt.close()

print(f"\nQ3: {row}x{col} maze solved in {dt:.4f}s")
