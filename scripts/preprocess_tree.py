
import os
import sys
import argparse
import numpy as np
import torch
import networkx as nx
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.node import Node


def load_vtp(vtp_path):
    import vtk
    
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()
    
    points = polydata.GetPoints()
    n_pts = points.GetNumberOfPoints()
    coords = np.array([points.GetPoint(i) for i in range(n_pts)], dtype=np.float32)
    
    radius_arr = polydata.GetPointData().GetArray("Radius")
    if radius_arr:
        radii = np.array([radius_arr.GetValue(i) for i in range(n_pts)], dtype=np.float32)
    else:
        radii = np.ones(n_pts, dtype=np.float32) * 0.5
    
    label_arr = polydata.GetPointData().GetArray("label")
    if label_arr:
        labels = np.array([int(label_arr.GetValue(i)) for i in range(n_pts)], dtype=np.int32)
    else:
        labels = np.zeros(n_pts, dtype=np.int32)
    
    lines = polydata.GetLines()
    segments = []
    lines.InitTraversal()
    idList = vtk.vtkIdList()
    
    while lines.GetNextCell(idList):
        n_seg_pts = idList.GetNumberOfIds()
        if n_seg_pts < 2:
            continue
        
        indices = [idList.GetId(j) for j in range(n_seg_pts)]
        segments.append({
            'indices': indices,
            'points': coords[indices].copy(),
            'radius': radii[indices].copy(),
            'label': labels[indices[0]]
        })
    
    return coords, radii, labels, segments


def compute_segment_properties(points):
    points = np.array(points[:, :3] if points.ndim == 2 and points.shape[1] > 3 else points)
    
    start = points[0]
    end = points[-1]
    
    straight_distance = float(np.linalg.norm(end - start))
    
    diffs = np.diff(points, axis=0)
    curve_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    
    curvature = 1.0 - (straight_distance / curve_length) if curve_length > 1e-6 else 0.0
    
    axis = end - start
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-6:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = axis / axis_norm
        max_distance = 0
        max_point = points[len(points)//2]
        
        for point in points:
            v = point - start
            proj = np.dot(v, axis) * axis
            perp = v - proj
            distance = np.linalg.norm(perp)
            
            if distance > max_distance:
                max_distance = distance
                max_point = point
        
        direction = max_point - start
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-6:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction = (direction / direction_norm).astype(np.float32)
    
    return curve_length, straight_distance, curvature, direction


def create_10d_position(xyz, avg_radius, curve_length, straight_distance, curvature, direction):
    return np.array([
        xyz[0], xyz[1], xyz[2],
        avg_radius,
        curve_length,
        straight_distance,
        curvature,
        direction[0], direction[1], direction[2]
    ], dtype=np.float32)


def compute_max_deviation(points: np.ndarray) -> float:
    points = np.array(points[:, :3] if points.ndim == 2 and points.shape[1] > 3 else points)
    
    if len(points) < 2:
        return 0.0
    
    start = points[0]
    end = points[-1]
    chord = end - start
    chord_length = np.linalg.norm(chord)
    
    if chord_length < 1e-6:
        return float(np.max(np.linalg.norm(points - start, axis=1)))
    
    chord_dir = chord / chord_length
    max_deviation = 0.0
    
    for p in points:
        v = p - start
        proj_length = np.dot(v, chord_dir)
        proj = proj_length * chord_dir
        perp = v - proj
        deviation = np.linalg.norm(perp)
        max_deviation = max(max_deviation, deviation)
    
    return float(max_deviation)


def needs_hint_node(segment: dict, deviation_threshold: float = 2.0) -> bool:
    points = segment['points']
    max_dev = compute_max_deviation(points)
    return max_dev >= deviation_threshold


def calculate_auto_hints_per_segment_for_files(vtp_files, target_nodes=25, deviation_threshold=2.0):
    original_nodes_list = []
    hint_segments_list = []

    sample_files = vtp_files[:min(100, len(vtp_files))]

    for vtp_path in sample_files:
        try:
            _, _, _, segments = load_vtp(vtp_path)
            if len(segments) == 0:
                continue

            endpoint_set = set()
            segments_needing_hints = 0

            for seg in segments:
                points = seg['points']
                start_pos = tuple(np.round(points[0][:3], 5))
                end_pos = tuple(np.round(points[-1][:3], 5))
                endpoint_set.add(start_pos)
                endpoint_set.add(end_pos)

                max_dev = compute_max_deviation(points)
                if max_dev >= deviation_threshold:
                    segments_needing_hints += 1

            original_nodes_list.append(len(endpoint_set))
            hint_segments_list.append(segments_needing_hints)

        except Exception as e:
            continue

    if not original_nodes_list:
        return 2

    avg_original_nodes = np.mean(original_nodes_list)
    avg_hint_segments = np.mean(hint_segments_list)

    print(f"  [Analysis] {len(original_nodes_list)} sample trees")
    print(f"  - Avg original nodes: {avg_original_nodes:.1f}")
    print(f"  - Avg hint segments: {avg_hint_segments:.1f}")

    if avg_hint_segments < 0.1:
        return 2


    hints_needed = (target_nodes - avg_original_nodes) / avg_hint_segments
    hints_per_segment = max(2, min(8, int(round(hints_needed))))

    print(f"  - Required hints/segment: {hints_needed:.1f} -> {hints_per_segment}")

    return hints_per_segment


def build_key_graph_with_hints(segments, use_hint_nodes=True, hint_deviation_threshold=2.0,
                                hints_per_segment=2, auto_adjust_hints=False, target_nodes=25):
    if auto_adjust_hints and use_hint_nodes:
        hints_per_segment = calculate_auto_hints_per_segment_for_files(
            segments, target_nodes=target_nodes, deviation_threshold=hint_deviation_threshold
        )
    endpoint_to_segs = {}
    endpoint_properties = {}
    
    for seg_idx, seg in enumerate(segments):
        points = seg['points']
        start_pos = tuple(np.round(points[0][:3], 5))
        end_pos = tuple(np.round(points[-1][:3], 5))
        
        curve_length, straight_distance, curvature, direction = compute_segment_properties(points)
        avg_radius = float(np.mean(seg['radius']))
        
        seg_props = {
            'curve_length': curve_length,
            'straight_distance': straight_distance,
            'curvature': curvature,
            'direction': direction,
            'avg_radius': avg_radius
        }
        
        if start_pos not in endpoint_to_segs:
            endpoint_to_segs[start_pos] = []
            endpoint_properties[start_pos] = []
        endpoint_to_segs[start_pos].append((seg_idx, 'start'))
        endpoint_properties[start_pos].append(seg_props)
        
        if end_pos not in endpoint_to_segs:
            endpoint_to_segs[end_pos] = []
            endpoint_properties[end_pos] = []
        endpoint_to_segs[end_pos].append((seg_idx, 'end'))
        endpoint_properties[end_pos].append(seg_props)
    
    key_graph = nx.Graph()
    pos_to_node = {}
    node_id = 0
    
    for pos in endpoint_to_segs.keys():
        pos_to_node[pos] = node_id
        
        props_list = endpoint_properties[pos]
        avg_radius = np.mean([p['avg_radius'] for p in props_list])
        avg_curve_length = np.mean([p['curve_length'] for p in props_list])
        avg_straight_distance = np.mean([p['straight_distance'] for p in props_list])
        avg_curvature = np.mean([p['curvature'] for p in props_list])
        
        direction = props_list[0]['direction']
        
        position_10d = create_10d_position(
            xyz=np.array(pos[:3], dtype=np.float32),
            avg_radius=avg_radius,
            curve_length=avg_curve_length,
            straight_distance=avg_straight_distance,
            curvature=avg_curvature,
            direction=direction
        )
        
        key_graph.add_node(
            node_id, 
            position=position_10d,
            node_type='endpoint'
        )
        node_id += 1
    
    hint_count = 0
    
    for seg_idx, seg in enumerate(segments):
        points = seg['points']
        start_pos = tuple(np.round(points[0][:3], 5))
        end_pos = tuple(np.round(points[-1][:3], 5))
        
        start_node = pos_to_node[start_pos]
        end_node = pos_to_node[end_pos]
        
        if start_node == end_node:
            continue
        
        edge_radius_mean = float(np.mean(seg['radius']))
        
        curve_length, straight_distance, curvature, direction = compute_segment_properties(points)
        
        if use_hint_nodes and needs_hint_node(seg, hint_deviation_threshold):
            n = len(points)

            hint_indices = []
            for i in range(1, hints_per_segment + 1):
                t = i / (hints_per_segment + 1)
                idx = max(1, min(n - 2, int(t * n)))
                hint_indices.append(idx)

            hint_indices = sorted(set(hint_indices))
            actual_hints = len(hint_indices)

            hint_node_ids = []
            prev_idx = 0

            for h_idx, seg_end_idx in enumerate(hint_indices):
                hint_xyz = points[seg_end_idx][:3].copy()

                seg_points = points[prev_idx:seg_end_idx+1]
                seg_radii = seg['radius'][prev_idx:seg_end_idx+1]
                l, d, k, dir_vec = compute_segment_properties(seg_points)
                r = float(np.mean(seg_radii))

                hint_id = node_id
                position_10d_hint = create_10d_position(hint_xyz, r, l, d, k, dir_vec)
                key_graph.add_node(
                    hint_id,
                    position=position_10d_hint,
                    node_type='hint'
                )
                hint_node_ids.append(hint_id)
                node_id += 1
                prev_idx = seg_end_idx

            all_nodes = [start_node] + hint_node_ids + [end_node]
            for i in range(len(all_nodes) - 1):
                key_graph.add_edge(
                    all_nodes[i], all_nodes[i+1],
                    segment_idx=seg_idx,
                    radius_mean=edge_radius_mean
                )

            hint_count += actual_hints
        else:
            key_graph.add_edge(
                start_node, end_node,
                segment_idx=seg_idx,
                radius_mean=edge_radius_mean
            )
    
    return key_graph, hint_count


def normalize_graph(graph):
    positions = np.array([graph.nodes[n]['position'][:3] for n in graph.nodes()])
    
    if len(positions) == 0:
        return graph
    
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    center = (min_coords + max_coords) / 2
    extent = np.max(max_coords - min_coords)
    
    coord_scale = 2.0 / extent if extent > 1e-6 else 1.0
    
    all_radii = []
    all_curve_lengths = []
    all_straight_distances = []
    
    for node in graph.nodes():
        pos = graph.nodes[node]['position']
        all_radii.append(pos[3])
        all_curve_lengths.append(pos[4])
        all_straight_distances.append(pos[5])
    
    max_radius = max(all_radii) if all_radii else 1.0
    max_curve_length = max(all_curve_lengths) if all_curve_lengths else 1.0
    max_straight_distance = max(all_straight_distances) if all_straight_distances else 1.0
    
    max_radius = max(max_radius, 1e-6)
    max_curve_length = max(max_curve_length, 1e-6)
    max_straight_distance = max(max_straight_distance, 1e-6)
    
    graph.graph['norm_center'] = center.astype(np.float32)
    graph.graph['norm_scale'] = float(coord_scale)
    graph.graph['original_extent'] = float(extent)
    graph.graph['max_radius'] = float(max_radius)
    graph.graph['max_curve_length'] = float(max_curve_length)
    graph.graph['max_straight_distance'] = float(max_straight_distance)
    
    for node in graph.nodes():
        pos_10d = graph.nodes[node]['position'].copy()
        
        pos_10d[:3] = (pos_10d[:3] - center) * coord_scale
        
        pos_10d[3] = pos_10d[3] / max_radius
        pos_10d[4] = pos_10d[4] / max_curve_length
        pos_10d[5] = pos_10d[5] / max_straight_distance

        
        graph.nodes[node]['position'] = pos_10d.astype(np.float32)
    
    return graph


def find_root_node(graph):
    components = list(nx.connected_components(graph))
    if len(components) > 1:
        largest_comp = max(components, key=len)
        candidate_nodes = largest_comp
        if len(components) > 1:
            total = graph.number_of_nodes()
            largest_size = len(largest_comp)
            print(f"  [Multi-component] {len(components)} components detected, "
                  f"using largest ({largest_size}/{total} nodes, "
                  f"{100*largest_size/total:.1f}%)")
    else:
        candidate_nodes = set(graph.nodes())

    leaf_nodes = []
    for n in candidate_nodes:
        node_type = graph.nodes[n].get('node_type', 'endpoint')
        if node_type == 'endpoint' and graph.degree(n) == 1:
            leaf_nodes.append(n)

    if not leaf_nodes:
        leaf_nodes = [n for n in candidate_nodes if graph.degree(n) == 1]

    if not leaf_nodes:
        return list(candidate_nodes)[0]

    def get_leaf_edge_radius(node):
        neighbors = list(graph.neighbors(node))
        if neighbors:
            edge_data = graph.edges[node, neighbors[0]]
            return edge_data.get('radius_mean', 0.0)
        return 0.0

    root = max(leaf_nodes, key=get_leaf_edge_radius)
    return root


def graph_to_binary_tree(graph):
    if graph.number_of_nodes() == 0:
        return None, 0, None, {}
    
    root_node_id = find_root_node(graph)
    
    dfs_tree = nx.dfs_tree(graph, source=root_node_id)
    max_level = nx.dag_longest_path_length(dfs_tree) if dfs_tree.number_of_nodes() > 1 else 0
    total_level = sum(nx.shortest_path_length(dfs_tree, root_node_id).values())
    
    nodes_dict = {}
    for node in dfs_tree.nodes():
        position = graph.nodes[node]['position']
        node_type = graph.nodes[node].get('node_type', 'endpoint')
        level = nx.shortest_path_length(dfs_tree, root_node_id, node)
        
        nodes_dict[node] = Node(
            value=node,
            radius=torch.tensor(position, dtype=torch.float32),
            left=None,
            right=None,
            level=level,
            treelevel=total_level,
            maxlevel=max_level,
            node_type=node_type
        )
    
    edge_radii = {}
    
    for node in dfs_tree.nodes():
        children = list(dfs_tree.successors(node))
        
        for child in children:
            if graph.has_edge(node, child):
                edge_data = graph.edges[node, child]
                radius_mean = edge_data.get('radius_mean', 1.0)
                edge_radii[(node, child)] = radius_mean
        
        if len(children) == 2:
            parent_pos = np.array(graph.nodes[node]['position'][:3])
            child1_pos = np.array(graph.nodes[children[0]]['position'][:3])
            child2_pos = np.array(graph.nodes[children[1]]['position'][:3])
            
            vec1 = child1_pos - parent_pos
            vec2 = child2_pos - parent_pos
            cross_y = np.cross(vec1, vec2)[1]
            
            if cross_y > 0:
                nodes_dict[node].left = nodes_dict[children[0]]
                nodes_dict[node].right = nodes_dict[children[1]]
            else:
                nodes_dict[node].left = nodes_dict[children[1]]
                nodes_dict[node].right = nodes_dict[children[0]]
                
        elif len(children) == 1:
            nodes_dict[node].left = nodes_dict[children[0]]
    
    return nodes_dict[root_node_id], len(graph), root_node_id, edge_radii


def process_tree_dataset(vtp_files, output_path, use_hint_nodes=True, hint_deviation_threshold=2.0,
                         hints_per_segment=2, auto_adjust_hints=False, target_nodes=25):
    actual_hints_per_segment = hints_per_segment
    if auto_adjust_hints and use_hint_nodes:
        print(f"[Auto hint adjustment] Analyzing dataset...")
        actual_hints_per_segment = calculate_auto_hints_per_segment_for_files(
            vtp_files, target_nodes=target_nodes, deviation_threshold=hint_deviation_threshold
        )
        print(f"[Auto hint adjustment] Final hints_per_segment: {actual_hints_per_segment}")

    print(f"="*60)
    print(f"Stage 1 (Tree) Preprocessing (10D Node Attributes)")
    print(f"  Node attributes: 10D [x, y, z, r, l, d, k, nx, ny, nz]")
    print(f"  Hint Node: {'ON' if use_hint_nodes else 'OFF'}")
    if use_hint_nodes:
        print(f"  Hint Threshold: {hint_deviation_threshold:.1f} mm (max deviation)")
        print(f"  Hints per segment: {actual_hints_per_segment} {'(auto)' if auto_adjust_hints else '(manual)'}")
        if auto_adjust_hints:
            print(f"  Target nodes: {target_nodes}")
    print(f"  Edge radii: stored separately (edge_radii)")
    print(f"  Scale info: stored (scale_infos)")
    print(f"  Input files: {len(vtp_files)}")
    print(f"="*60)
    
    trees = []
    graphs = []
    num_nodes_list = []
    file_names = []
    scale_infos = []
    edge_radii_list = []
    
    total_hint_pairs = 0
    total_segments = 0
    segments_with_hint = 0
    
    for i, vtp_path in enumerate(vtp_files):
        filename = os.path.basename(vtp_path)
        
        try:
            coords, radii, labels, segments = load_vtp(vtp_path)
            
            if len(segments) == 0:
                print(f"  [{i+1}] {filename}: skip (no segments)")
                continue
            
            total_segments += len(segments)
            
            key_graph, hint_count = build_key_graph_with_hints(
                segments,
                use_hint_nodes=use_hint_nodes,
                hint_deviation_threshold=hint_deviation_threshold,
                hints_per_segment=actual_hints_per_segment,
                auto_adjust_hints=False,
                target_nodes=target_nodes
            )
            
            total_hint_pairs += hint_count
            segments_with_hint += hint_count
            
            key_graph = normalize_graph(key_graph)
            
            scale_info = {
                'center': key_graph.graph.get('norm_center', np.zeros(3, dtype=np.float32)),
                'scale': key_graph.graph.get('norm_scale', 1.0),
                'extent': key_graph.graph.get('original_extent', 1.0),
                'max_radius': key_graph.graph.get('max_radius', 1.0),
                'max_curve_length': key_graph.graph.get('max_curve_length', 1.0),
                'max_straight_distance': key_graph.graph.get('max_straight_distance', 1.0)
            }
            
            root_node, num_nodes, root_id, edge_radii = graph_to_binary_tree(key_graph)
            
            if root_node is None:
                print(f"  [{i+1}] {filename}: skip (tree creation failed)")
                continue
            
            trees.append(root_node)
            graphs.append(key_graph)
            num_nodes_list.append(num_nodes)
            file_names.append(filename)
            scale_infos.append(scale_info)
            edge_radii_list.append(edge_radii)
            
            if (i + 1) % 100 == 0 or i == 0:
                hint_str = f", hints: {hint_count}" if use_hint_nodes else ""
                print(f"  [{i+1}/{len(vtp_files)}] {filename}: {num_nodes} nodes, {len(edge_radii)} edges{hint_str}")
                
        except Exception as e:
            print(f"  [{i+1}] {filename}: error - {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(trees) == 0:
        print("Error: No valid trees!")
        return None

    print(f"\nResult:")
    print(f"  Total trees: {len(trees)}")
    print(f"  Node count range: {min(num_nodes_list)} ~ {max(num_nodes_list)}")
    print(f"  Node attribute dim: 10D")

    if use_hint_nodes:
        print(f"\nHint Node Statistics:")
        print(f"  Total segments: {total_segments}")
        print(f"  Segments with hints: {segments_with_hint}")
        print(f"  Hint insertion rate: {100*segments_with_hint/max(total_segments,1):.1f}%")
        print(f"  Hints per segment: {actual_hints_per_segment}")
        print(f"  Total hint nodes: {total_hint_pairs}")

    hint_stats = {
        'enabled': use_hint_nodes,
        'threshold': hint_deviation_threshold,
        'hints_per_segment': actual_hints_per_segment,
        'auto_adjusted': auto_adjust_hints,
        'target_nodes': target_nodes if auto_adjust_hints else None,
        'total_segments': total_segments,
        'segments_with_hint': segments_with_hint,
        'total_hint_nodes': total_hint_pairs
    }
    
    output_data = {
        'data': trees,
        'graphs': graphs,
        'num_nodes': num_nodes_list,
        'file_names': file_names,
        'scale_infos': scale_infos,
        'edge_radii': edge_radii_list,
        'hint_stats': hint_stats
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torch.save(output_data, output_path)
    
    print(f"\nSaved: {output_path}")
    print(f"  Trees: {len(trees)}")
    print(f"  Node attr dim: 10D")
    
    return output_data


def rotate_curve(curve):
    start = curve[0]
    end = curve[-1]
    direction = end - start
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    
    target = np.array([1, 0, 0], dtype=np.float32)
    
    v = np.cross(direction, target)
    s = np.linalg.norm(v)
    c = np.dot(direction, target)
    
    if s < 1e-8:
        if c > 0:
            R = np.eye(3)
        else:
            R = np.diag([-1, 1, -1])
    else:
        v = v / s
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx * s + vx @ vx * (1 - c)
    
    curve_centered = curve - start
    rotated = (R @ curve_centered.T).T
    
    return rotated.astype(np.float32)


def resample_curve(curve, target_distance=0.01):
    distances = np.linalg.norm(np.diff(curve[:, :3], axis=0), axis=1)
    cumulative = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative[-1]
    
    if total_length < 1e-6:
        return curve
    
    num_points = max(2, int(total_length / target_distance) + 1)
    new_distances = np.linspace(0, total_length, num_points)
    
    resampled = np.zeros((num_points, curve.shape[1]), dtype=np.float32)
    for i, d in enumerate(new_distances):
        idx = np.searchsorted(cumulative, d, side='right') - 1
        idx = min(idx, len(curve) - 2)
        
        if cumulative[idx + 1] - cumulative[idx] < 1e-8:
            t = 0
        else:
            t = (d - cumulative[idx]) / (cumulative[idx + 1] - cumulative[idx])
        
        resampled[i] = curve[idx] + t * (curve[idx + 1] - curve[idx])
    
    return resampled


def calculate_distance_features(points):
    n_pts = len(points)
    start = points[0, :3]
    end = points[-1, :3]
    
    result = np.zeros((n_pts, 8), dtype=np.float32)
    result[:, :3] = points[:, :3]
    result[:, 3] = points[:, 3] if points.shape[1] > 3 else 0
    
    for i in range(n_pts):
        result[i, 4] = np.linalg.norm(points[i, :3] - start)
        result[i, 5] = np.linalg.norm(points[i, :3] - end)
        
        if i == 0:
            result[i, 6] = 0
            result[i, 7] = 0
        else:
            result[i, 6] = np.linalg.norm(points[i, :3] - points[i-1, :3])
            cumulative = np.sum(np.linalg.norm(np.diff(points[:i+1, :3], axis=0), axis=1))
            result[i, 7] = cumulative
    
    return result.astype(np.float32)


def process_sequence_dataset(vtp_files, output_path, max_seq_len=200):
    print(f"="*60)
    print(f"Stage 2 (Sequence) Preprocessing")
    print(f"  Input files: {len(vtp_files)}")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"="*60)
    
    all_sequences = []
    all_conditions = []
    all_lengths = []
    all_filenames = []
    
    for i, vtp_path in enumerate(vtp_files):
        filename = os.path.basename(vtp_path)
        
        try:
            coords, radii, labels, segments = load_vtp(vtp_path)
            
            key_graph, _ = build_key_graph_with_hints(segments, use_hint_nodes=False)
            root_id = find_root_node(key_graph)
            
            depth_dict = nx.single_source_shortest_path_length(key_graph, root_id)
            max_depth = max(depth_dict.values()) if depth_dict else 1
            
            for seg_idx, seg in enumerate(segments):
                points = seg['points']
                seg_radii = seg['radius']
                
                if len(points) < 5:
                    continue
                
                rotated_coords = rotate_curve(points[:, :3] if points.ndim == 2 and points.shape[1] > 3 else points)
                
                rotated_with_radius = np.hstack((rotated_coords, (seg_radii * 100).reshape(-1, 1)))
                
                sample_points = resample_curve(rotated_with_radius, target_distance=0.01)
                
                if len(sample_points) < 2:
                    continue
                
                data = calculate_distance_features(sample_points)
                
                curve_length, straight_distance, curvature, _ = compute_segment_properties(points)
                avg_radius = float(np.mean(seg_radii) * 100)
                
                start_pos = tuple(np.round(points[0][:3], 5))
                seg_depth = 0
                for node in key_graph.nodes():
                    node_pos = tuple(np.round(key_graph.nodes[node]['position'][:3], 5))
                    if np.linalg.norm(np.array(start_pos) - np.array(node_pos)) < 0.1:
                        seg_depth = depth_dict.get(node, 0)
                        break
                
                normalized_depth = seg_depth / max_depth if max_depth > 0 else 0.0
                
                condition = [avg_radius, curve_length, straight_distance, curvature, normalized_depth]
                
                all_sequences.append(data)
                all_conditions.append(condition)
                all_lengths.append(data.shape[0])
                all_filenames.append(f"{filename}_{seg_idx}")
            
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i+1}/{len(vtp_files)}] {filename}: segments processed")
                
        except Exception as e:
            print(f"  [{i+1}] {filename}: error - {e}")
            continue

    if len(all_sequences) == 0:
        print("Error: No valid sequences!")
        return None

    print(f"\nResult:")
    print(f"  Total sequences: {len(all_sequences)}")
    print(f"  Sequence length range: {min(all_lengths)} ~ {max(all_lengths)}")
    
    final_max_len = min(max_seq_len, max(all_lengths))
    print(f"  Final sequence length: {final_max_len}")
    
    padded_sequences = np.zeros((len(all_sequences), final_max_len, 8), dtype=np.float32)
    
    for i, seq in enumerate(all_sequences):
        seq_len = min(seq.shape[0], final_max_len)
        padded_sequences[i, :seq_len] = seq[:seq_len]
    
    conditions_array = np.array(all_conditions, dtype=np.float32)
    
    nan_mask_seq = np.isnan(padded_sequences) | np.isinf(padded_sequences)
    if nan_mask_seq.any():
        print(f"  WARNING: {np.sum(nan_mask_seq)} NaN/Inf values found - replaced with 0")
        padded_sequences[nan_mask_seq] = 0.0

    nan_mask_cond = np.isnan(conditions_array) | np.isinf(conditions_array)
    if nan_mask_cond.any():
        print(f"  WARNING: {np.sum(nan_mask_cond)} NaN/Inf values in conditions - replaced with 0")
        conditions_array[nan_mask_cond] = 0.0
    
    scale_factors = {}
    
    radius_data = padded_sequences[:, :, 3]
    non_zero_mask = radius_data != 0
    if non_zero_mask.any():
        scale_factors['seq_radius_divisor'] = 100.0
        padded_sequences[:, :, 3] = np.where(non_zero_mask, radius_data / 100.0, 0)
    
    scale_factors['cond_radius_divisor'] = 100.0
    conditions_array[:, 0] = conditions_array[:, 0] / 100.0
    
    length_max = conditions_array[:, 1].max()
    if length_max > 10:
        scale_factors['cond_length_divisor'] = length_max / 10.0
        conditions_array[:, 1] = conditions_array[:, 1] / scale_factors['cond_length_divisor']
    else:
        scale_factors['cond_length_divisor'] = 1.0
    
    distance_max = conditions_array[:, 2].max()
    if distance_max > 10:
        scale_factors['cond_distance_divisor'] = distance_max / 10.0
        conditions_array[:, 2] = conditions_array[:, 2] / scale_factors['cond_distance_divisor']
    else:
        scale_factors['cond_distance_divisor'] = 1.0
    
    output_data = {
        'data': torch.from_numpy(padded_sequences),
        'condition': torch.from_numpy(conditions_array),
        'length': torch.tensor(all_lengths, dtype=torch.int32),
        'filenames': all_filenames,
        'scale_factors': scale_factors
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torch.save(output_data, output_path)
    
    print(f"\n  Saved: {output_path}")
    print(f"  Data shape: {output_data['data'].shape}")
    print(f"  Condition shape: {output_data['condition'].shape}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description='VTP to PartVessel Preprocessing (10D)')

    parser.add_argument('mode', choices=['tree', 'seq'],
                        help='Preprocessing mode: tree (Stage 1) or seq (Stage 2)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='VTP file directory')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output PT file path')
    parser.add_argument('--pattern', type=str, default='*.vtp',
                        help='VTP file pattern (default: *.vtp)')
    parser.add_argument('--max_seq_len', type=int, default=200,
                        help='Stage 2 max sequence length')

    parser.add_argument('--use_hint_nodes', type=bool, default=True,
                        help='Whether to insert hint nodes (default True)')
    parser.add_argument('--hint_deviation_threshold', type=float, default=2.0,
                        help='Hint insertion threshold - max deviation (mm, default 2.0)')
    parser.add_argument('--hints_per_segment', type=int, default=2,
                        help='Hints per segment (default 2, ignored if auto_adjust)')
    parser.add_argument('--auto_adjust_hints', action='store_true',
                        help='Auto-adjust hints based on dataset complexity (recommended)')
    parser.add_argument('--target_nodes', type=int, default=25,
                        help='Target node count for auto-adjustment (default 25)')

    args = parser.parse_args()
    
    search_pattern = os.path.join(args.input_dir, args.pattern)
    vtp_files = sorted(glob(search_pattern))
    
    if not vtp_files:
        print(f"Error: No VTP files found: {search_pattern}")
        return 1
    
    print(f"Found VTP files: {len(vtp_files)}")
    
    if args.mode == 'tree':
        result = process_tree_dataset(
            vtp_files,
            args.output_path,
            use_hint_nodes=args.use_hint_nodes,
            hint_deviation_threshold=args.hint_deviation_threshold,
            hints_per_segment=args.hints_per_segment,
            auto_adjust_hints=args.auto_adjust_hints,
            target_nodes=args.target_nodes
        )
    else:
        result = process_sequence_dataset(vtp_files, args.output_path, args.max_seq_len)
    
    return 0 if result is not None else 1


if __name__ == '__main__':
    sys.exit(main())
