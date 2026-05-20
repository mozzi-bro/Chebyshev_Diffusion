
import os
import sys
import argparse
import numpy as np
import torch
import networkx as nx
from glob import glob
from typing import List, Dict, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import read_ply, ply_to_graph


def load_generated_plys(gen_dir: str, pattern: str) -> Tuple[List[nx.Graph], List[np.ndarray]]:
    ply_files = sorted(glob(os.path.join(gen_dir, pattern)))

    if not ply_files:
        raise FileNotFoundError(f"No PLY files found: {os.path.join(gen_dir, pattern)}")

    graphs = []
    point_clouds = []

    for ply_path in ply_files:
        try:
            graph = ply_to_graph(ply_path)
            graphs.append(graph)

            vertices, _ = read_ply(ply_path)
            point_clouds.append(vertices[:, :3] if vertices.shape[1] > 3 else vertices)
        except Exception as e:
            print(f"  Warning: failed to load {os.path.basename(ply_path)} - {e}")
            continue

    print(f"Loaded PLY files: {len(graphs)}")
    if graphs:
        print(f"  Node count range: {min(g.number_of_nodes() for g in graphs)} ~ {max(g.number_of_nodes() for g in graphs)}")
    return graphs, point_clouds


def load_gt_key_graphs(pt_path: str) -> List[nx.Graph]:
    data = torch.load(pt_path, weights_only=False)

    if 'graphs' not in data:
        raise ValueError("PT file does not contain 'graphs' key")

    graphs = data['graphs']
    print(f"GT Key Graphs: {len(graphs)}")
    print(f"  Node count range: {min(g.number_of_nodes() for g in graphs)} ~ {max(g.number_of_nodes() for g in graphs)}")
    return graphs


def load_gt_vtp_as_pointclouds(vtp_dir: str, pattern: str = '*_final_smoothed.vtp',
                                num_points: int = 2048) -> np.ndarray:
    import vtk

    vtp_files = sorted(glob(os.path.join(vtp_dir, pattern)))
    print(f"VTP files: {len(vtp_files)}")

    all_pcs = []

    for vtp_path in tqdm(vtp_files, desc="Loading VTP"):
        try:
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(vtp_path)
            reader.Update()
            polydata = reader.GetOutput()

            points = polydata.GetPoints()
            n_pts = points.GetNumberOfPoints()
            coords = np.array([points.GetPoint(i) for i in range(n_pts)], dtype=np.float32)

            centroid = np.mean(coords, axis=0)
            coords_centered = coords - centroid
            max_dist = np.max(np.linalg.norm(coords_centered, axis=1))
            if max_dist > 0:
                coords_normalized = coords_centered / max_dist
            else:
                coords_normalized = coords_centered

            if len(coords_normalized) >= num_points:
                idx = np.random.choice(len(coords_normalized), num_points, replace=False)
            else:
                idx = np.random.choice(len(coords_normalized), num_points, replace=True)

            all_pcs.append(coords_normalized[idx])

        except Exception as e:
            print(f"  Warning: failed to load {os.path.basename(vtp_path)} - {e}")
            continue

    result = np.stack(all_pcs, axis=0)
    print(f"GT point clouds: {result.shape}")
    return result


def load_gen_ply_as_pointclouds(gen_dir: str, pattern: str, num_points: int = 2048) -> np.ndarray:
    ply_files = sorted(glob(os.path.join(gen_dir, pattern)))

    all_pcs = []

    for ply_path in tqdm(ply_files, desc="Loading PLY"):
        try:
            vertices, _ = read_ply(ply_path)
            coords = vertices[:, :3]

            centroid = np.mean(coords, axis=0)
            coords_centered = coords - centroid
            max_dist = np.max(np.linalg.norm(coords_centered, axis=1))
            if max_dist > 0:
                coords_normalized = coords_centered / max_dist
            else:
                coords_normalized = coords_centered

            if len(coords_normalized) >= num_points:
                idx = np.random.choice(len(coords_normalized), num_points, replace=False)
            else:
                idx = np.random.choice(len(coords_normalized), num_points, replace=True)

            all_pcs.append(coords_normalized[idx])

        except Exception as e:
            print(f"  Warning: failed to load {os.path.basename(ply_path)} - {e}")
            continue

    result = np.stack(all_pcs, axis=0)
    print(f"Generated point clouds: {result.shape}")
    return result


def jsd_between_point_cloud_sets(sample_pcs: np.ndarray, ref_pcs: np.ndarray,
                                  resolution: int = 28) -> float:
    from scipy.stats import entropy
    from sklearn.neighbors import NearestNeighbors

    grid = np.zeros((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    grid_coords = grid.reshape(-1, 3)

    def compute_occupancy(pclouds):
        grid_counters = np.zeros(len(grid_coords))
        nn = NearestNeighbors(n_neighbors=1).fit(grid_coords)

        for pc in tqdm(pclouds, desc="JSD occupancy", leave=False):
            _, indices = nn.kneighbors(pc)
            indices = indices.flatten()
            for i in indices:
                grid_counters[i] += 1

        return grid_counters

    sample_counts = compute_occupancy(sample_pcs)
    ref_counts = compute_occupancy(ref_pcs)

    P = sample_counts / (np.sum(sample_counts) + 1e-10)
    Q = ref_counts / (np.sum(ref_counts) + 1e-10)

    M = 0.5 * (P + Q)

    def kl_div(A, B):
        idx = np.logical_and(A > 0, B > 0)
        return np.sum(A[idx] * np.log2(A[idx] / B[idx]))

    jsd = 0.5 * kl_div(P, M) + 0.5 * kl_div(Q, M)
    return float(jsd)


def compute_chamfer_distance(sample_pcs: np.ndarray, ref_pcs: np.ndarray) -> float:
    from scipy.spatial.distance import cdist

    n = min(len(sample_pcs), len(ref_pcs))
    total_cd = 0.0

    for i in tqdm(range(n), desc="Computing CD"):
        dist_matrix = cdist(sample_pcs[i], ref_pcs[i], metric='sqeuclidean')
        cd1 = np.mean(np.min(dist_matrix, axis=1))
        cd2 = np.mean(np.min(dist_matrix, axis=0))
        total_cd += (cd1 + cd2) / 2

    return total_cd / n


def compute_point_metrics(gt_pcs: np.ndarray, gen_pcs: np.ndarray) -> Dict[str, float]:
    print("\n[Point-based Metrics]")

    results = {}

    print("  Computing JSD...")
    results['JSD'] = jsd_between_point_cloud_sets(gen_pcs, gt_pcs) * 1000

    print("  Computing CD...")
    results['CD'] = compute_chamfer_distance(gen_pcs, gt_pcs) * 1000

    return results


def degree_distribution(G: nx.Graph) -> np.ndarray:
    degrees = [d for _, d in G.degree()]
    if not degrees:
        return np.array([0])
    max_deg = max(degrees) + 1
    hist, _ = np.histogram(degrees, bins=range(max_deg + 1), density=True)
    return hist


def laplacian_spectrum(G: nx.Graph, k: int = 20) -> np.ndarray:
    from scipy.linalg import eigvalsh

    if G.number_of_nodes() == 0:
        return np.zeros(k)

    laplacian = nx.normalized_laplacian_matrix(G).toarray()
    eigenvalues = eigvalsh(laplacian)

    result = np.sort(eigenvalues)[:k]
    if len(result) < k:
        result = np.pad(result, (0, k - len(result)))
    return result


def rbf_mmd(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    def kernel(a, b):
        return np.exp(-np.sum((a - b) ** 2) / (2 * sigma ** 2))

    n1, n2 = len(X), len(Y)

    if n1 > 100:
        idx1 = np.random.choice(n1, 100, replace=False)
        X = X[idx1]
        n1 = 100
    if n2 > 100:
        idx2 = np.random.choice(n2, 100, replace=False)
        Y = Y[idx2]
        n2 = 100

    k11 = sum(kernel(X[i], X[j]) for i in range(n1) for j in range(n1)) / (n1 * n1)
    k22 = sum(kernel(Y[i], Y[j]) for i in range(n2) for j in range(n2)) / (n2 * n2)
    k12 = sum(kernel(X[i], Y[j]) for i in range(n1) for j in range(n2)) / (n1 * n2)

    return max(0, k11 + k22 - 2 * k12)


def compute_gwd_simple(gt_graphs: List[nx.Graph], gen_graphs: List[nx.Graph]) -> float:
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    n = min(len(gt_graphs), len(gen_graphs))
    total_gwd = 0.0

    for i in range(n):
        gt_pos = np.array([gt_graphs[i].nodes[n].get('position', [0,0,0])[:3]
                          for n in gt_graphs[i].nodes()])
        gen_pos = np.array([gen_graphs[i].nodes[n].get('position', [0,0,0])[:3]
                           for n in gen_graphs[i].nodes()])

        if len(gt_pos) == 0 or len(gen_pos) == 0:
            continue

        cost = cdist(gt_pos, gen_pos)

        max_size = max(len(gt_pos), len(gen_pos))
        if cost.shape[0] != cost.shape[1]:
            padded_cost = np.full((max_size, max_size), cost.max() * 2)
            padded_cost[:cost.shape[0], :cost.shape[1]] = cost
            cost = padded_cost

        row_ind, col_ind = linear_sum_assignment(cost)
        total_gwd += cost[row_ind, col_ind].mean()

    return total_gwd / n if n > 0 else 0.0


def compute_graph_metrics(gt_graphs: List[nx.Graph], gen_graphs: List[nx.Graph]) -> Dict[str, float]:
    print("\n[Graph-based Metrics]")

    results = {}

    print("  Computing Deg. MMD...")
    gt_degs = [degree_distribution(g) for g in tqdm(gt_graphs, desc="GT degree", leave=False)]
    gen_degs = [degree_distribution(g) for g in tqdm(gen_graphs, desc="Gen degree", leave=False)]

    max_len = max(max(len(d) for d in gt_degs), max(len(d) for d in gen_degs))
    gt_padded = np.array([np.pad(d, (0, max_len - len(d))) for d in gt_degs])
    gen_padded = np.array([np.pad(d, (0, max_len - len(d))) for d in gen_degs])

    results['Deg'] = rbf_mmd(gt_padded, gen_padded)

    print("  Computing Spec. MMD...")
    gt_specs = [laplacian_spectrum(g) for g in tqdm(gt_graphs, desc="GT spectrum", leave=False)]
    gen_specs = [laplacian_spectrum(g) for g in tqdm(gen_graphs, desc="Gen spectrum", leave=False)]

    gt_spec_arr = np.array(gt_specs)
    gen_spec_arr = np.array(gen_specs)

    results['Spec'] = rbf_mmd(gt_spec_arr, gen_spec_arr)

    print("  Computing GWD...")
    results['GWD'] = compute_gwd_simple(gt_graphs, gen_graphs)

    return results


def main():
    parser = argparse.ArgumentParser(description='PartVessel Quantitative Evaluation')
    parser.add_argument('mode', choices=['point', 'graph', 'all'],
                        help='Evaluation mode: point, graph, all')

    parser.add_argument('--gt_vtp_dir', type=str, help='GT VTP directory')
    parser.add_argument('--gen_dir', type=str, required=True, help='Generated results directory')
    parser.add_argument('--pattern', type=str, default='gen_skeleton_*.ply',
                        help='Generated PLY file pattern')

    parser.add_argument('--gt_pt', type=str, help='GT Key Graph PT file')
    parser.add_argument('--tree_pattern', type=str, default='gen_tree_*.ply',
                        help='Generated Tree PLY file pattern')

    parser.add_argument('--num_points', type=int, default=2048, help='Number of sampled points')
    parser.add_argument('--output', type=str, help='Output file for results')

    args = parser.parse_args()

    results = {}

    if args.mode in ['point', 'all']:
        if not args.gt_vtp_dir:
            raise ValueError("--gt_vtp_dir is required for point metrics")

        print("="*60)
        print("Point-based Metrics (Full Skeleton)")
        print("="*60)

        gt_pcs = load_gt_vtp_as_pointclouds(args.gt_vtp_dir, num_points=args.num_points)
        gen_pcs = load_gen_ply_as_pointclouds(args.gen_dir, args.pattern, num_points=args.num_points)

        point_results = compute_point_metrics(gt_pcs, gen_pcs)
        results.update(point_results)

    if args.mode in ['graph', 'all']:
        if not args.gt_pt:
            raise ValueError("--gt_pt is required for graph metrics")

        print("="*60)
        print("Graph-based Metrics (Key Graph)")
        print("="*60)

        gt_graphs = load_gt_key_graphs(args.gt_pt)
        gen_graphs, _ = load_generated_plys(args.gen_dir, args.tree_pattern)

        graph_results = compute_graph_metrics(gt_graphs, gen_graphs)
        results.update(graph_results)

    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)

    if 'JSD' in results:
        print(f"\nPoint-based (x1e3):")
        print(f"  JSD:  {results['JSD']:.3f}")
        print(f"  CD:   {results['CD']:.3f}")

    if 'Deg' in results:
        print(f"\nGraph-based:")
        print(f"  Deg:  {results['Deg']:.6f}")
        print(f"  Spec: {results['Spec']:.6f}")
        print(f"  GWD:  {results['GWD']:.6f}")

    print("="*60)

    print("\n[Reference Comparison - ImageCAS]")
    print("| Metric | Paper | Measured | Status |")
    print("|--------|-------|----------|--------|")
    if 'JSD' in results:
        status = "OK" if results['JSD'] < 100 else "X"
        print(f"| JSD    | 50.1  | {results['JSD']:.1f}    | {status} |")
        status = "OK" if results['CD'] < 50 else "X"
        print(f"| CD     | 24.4  | {results['CD']:.1f}    | {status} |")
    if 'Deg' in results:
        status = "OK" if results['Deg'] < 1.0 else "X"
        print(f"| Deg    | 0.601 | {results['Deg']:.3f}   | {status} |")
        status = "OK" if results['Spec'] < 0.5 else "X"
        print(f"| Spec   | 0.079 | {results['Spec']:.3f}   | {status} |")
        status = "OK" if results['GWD'] < 0.1 else "X"
        print(f"| GWD    | 0.029 | {results['GWD']:.3f}   | {status} |")

    if args.output:
        with open(args.output, 'w') as f:
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == '__main__':
    main()
