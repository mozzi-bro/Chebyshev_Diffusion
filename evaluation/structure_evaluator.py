
import numpy as np
import networkx as nx
import torch
from scipy.linalg import eigvalsh
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import List, Optional, Tuple, Dict


def degree_distribution(graph: nx.Graph) -> np.ndarray:
    degrees = [d for n, d in graph.degree()]
    if len(degrees) == 0:
        return np.array([0])
    
    max_degree = max(degrees) + 1
    hist, _ = np.histogram(degrees, bins=range(max_degree + 1), density=True)
    return hist


def laplacian_spectrum(graph: nx.Graph, k: Optional[int] = None) -> np.ndarray:
    if graph.number_of_nodes() == 0:
        return np.array([0])
    
    laplacian = nx.normalized_laplacian_matrix(graph).toarray()
    
    eigenvalues = eigvalsh(laplacian)
    eigenvalues = np.sort(eigenvalues)
    
    if k is not None:
        eigenvalues = eigenvalues[:k]
    
    return eigenvalues


def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))


def compute_mmd(samples1: List[np.ndarray], samples2: List[np.ndarray], 
                kernel: str = 'rbf', sigma: float = 1.0) -> float:
    if len(samples1) == 0 or len(samples2) == 0:
        return 0.0
    
    max_len = max(
        max(len(s) for s in samples1),
        max(len(s) for s in samples2)
    )
    
    def pad_sample(s, length):
        if len(s) >= length:
            return s[:length]
        return np.pad(s, (0, length - len(s)), mode='constant', constant_values=0)
    
    samples1_padded = np.array([pad_sample(s, max_len) for s in samples1])
    samples2_padded = np.array([pad_sample(s, max_len) for s in samples2])
    
    n1, n2 = len(samples1_padded), len(samples2_padded)
    
    if kernel == 'rbf':
        k11 = 0.0
        for i in range(n1):
            for j in range(n1):
                k11 += gaussian_kernel(samples1_padded[i], samples1_padded[j], sigma)
        k11 /= (n1 * n1)
        
        k22 = 0.0
        for i in range(n2):
            for j in range(n2):
                k22 += gaussian_kernel(samples2_padded[i], samples2_padded[j], sigma)
        k22 /= (n2 * n2)
        
        k12 = 0.0
        for i in range(n1):
            for j in range(n2):
                k12 += gaussian_kernel(samples1_padded[i], samples2_padded[j], sigma)
        k12 /= (n1 * n2)
        
        mmd = k11 + k22 - 2 * k12
        
    else:
        mean1 = np.mean(samples1_padded, axis=0)
        mean2 = np.mean(samples2_padded, axis=0)
        mmd = np.sum((mean1 - mean2) ** 2)
    
    return max(0, mmd)


def degree_mmd(graphs1: List[nx.Graph], graphs2: List[nx.Graph], sigma: float = 1.0) -> float:
    deg_dist1 = [degree_distribution(g) for g in graphs1]
    deg_dist2 = [degree_distribution(g) for g in graphs2]
    
    return compute_mmd(deg_dist1, deg_dist2, kernel='rbf', sigma=sigma)


def spectral_mmd(graphs1: List[nx.Graph], graphs2: List[nx.Graph], 
                 sigma: float = 1.0, k: Optional[int] = None) -> float:
    spec1 = [laplacian_spectrum(g, k) for g in graphs1]
    spec2 = [laplacian_spectrum(g, k) for g in graphs2]
    
    return compute_mmd(spec1, spec2, kernel='rbf', sigma=sigma)


def sample_points_on_edges(graph: nx.Graph, num_samples_per_edge: int = 10) -> np.ndarray:
    points = []
    
    for u, v in graph.edges():
        pos_u = graph.nodes[u].get('position', [0, 0, 0])
        pos_v = graph.nodes[v].get('position', [0, 0, 0])
        
        pos_u = np.asarray(pos_u).flatten()[:3]
        pos_v = np.asarray(pos_v).flatten()[:3]
        
        if len(pos_u) < 3:
            pos_u = np.pad(pos_u, (0, 3 - len(pos_u)))
        if len(pos_v) < 3:
            pos_v = np.pad(pos_v, (0, 3 - len(pos_v)))
        
        for t in np.linspace(0, 1, num_samples_per_edge):
            point = pos_u + t * (pos_v - pos_u)
            points.append(point)
    
    for node in graph.nodes():
        pos = graph.nodes[node].get('position', [0, 0, 0])
        pos = np.asarray(pos).flatten()[:3]
        if len(pos) < 3:
            pos = np.pad(pos, (0, 3 - len(pos)))
        points.append(pos)
    
    if len(points) == 0:
        return np.zeros((1, 3))
    
    return np.array(points)


def graph_to_point_cloud(graph: nx.Graph, use_edge_sampling: bool = True,
                         num_samples_per_edge: int = 10) -> np.ndarray:
    if use_edge_sampling and graph.number_of_edges() > 0:
        return sample_points_on_edges(graph, num_samples_per_edge)
    else:
        points = []
        for node in graph.nodes():
            pos = graph.nodes[node].get('position', [0, 0, 0])
            if isinstance(pos, (list, tuple, np.ndarray)):
                pos = np.array(pos).flatten()[:3]
            else:
                pos = np.array([0, 0, 0])
            if len(pos) < 3:
                pos = np.pad(pos, (0, 3 - len(pos)))
            points.append(pos)
        
        if len(points) == 0:
            return np.zeros((1, 3))
        
        return np.array(points)


def sinkhorn_distance(x: np.ndarray, y: np.ndarray, 
                      epsilon: float = 0.1, max_iters: int = 100,
                      threshold: float = 1e-6) -> float:
    n, m = len(x), len(y)
    
    if n == 0 or m == 0:
        return 0.0
    
    C = cdist(x, y, metric='sqeuclidean')
    
    K = np.exp(-C / epsilon)
    
    a = np.ones(n) / n
    b = np.ones(m) / m
    
    u = np.ones(n)
    v = np.ones(m)
    
    for _ in range(max_iters):
        u_prev = u.copy()
        
        u = a / (K @ v + 1e-10)
        v = b / (K.T @ u + 1e-10)
        
        if np.max(np.abs(u - u_prev)) < threshold:
            break
    
    P = np.diag(u) @ K @ np.diag(v)
    
    distance = np.sum(P * C)
    
    return np.sqrt(max(0, distance))


def graph_wasserstein_distance(graph1: nx.Graph, graph2: nx.Graph,
                               use_sinkhorn: bool = True,
                               use_edge_sampling: bool = True,
                               num_samples_per_edge: int = 10,
                               epsilon: float = 0.1) -> float:
    pc1 = graph_to_point_cloud(graph1, use_edge_sampling, num_samples_per_edge)
    pc2 = graph_to_point_cloud(graph2, use_edge_sampling, num_samples_per_edge)
    
    n1, n2 = len(pc1), len(pc2)
    
    if n1 == 0 or n2 == 0:
        return 0.0
    
    if use_sinkhorn:
        return sinkhorn_distance(pc1, pc2, epsilon=epsilon)
    else:
        cost_matrix = cdist(pc1, pc2, metric='euclidean')
        
        if n1 != n2:
            max_n = max(n1, n2)
            padded_cost = np.full((max_n, max_n), fill_value=np.max(cost_matrix) * 2)
            padded_cost[:n1, :n2] = cost_matrix
            cost_matrix = padded_cost
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        gwd = cost_matrix[row_ind, col_ind].mean()
        
        return gwd


def batch_graph_wasserstein_distance(graphs1: List[nx.Graph], graphs2: List[nx.Graph],
                                     use_sinkhorn: bool = True,
                                     use_edge_sampling: bool = True) -> float:
    n = min(len(graphs1), len(graphs2))
    if n == 0:
        return 0.0
    
    total_gwd = 0.0
    for i in range(n):
        total_gwd += graph_wasserstein_distance(
            graphs1[i], graphs2[i], 
            use_sinkhorn=use_sinkhorn,
            use_edge_sampling=use_edge_sampling
        )
    
    return total_gwd / n


def chamfer_distance_graphs(graphs1: List[nx.Graph], graphs2: List[nx.Graph],
                           use_edge_sampling: bool = True) -> float:
    n = min(len(graphs1), len(graphs2))
    if n == 0:
        return 0.0
    
    total_cd = 0.0
    for i in range(n):
        pc1 = graph_to_point_cloud(graphs1[i], use_edge_sampling)
        pc2 = graph_to_point_cloud(graphs2[i], use_edge_sampling)
        
        dist_matrix = cdist(pc1, pc2, metric='euclidean')
        
        cd1 = np.mean(np.min(dist_matrix, axis=1))
        cd2 = np.mean(np.min(dist_matrix, axis=0))
        
        total_cd += (cd1 + cd2) / 2
    
    return total_cd / n


def extract_largest_component(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph
    
    if nx.is_connected(graph):
        return graph
    
    components = list(nx.connected_components(graph))
    largest_cc = max(components, key=len)
    return graph.subgraph(largest_cc).copy()


def evaluate_graphs(gt_graphs: List[nx.Graph], pred_graphs: List[nx.Graph],
                   use_sinkhorn: bool = True, 
                   use_edge_sampling: bool = True,
                   extract_largest: bool = True) -> Dict[str, float]:
    if extract_largest:
        gt_graphs = [extract_largest_component(g) for g in gt_graphs]
        pred_graphs = [extract_largest_component(g) for g in pred_graphs]
    
    gt_graphs = [g for g in gt_graphs if g.number_of_nodes() > 0]
    pred_graphs = [g for g in pred_graphs if g.number_of_nodes() > 0]
    
    if len(gt_graphs) == 0 or len(pred_graphs) == 0:
        return {
            'Deg': 0.0,
            'Spec': 0.0,
            'GWD': 0.0,
            'CD': 0.0
        }
    
    deg = degree_mmd(gt_graphs, pred_graphs)
    spec = spectral_mmd(gt_graphs, pred_graphs)
    gwd = batch_graph_wasserstein_distance(gt_graphs, pred_graphs, 
                                           use_sinkhorn=use_sinkhorn,
                                           use_edge_sampling=use_edge_sampling)
    cd = chamfer_distance_graphs(gt_graphs, pred_graphs, use_edge_sampling=use_edge_sampling)
    
    return {
        'Deg': deg,
        'Spec': spec,
        'GWD': gwd,
        'CD': cd
    }


def get_stats_eval(args):
    use_sinkhorn = True
    use_edge_sampling = True
    extract_largest = getattr(args, 'max_subgraph', True)

    def evaluate(gt_graphs, pred_graphs):
        results = evaluate_graphs(
            gt_graphs, pred_graphs,
            use_sinkhorn=use_sinkhorn,
            use_edge_sampling=use_edge_sampling,
            extract_largest=extract_largest
        )

        return {
            'degree_mmd': results['Deg'],
            'spectral_mmd': results['Spec'],
            'gwd': results['GWD'],
            'chamfer_distance': results['CD']
        }

    return evaluate
