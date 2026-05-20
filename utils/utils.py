import torch
from tensorboardX import SummaryWriter
import logging
import random
import os
import numpy as np
import networkx as nx
from datetime import datetime
from plyfile import PlyData, PlyElement


def setup_logging(args):
    current_time = datetime.now().strftime('%m_%d_%H_%M')
    model_name = getattr(args, 'model', 'model')
    dataset_name = getattr(args, 'dataset', 'dataset')
    log_dir = os.path.join(args.log_dir, model_name + '_' + dataset_name + '_' + current_time)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    models_dir = os.path.join(log_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    writer = SummaryWriter(log_dir)
    logging.info(str(args))

    return writer, log_dir


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ply_to_graph(filename):
    vertices, edges = read_ply(filename)
    graph = nx.Graph()
    for i, vertex in enumerate(vertices):
        position = list(vertex)
        graph.add_node(i, position=position)

    for edge in edges:
        source = edge[0]
        target = edge[1]
        graph.add_edge(source, target)

    return graph


def graph_to_ply(graph, filename):
    nodes = list(graph.nodes())
    edges = list(graph.edges())

    vertex_coordinates = np.array([graph.nodes[n]['position'] for n in nodes]).squeeze()

    if vertex_coordinates.shape[-1] == 3:
        vertex = np.array([
            (vertex_coordinates[i, 0], vertex_coordinates[i, 1], vertex_coordinates[i, 2])
            for i in range(vertex_coordinates.shape[0])
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    elif vertex_coordinates.shape[-1] == 4:
        vertex = np.array([
            (vertex_coordinates[i, 0], vertex_coordinates[i, 1], vertex_coordinates[i, 2], vertex_coordinates[i, 3])
            for i in range(vertex_coordinates.shape[0])
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('r', 'f4')])

    elif vertex_coordinates.shape[-1] == 6:
        vertex = np.array([
            (vertex_coordinates[i, 0], vertex_coordinates[i, 1], vertex_coordinates[i, 2],
             vertex_coordinates[i, 3], vertex_coordinates[i, 4], vertex_coordinates[i, 5]) for i in
            range(vertex_coordinates.shape[0])
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    elif vertex_coordinates.shape[-1] == 9:
        vertex = np.array([
            (vertex_coordinates[i, 0], vertex_coordinates[i, 1], vertex_coordinates[i, 2],
             vertex_coordinates[i, 3], vertex_coordinates[i, 4], vertex_coordinates[i, 5],
             vertex_coordinates[i, 6], vertex_coordinates[i, 7], vertex_coordinates[i, 8]) for i in
            range(vertex_coordinates.shape[0])
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                  ('l', 'f4'), ('d', 'f4'), ('c', 'f4'),
                  ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    elif vertex_coordinates.shape[-1] == 10:
        vertex = np.array([
            (vertex_coordinates[i, 0], vertex_coordinates[i, 1], vertex_coordinates[i, 2],
             vertex_coordinates[i, 3], vertex_coordinates[i, 4], vertex_coordinates[i, 5],
             vertex_coordinates[i, 6], vertex_coordinates[i, 7], vertex_coordinates[i, 8], vertex_coordinates[i, 9]) for
            i in
            range(vertex_coordinates.shape[0])
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('r', 'f4'),
                  ('l', 'f4'), ('d', 'f4'), ('c', 'f4'),
                  ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    edge_indices = np.array([
        (nodes.index(u), nodes.index(v)) for u, v in edges
    ], dtype=[('vertex1', 'i4'), ('vertex2', 'i4')])

    vertex_element = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    edge_element = PlyElement.describe(edge_indices, 'edge', comments=['edge indices'])

    PlyData([vertex_element, edge_element], text=True).write(filename)


def read_ply(filename):
    data = PlyData.read(filename)
    vertex_data = data['vertex']

    properties = vertex_data.properties

    vertices = np.array([
        tuple(vertex[prop.name] for prop in properties) for vertex in vertex_data
    ])
    edges = data['edge']

    return vertices, edges


def read_pcd(filename):
    data = PlyData.read(filename)
    vertex_data = data['vertex']

    properties = vertex_data.properties

    vertices = np.array([
        tuple(vertex[prop.name] for prop in properties) for vertex in vertex_data
    ])

    return vertices
