
import networkx as nx
import numpy as np


def count_fn(f):
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        return f(*args, **kwargs)

    wrapper.count = 0
    return wrapper


@count_fn
def create_node(data, radius, left=None, right=None, node_type='endpoint'):
    return Node(data, radius, left, right, node_type=node_type)


class Node:
    
    def __init__(self, value, radius, left=None, right=None, 
                 level=None, treelevel=None, maxlevel=None,
                 node_type='endpoint'):
        self.left = left
        self.data = value
        self.radius = radius
        self.right = right
        self.level = level

        self.treelevel = treelevel
        self.maxlevel = maxlevel
        self.node_type = node_type


    def to_gpu(self):
        self.radius = self.radius.cuda()
        if self.left:
            self.left.to_gpu()
        if self.right:
            self.right.to_gpu()
    

    def is_leaf(self):
        return self.left is None and self.right is None

    def is_one_child(self):
        return (self.left is None) != (self.right is None)

    def child_num(self):
        return sum(child is not None for child in (self.left, self.right))

    def to_graph(self, dec=False, input_size=None):
        graph = nx.Graph()
        self.add_node(graph, dec)
        return graph

    def add_node(self, graph, dec=False):
        radius = self.radius.cpu().detach().numpy()

        if dec:
            radius = np.squeeze(radius)

        radius = np.asarray(radius).flatten()
        position = radius[:3].copy() if len(radius) >= 3 else radius.copy()
        
        graph.add_node(
            self.data,
            position=position,
            radius=radius,
            node_type=self.node_type
        )

        if self.left:
            graph.add_edge(self.data, self.left.data)
            self.left.add_node(graph, dec)

        if self.right:
            graph.add_edge(self.data, self.right.data)
            self.right.add_node(graph, dec)

    def get_node_count(self):
        count = 1
        if self.left:
            count += self.left.get_node_count()
        if self.right:
            count += self.right.get_node_count()
        return count
    

    def __repr__(self):
        pos_shape = self.radius.shape if hasattr(self.radius, 'shape') else 'N/A'
        children = []
        if self.left:
            children.append('L')
        if self.right:
            children.append('R')
        children_str = ','.join(children) if children else 'leaf'
        return f"Node(id={self.data}, type={self.node_type}, shape={pos_shape}, children=[{children_str}])"
