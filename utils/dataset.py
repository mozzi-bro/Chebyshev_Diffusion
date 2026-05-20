
import os

import torch
from torch.utils.data import Dataset


def coll_function(batch):
    first_item = batch[0]

    if len(first_item) == 6:
        trees, num_nodes, graphs, file_names, scale_infos, edge_radii_list = zip(*batch)
        return trees, num_nodes, graphs, file_names, scale_infos, edge_radii_list
    else:
        trees, num_nodes, graphs, file_names = zip(*batch)
        scale_infos = tuple(None for _ in trees)
        edge_radii_list = tuple(None for _ in trees)
        return trees, num_nodes, graphs, file_names, scale_infos, edge_radii_list


class TreeDataset(Dataset):
    
    def __init__(self, dataset_name, data_dir, is_train=True):
        if is_train is True:
            self.training = True
        else:
            self.training = False
        self.pt_file = os.path.join(data_dir, f"{dataset_name}.pt")

        self.data = []
        self.graphs = []
        self.file_names = []
        self.num_nodes = []
        self.max_nodes = 0

        self.scale_infos = []
        self.edge_radii = []

        if os.path.exists(self.pt_file):
            self.load_data()
        else:
            pass

        split_idx = int(len(self.data) * 0.9)

        self.train_data = self.data[:split_idx]
        self.test_data = self.data[split_idx:]
        self.train_graphs = self.graphs[:split_idx]
        self.test_graphs = self.graphs[split_idx:]
        self.train_file_names = self.file_names[:split_idx]
        self.test_file_names = self.file_names[split_idx:]
        self.train_num_nodes = self.num_nodes[:split_idx]
        self.test_num_nodes = self.num_nodes[split_idx:]
        
        self.train_scale_infos = self.scale_infos[:split_idx]
        self.test_scale_infos = self.scale_infos[split_idx:]
        self.train_edge_radii = self.edge_radii[:split_idx]
        self.test_edge_radii = self.edge_radii[split_idx:]

    def __len__(self):
        if self.training is True:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.training is True:
            num_nodes = torch.tensor(self.train_num_nodes[idx])

            scale_info = self.train_scale_infos[idx] if idx < len(self.train_scale_infos) else None
            edge_radii = self.train_edge_radii[idx] if idx < len(self.train_edge_radii) else None
            
            return (
                self.train_data[idx],
                num_nodes,
                self.train_graphs[idx],
                self.train_file_names[idx],
                scale_info,
                edge_radii
            )
        else:
            num_nodes = torch.tensor(self.test_num_nodes[idx])

            scale_info = self.test_scale_infos[idx] if idx < len(self.test_scale_infos) else None
            edge_radii = self.test_edge_radii[idx] if idx < len(self.test_edge_radii) else None
            
            return (
                self.test_data[idx],
                num_nodes,
                self.test_graphs[idx],
                self.test_file_names[idx],
                scale_info,
                edge_radii
            )

    def load_data(self):
        loaded_data = torch.load(self.pt_file, weights_only=False)
        
        self.data = loaded_data['data']
        self.graphs = loaded_data['graphs']
        self.num_nodes = loaded_data['num_nodes']
        self.file_names = loaded_data['file_names']
        
        if 'scale_infos' in loaded_data:
            self.scale_infos = loaded_data['scale_infos']
        else:
            self.scale_infos = [None] * len(self.data)

        if 'edge_radii' in loaded_data:
            self.edge_radii = loaded_data['edge_radii']
        else:
            self.edge_radii = [{}] * len(self.data)
