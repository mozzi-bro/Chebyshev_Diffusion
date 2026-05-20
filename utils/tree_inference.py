
import numpy as np
from utils.node import *
from utils import utils
from datetime import datetime
from utils.torch_f import Fold, encode_structure_fold
import torch
import os


def encode_testing(root, encoder, input_size=None):
    if input_size is None:
        input_size = root.radius.numel()
    
    def encode_node(node, encoder):
        node_attr = node.radius.reshape(-1, input_size)
        
        if node.right is None and node.left is None:
            return encoder.leafEncoder(node_attr)
        elif node.left is None and node.right is not None:
            right_feature = encode_node(node.right, encoder)
            return encoder.internalEncoder(node_attr, right_feature)
        elif node.right is None and node.left is not None:
            left_feature = encode_node(node.left, encoder)
            return encoder.internalEncoder(node_attr, left_feature)
        else:
            right_feature = encode_node(node.right, encoder)
            left_feature = encode_node(node.left, encoder)
            return encoder.bifurcationEncoder(node_attr, right_feature, left_feature)

    root_feature = encode_node(root, encoder)
    z = encoder.sampleEncoder(root_feature)

    return z


def decode_testing(vector, max, decoder, input_size=None):
    def decode_node(vector, max, decoder):

        cl = decoder.nodeClassifier(vector)
        _, label = torch.max(cl, 1)
        label = label.data

        if label.item() == 0 and create_node.count <= max:
            node = decoder.featureDecoder(vector)
            return create_node(create_node.count, node)

        elif label.item() == 1 and create_node.count <= max:
            right, node = decoder.internalDecoder(vector)
            d = create_node(create_node.count, node)
            d.right = decode_node(right, max, decoder)
            return d

        elif label.item() == 2 and create_node.count <= max:
            left, right, node = decoder.bifurcationDecoder(vector)
            d = create_node(create_node.count, node)
            d.right = decode_node(right, max, decoder)
            d.left = decode_node(left, max, decoder)
            return d

    create_node.count = 0
    vector = decoder.sample_decoder(vector)
    dec = decode_node(vector, max, decoder)

    return dec


def decode_testing_with_edge_radii(vector, max, decoder):
    decoder_latent = decoder.sample_decoder(vector)
    pred_scale = decoder.predict_scale(decoder_latent).item()
    
    node_latents = {}
    
    def decode_node_with_latent(vector, max, decoder, parent_id=None):
        cl = decoder.nodeClassifier(vector)
        _, label = torch.max(cl, 1)
        label = label.data

        if label.item() == 0 and create_node.count <= max:
            node = decoder.featureDecoder(vector)
            d = create_node(create_node.count, node)
            node_latents[d.data] = vector.clone()
            return d

        elif label.item() == 1 and create_node.count <= max:
            right, node = decoder.internalDecoder(vector)
            d = create_node(create_node.count, node)
            node_latents[d.data] = vector.clone()
            d.right = decode_node_with_latent(right, max, decoder, d.data)
            return d

        elif label.item() == 2 and create_node.count <= max:
            left, right, node = decoder.bifurcationDecoder(vector)
            d = create_node(create_node.count, node)
            node_latents[d.data] = vector.clone()
            d.right = decode_node_with_latent(right, max, decoder, d.data)
            d.left = decode_node_with_latent(left, max, decoder, d.data)
            return d
        
        return None

    create_node.count = 0
    decoded_vector = decoder.sample_decoder(vector)
    dec = decode_node_with_latent(decoded_vector, max, decoder)
    
    pred_edge_radii = {}
    
    def predict_edge_radii_recursive(node):
        if node is None:
            return
        
        parent_id = node.data
        
        if node.left is not None and parent_id in node_latents and node.left.data in node_latents:
            child_id = node.left.data
            parent_latent = node_latents[parent_id]
            child_latent = node_latents[child_id]
            pred_radius = decoder.predict_edge_radius(parent_latent, child_latent).item()
            pred_edge_radii[(parent_id, child_id)] = pred_radius
            predict_edge_radii_recursive(node.left)
        
        if node.right is not None and parent_id in node_latents and node.right.data in node_latents:
            child_id = node.right.data
            parent_latent = node_latents[parent_id]
            child_latent = node_latents[child_id]
            pred_radius = decoder.predict_edge_radius(parent_latent, child_latent).item()
            pred_edge_radii[(parent_id, child_id)] = pred_radius
            predict_edge_radii_recursive(node.right)
    
    if dec is not None:
        predict_edge_radii_recursive(dec)
    
    return dec, pred_scale, pred_edge_radii


def generate_samples(decoder, arg, num_samples=100, max_depth=50):
    decoder.eval()
    current_time = datetime.now().strftime('%m_%d_%H_%M')
    recon_dir = os.path.join(arg.output_dir, arg.dataset + '_' + current_time, "generation")
    os.makedirs(recon_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(num_samples):
            random_latent = torch.randn(1, arg.latent_size, device=arg.device)

            new_tree = decode_testing(vector=random_latent, max=max_depth, decoder=decoder)
            new_graph = new_tree.to_graph(dec=True)

            utils.graph_to_ply(new_graph, os.path.join(recon_dir, f'generated_sample_{i}.ply'))


def reconstruction(encoder, decoder, dataloader, arg, max_depth, mode):
    decoder.eval()
    encoder.eval()
    current_time = datetime.now().strftime('%m_%d_%H_%M')
    recon_dir = os.path.join(arg.output_dir, arg.dataset + '_' + current_time, f"recon_{mode}")
    ref_dir = os.path.join(arg.output_dir, arg.dataset + '_' + current_time, f"ref_{mode}")
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if len(batch_data) == 6:
                trees, num_nodes, gt_graphs, file_names, scale_infos, edge_radii_list = batch_data
            else:
                trees, num_nodes, gt_graphs, file_names = batch_data
                scale_infos = [None] * len(trees)
                edge_radii_list = [None] * len(trees)

            recon_graphs = []
            for i, (tree, gt_graph) in enumerate(zip(trees, gt_graphs)):
                test_enc_fold = Fold(arg.device)
                test_enc_fold_nodes = [encode_structure_fold(test_enc_fold, tree) for tree in trees]
                test_enc_fold_nodes = test_enc_fold.apply(encoder, [test_enc_fold_nodes])
                test_enc_fold_nodes = torch.split(test_enc_fold_nodes[0], 1, 0)

                for j, test_fold_node in enumerate(test_enc_fold_nodes):
                    test_root_code, _ = torch.chunk(test_fold_node, 2, 1)
                    recon_tree = decode_testing(vector=test_root_code, max=max_depth, decoder=decoder)
                    recon_graph = recon_tree.to_graph(dec=True)
                    recon_graphs.append(recon_graph)
                    base_name = os.path.splitext(file_names[i])[0]
                    utils.graph_to_ply(gt_graph, os.path.join(ref_dir, f"{base_name}_original.ply"))
                    utils.graph_to_ply(recon_graph, os.path.join(recon_dir, f"{base_name}_reconstructed.ply"))


if __name__ == '__main__':
    from config.tree_config import tree_args
    from torch.utils.data import DataLoader
    from utils.dataset import *
    from models.tree_vae import RecursiveEncoder, RecursiveDecoder

    tree_args = tree_args()
    
    tree_args.dataset = 'imagecas'
    tree_args.input_size = 10
    
    train_dataset = TreeDataset(tree_args.dataset, tree_args.data_path, is_train=True)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=tree_args.batch_size, 
        num_workers=0, 
        shuffle=True,
        collate_fn=coll_function
    )

    test_dataset = TreeDataset(tree_args.dataset, tree_args.data_path, is_train=False)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=tree_args.batch_size, 
        num_workers=0, 
        shuffle=True,
        collate_fn=coll_function
    )

    encoder = RecursiveEncoder(
        input_size=tree_args.input_size, 
        feature_size=tree_args.latent_size,
        hidden_size=tree_args.hidden_size
    ).to(tree_args.device)

    decoder = RecursiveDecoder(
        latent_size=tree_args.latent_size, 
        hidden_size=tree_args.hidden_size,
        output_size=tree_args.input_size, 
        args=tree_args
    ).to(tree_args.device)
    
    check_point_path = r"./logs/imagecas_02_14_12_12/models/19999.pth"
    check_point = torch.load(check_point_path, weights_only=False)
    encoder.load_state_dict(check_point['encoder'])
    decoder.load_state_dict(check_point['decoder'])

    generate_samples(decoder, tree_args, num_samples=100, max_depth=50)
    reconstruction(encoder, decoder, test_dataloader, tree_args, max_depth=50, mode='test')
    reconstruction(encoder, decoder, train_dataloader, tree_args, max_depth=50, mode='train')
