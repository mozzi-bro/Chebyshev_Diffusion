
import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import logging

from models.tree_vae import RecursiveEncoder, RecursiveDecoder
from utils.torch_f import Fold, encode_structure_fold, decode_structure_fold
from utils.utils import setup_logging, graph_to_ply
from utils.tree_inference import decode_testing
from evaluation.structure_evaluator import get_stats_eval


def check_nan_inf(tensor, name="tensor"):
    if tensor is None:
        return False
    
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan or has_inf:
        logging.warning(f"NaN or Inf found in {name}")
        return True
    return False


def get_kl_weight(epoch, warmup_epochs, kl_weight_start, kl_weight_target):
    if epoch < warmup_epochs:
        return kl_weight_start + (kl_weight_target - kl_weight_start) * (epoch / warmup_epochs)
    return kl_weight_target


def compute_edge_radius_loss_with_decoder_latent(decoder, root_code, tree, edge_radii, device):
    if edge_radii is None or len(edge_radii) == 0:
        return torch.tensor(0.0, device=device)
    
    node_latents = {}
    
    def decode_and_collect_latents(latent, node):
        if node is None:
            return
        
        node_latents[node.data] = latent.detach()
        
        if node.is_leaf():
            pass
        elif node.is_one_child():
            child_latent, _ = decoder.internalDecoder(latent)
            child_latent = child_latent.detach()
            child_node = node.right if node.right is not None else node.left
            decode_and_collect_latents(child_latent, child_node)
        else:
            left_latent, right_latent, _ = decoder.bifurcationDecoder(latent)
            left_latent = left_latent.detach()
            right_latent = right_latent.detach()
            decode_and_collect_latents(right_latent, node.right)
            decode_and_collect_latents(left_latent, node.left)
    
    decoder_root_latent = decoder.sample_decoder(root_code).detach()
    
    decode_and_collect_latents(decoder_root_latent, tree)
    
    total_loss = torch.tensor(0.0, device=device)
    edge_count = 0
    
    def compute_edge_losses(node, parent_idx=None):
        nonlocal total_loss, edge_count
        
        if node is None:
            return
        
        if parent_idx is not None and parent_idx in node_latents and node.data in node_latents:
            gt_radius = None
            for key in [(parent_idx, node.data), (node.data, parent_idx)]:
                if key in edge_radii:
                    gt_radius = edge_radii[key]
                    break
            
            if gt_radius is not None:
                parent_latent = node_latents[parent_idx]
                child_latent = node_latents[node.data]
                
                pred_radius = decoder.predict_edge_radius(parent_latent, child_latent)
                gt_radius_tensor = torch.tensor([[gt_radius]], device=device, dtype=torch.float32)
                
                total_loss = total_loss + F.mse_loss(pred_radius, gt_radius_tensor)
                edge_count += 1
        
        if node.right is not None:
            compute_edge_losses(node.right, node.data)
        if node.left is not None:
            compute_edge_losses(node.left, node.data)
    
    compute_edge_losses(tree)
    
    if edge_count > 0:
        return total_loss / edge_count
    return total_loss


def train_one_epoch(encoder, decoder, dataloader, optimizer, arg, current_kl_weight):
    encoder.train()
    decoder.train()

    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_scale_loss = 0.0
    total_edge_radius_loss = 0.0
    valid_batches = 0

    scale_weight = getattr(arg, 'scale_weight', 1.0)
    edge_radius_weight = getattr(arg, 'edge_radius_weight', 1.0)

    for batch_idx, batch_data in enumerate(dataloader):
        try:
            if len(batch_data) == 6:
                trees, num_nodes, graphs, file_names, scale_infos, edge_radii_list = batch_data
            else:
                trees, num_nodes, graphs, file_names = batch_data[:4]
                scale_infos = [None] * len(trees)
                edge_radii_list = [None] * len(trees)

            for tree in trees:
                tree.to_gpu()

            enc_fold = Fold(arg.device)
            enc_fold_nodes = [encode_structure_fold(enc_fold, tree) for tree in trees]
            enc_fold_nodes = enc_fold.apply(encoder, [enc_fold_nodes])
            
            if check_nan_inf(enc_fold_nodes[0], "encoder output"):
                logging.error(f"Batch {batch_idx}: Skipping due to NaN in encoder")
                continue
            
            enc_fold_nodes = torch.split(enc_fold_nodes[0], 1, 0)

            dec_fold = Fold(arg.device)
            dec_fold_nodes = []
            kld_fold_nodes = []
            root_codes = []

            for tree, fold_node in zip(trees, enc_fold_nodes):
                root_code, kl_div = torch.chunk(fold_node, 2, 1)
                dec_fold_nodes.append(decode_structure_fold(dec_fold, root_code, tree))
                kld_fold_nodes.append(kl_div)
                root_codes.append(root_code)

            total_loss = dec_fold.apply(decoder, [dec_fold_nodes, kld_fold_nodes])
            num_nodes_tensor = torch.as_tensor(num_nodes, device=arg.device)
            
            recon_loss = torch.div(total_loss[0], num_nodes_tensor).sum() / len(trees)
            
            kl_stacked = torch.stack(kld_fold_nodes)
            free_bits = getattr(arg, 'free_bits', 0.0)
            if free_bits > 0:
                kl_per_dim = kl_stacked.mean(dim=0)
                kl_loss = torch.clamp(kl_per_dim, min=free_bits).sum()
            else:
                kl_loss = kl_stacked.sum() / len(trees)
            
            scale_loss = torch.tensor(0.0, device=arg.device)
            if scale_weight > 0:
                scale_count = 0
                for root_code, scale_info in zip(root_codes, scale_infos):
                    if scale_info is not None and 'scale' in scale_info:
                        gt_scale = torch.tensor([[scale_info['scale']]], device=arg.device, dtype=torch.float32)
                        
                        decoder_root_latent = decoder.sample_decoder(root_code).detach()
                        pred_scale = decoder.predict_scale(decoder_root_latent)
                        
                        scale_loss = scale_loss + F.mse_loss(pred_scale, gt_scale)
                        scale_count += 1
                if scale_count > 0:
                    scale_loss = scale_loss / scale_count
            
            edge_radius_loss = torch.tensor(0.0, device=arg.device)
            if edge_radius_weight > 0:
                edge_loss_count = 0
                for root_code, tree, edge_radii in zip(root_codes, trees, edge_radii_list):
                    if edge_radii is not None and len(edge_radii) > 0:
                        edge_loss = compute_edge_radius_loss_with_decoder_latent(
                            decoder, root_code, tree, edge_radii, arg.device
                        )
                        if edge_loss.item() > 0:
                            edge_radius_loss = edge_radius_loss + edge_loss
                            edge_loss_count += 1
                if edge_loss_count > 0:
                    edge_radius_loss = edge_radius_loss / edge_loss_count
            
            loss = (recon_loss + 
                    current_kl_weight * kl_loss + 
                    scale_weight * scale_loss + 
                    edge_radius_weight * edge_radius_loss)
            
            if check_nan_inf(loss, "total loss"):
                logging.error(f"Batch {batch_idx}: Skipping due to NaN in loss")
                continue

            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 
                max_norm=1.0
            )
            
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_scale_loss += scale_loss.item()
            total_edge_radius_loss += edge_radius_loss.item()
            valid_batches += 1

        except Exception as e:
            logging.error(f"Batch {batch_idx}: Exception - {e}")
            continue

    if valid_batches == 0:
        logging.error("No valid batches in this epoch!")
        return float('nan'), float('nan'), float('nan'), float('nan')
    
    total_recon_loss /= valid_batches
    total_kl_loss /= valid_batches
    total_scale_loss /= valid_batches
    total_edge_radius_loss /= valid_batches

    return total_recon_loss, total_kl_loss, total_scale_loss, total_edge_radius_loss


def test(encoder, decoder, dataloader, arg, mode='Test'):
    encoder.eval()
    decoder.eval()
    all_results = []
    all_scale_errors = []
    all_edge_radius_errors = []

    with torch.no_grad():
        for batch_data in dataloader:
            if len(batch_data) == 6:
                trees, num_nodes, gt_graphs, file_names, scale_infos, edge_radii_list = batch_data
            else:
                trees, num_nodes, gt_graphs, file_names = batch_data
                scale_infos = [None] * len(trees)
                edge_radii_list = [None] * len(trees)
            
            for tree in trees:
                tree.to_gpu()

            enc_fold = Fold(arg.device)
            enc_fold_nodes = [encode_structure_fold(enc_fold, tree) for tree in trees]
            enc_fold_nodes = enc_fold.apply(encoder, [enc_fold_nodes])
            enc_fold_nodes = torch.split(enc_fold_nodes[0], 1, 0)

            gt_graphs_batch = []
            recon_graphs = []

            for i, (tree, fold_node) in enumerate(zip(trees, enc_fold_nodes)):
                root_code, kl_div = torch.chunk(fold_node, 2, 1)
                recon_tree = decode_testing(root_code, 50, decoder, arg.input_size)

                gt_graph = tree.to_graph(dec=False, input_size=arg.input_size)
                recon_graph = recon_tree.to_graph(dec=True, input_size=arg.input_size)

                gt_graphs_batch.append(gt_graph)
                recon_graphs.append(recon_graph)

                if scale_infos[i] is not None and 'scale' in scale_infos[i]:
                    decoder_root_latent = decoder.sample_decoder(root_code)
                    pred_scale = decoder.predict_scale(decoder_root_latent)
                    gt_scale = scale_infos[i]['scale']
                    scale_error = abs(pred_scale.item() - gt_scale)
                    all_scale_errors.append(scale_error)

            try:
                stats_eval_fn = get_stats_eval(arg)
                stats_results = stats_eval_fn(gt_graphs_batch, recon_graphs)
                all_results.append(stats_results)
            except Exception as e:
                logging.warning(f"Evaluation failed: {e}")

    if len(all_results) > 0:
        eval_results = {key: np.mean([result[key] for result in all_results]) for key in all_results[0].keys()}
    else:
        eval_results = {'chamfer_distance': float('inf')}
    
    if all_scale_errors:
        eval_results['scale_mae'] = np.mean(all_scale_errors)
    if all_edge_radius_errors:
        eval_results['edge_radius_mae'] = np.mean(all_edge_radius_errors)
    
    logging.info(f"{mode} | " + " | ".join(
        [f"{key}: {value:.4f}" for key, value in eval_results.items()]
    ))

    return eval_results


def train_model(encoder, decoder, train_data, test_data, optimizer, arg):
    best_cd = float('inf')
    best_gwd = float('inf')
    
    patience = getattr(arg, 'patience', 10)
    early_stopping = getattr(arg, 'early_stopping', True)
    patience_counter = 0
    
    writer, arg.log_dir = setup_logging(arg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=arg.lr_step_size, gamma=arg.lr_gamma)
    
    kl_warmup_epochs = getattr(arg, 'kl_warmup_epochs', 500)
    kl_weight_start = getattr(arg, 'kl_weight_start', 0.0)
    kl_weight_target = arg.kl_weight
    
    logging.info("=" * 60)
    logging.info("Tree VAE Training (Decoder Latent Consistency)")
    logging.info("=" * 60)
    logging.info(f"Scale/EdgeRadius: Using decoder latent (consistent with inference)")
    logging.info(f"KL Annealing: warmup={kl_warmup_epochs} epochs, "
                f"start={kl_weight_start}, target={kl_weight_target}")
    logging.info(f"Note: kl_weight={kl_weight_target} is already very low (beta-VAE style)")
    logging.info(f"Early Stopping: {'ON' if early_stopping else 'OFF'}, patience={patience}")
    
    print(f"\nScale/EdgeRadius: Decoder Latent training", flush=True)
    print(f"KL Annealing: {kl_warmup_epochs} epochs warmup -> target {kl_weight_target}", flush=True)
    print(f"Early Stopping: {'ON' if early_stopping else 'OFF'} (patience={patience})", flush=True)
    print(f"Training for {arg.epochs} epochs...\n", flush=True)
    
    for epoch in range(arg.epochs):
        start_time = time.time()
        
        current_kl_weight = get_kl_weight(
            epoch, kl_warmup_epochs, kl_weight_start, kl_weight_target
        )

        total_recon_loss, total_kl_loss, total_scale_loss, total_edge_radius_loss = \
            train_one_epoch(encoder, decoder, train_data, optimizer, arg, current_kl_weight)
        
        if np.isnan(total_recon_loss) or np.isnan(total_kl_loss):
            logging.error(f"Epoch [{epoch + 1}]: NaN detected! Saving emergency checkpoint.")
            torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},
                       os.path.join(arg.log_dir, 'models', f'emergency_epoch_{epoch}.pth'))
            continue
        
        writer.add_scalar('Train/Recon_Loss', total_recon_loss, epoch)
        writer.add_scalar('Train/KL_Loss', total_kl_loss, epoch)
        writer.add_scalar('Train/Scale_Loss', total_scale_loss, epoch)
        writer.add_scalar('Train/EdgeRadius_Loss', total_edge_radius_loss, epoch)
        writer.add_scalar('Train/KL_Weight', current_kl_weight, epoch)
        writer.add_scalar('Train/Total_Loss', 
                         total_recon_loss + current_kl_weight * total_kl_loss + 
                         getattr(arg, 'scale_weight', 1.0) * total_scale_loss +
                         getattr(arg, 'edge_radius_weight', 1.0) * total_edge_radius_loss, 
                         epoch)

        if (epoch + 1) % arg.checkpoint == 0:
            eval_results_test = test(encoder, decoder, test_data, arg, mode='Test')
            eval_results_train = test(encoder, decoder, train_data, arg, mode='Train')

            for key, value in eval_results_test.items():
                writer.add_scalar(f'Test/{key}', value, epoch)
            for key, value in eval_results_train.items():
                writer.add_scalar(f'Train/{key}', value, epoch)

            epoch_cd = eval_results_test.get('chamfer_distance', float('inf'))
            epoch_gwd = eval_results_test.get('gwd', float('inf'))
            
            improved = False
            
            if epoch_cd < best_cd:
                best_cd = epoch_cd
                torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},
                           os.path.join(arg.log_dir, 'models', 'best_model_cd.pth'))
                logging.info(f'[Best CD] New best model saved with Chamfer Distance: {best_cd:.4f}')
                print(f"  * New Best CD: {best_cd:.4f}", flush=True)
                improved = True
            
            if epoch_gwd < best_gwd:
                best_gwd = epoch_gwd
                torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},
                           os.path.join(arg.log_dir, 'models', 'best_model_gwd.pth'))
                logging.info(f'[Best GWD] New best model saved with GWD: {best_gwd:.4f}')
                print(f"  * New Best GWD: {best_gwd:.4f}", flush=True)
            
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                logging.info(f'No improvement. Patience: {patience_counter}/{patience}')
            
            if early_stopping and patience_counter >= patience:
                print(f"\n{'='*60}", flush=True)
                print(f"[Early Stopping] No improvement for {patience} checkpoints.", flush=True)
                print(f"Best CD: {best_cd:.4f}, Best GWD: {best_gwd:.4f}", flush=True)
                print(f"Stopping at epoch {epoch + 1}", flush=True)
                print(f"{'='*60}\n", flush=True)
                logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

            torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},
                       os.path.join(arg.log_dir, 'models', f'{epoch}.pth'))
        
        scheduler.step()
        end_time = time.time()
        
        if epoch < 10 or (epoch + 1) % 100 == 0 or (epoch + 1) % arg.checkpoint == 0:
            print(
                f"Epoch [{epoch + 1:5d}/{arg.epochs}] {end_time - start_time:.1f}s | "
                f"Recon: {total_recon_loss:.4f} | KL: {total_kl_loss:.2f} | "
                f"Scale: {total_scale_loss:.5f} | EdgeR: {total_edge_radius_loss:.4f}",
                flush=True
            )
        
        if (epoch + 1) % 10 == 0 or epoch < 10:
            logging.info(
                f"Epoch [{epoch + 1}/{arg.epochs}] | Time: {end_time - start_time:.2f}s | "
                f"KL_w: {current_kl_weight:.6f} | Recon: {total_recon_loss:.4f} | "
                f"KL: {total_kl_loss:.4f} | Scale: {total_scale_loss:.4f} | "
                f"EdgeR: {total_edge_radius_loss:.4f}"
            )

    print(f"\n{'='*60}", flush=True)
    print(f"Training Complete!", flush=True)
    print(f"  Best CD:  {best_cd:.4f} -> best_model_cd.pth", flush=True)
    print(f"  Best GWD: {best_gwd:.4f} -> best_model_gwd.pth", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    logging.info(f"Training complete. Best CD: {best_cd:.4f}, Best GWD: {best_gwd:.4f}")

    return encoder, decoder


def coll_function(batch):
    trees = [item[0] for item in batch]
    num_nodes = [item[1] for item in batch]
    graphs = [item[2] for item in batch]
    file_names = [item[3] for item in batch]
    
    scale_infos = [item[4] if len(item) > 4 else None for item in batch]
    edge_radii = [item[5] if len(item) > 5 else None for item in batch]
    
    return trees, num_nodes, graphs, file_names, scale_infos, edge_radii


if __name__ == '__main__':
    print("=" * 60, flush=True)
    print("Tree VAE Training", flush=True)
    print("Decoder Latent Consistency (Scale/EdgeRadius)", flush=True)
    print("=" * 60, flush=True)
    
    from config.tree_config import tree_args
    from torch.utils.data import DataLoader
    from utils.dataset import *
    from utils.utils import set_seed

    args = tree_args()
    set_seed(args.seed)
    
    if not hasattr(args, 'scale_weight'):
        args.scale_weight = 1.0
    if not hasattr(args, 'edge_radius_weight'):
        args.edge_radius_weight = 1.0
    if not hasattr(args, 'kl_warmup_epochs'):
        args.kl_warmup_epochs = 0
    if not hasattr(args, 'kl_weight_start'):
        args.kl_weight_start = args.kl_weight
    
    train_dataset = TreeDataset(args.dataset, args.data_path, is_train=True)
    test_dataset = TreeDataset(args.dataset, args.data_path, is_train=False)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        collate_fn=coll_function
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        collate_fn=coll_function
    )

    Encoder = RecursiveEncoder(
        input_size=args.input_size, 
        feature_size=args.latent_size,
        hidden_size=args.hidden_size
    ).to(args.device)
    
    Decoder = RecursiveDecoder(
        latent_size=args.latent_size, 
        hidden_size=args.hidden_size,
        output_size=args.input_size,
        args=args
    ).to(args.device)
    
    total_params = sum(p.numel() for p in Encoder.parameters()) + sum(p.numel() for p in Decoder.parameters())
    
    print(f"Config: dataset={args.dataset}, batch={args.batch_size}, lr={args.lr}, device={args.device}", flush=True)
    print(f"Data: {len(train_dataset)} train, {len(test_dataset)} test | Model: {total_params/1e6:.1f}M params", flush=True)

    opt = torch.optim.Adam(
        list(Encoder.parameters()) + list(Decoder.parameters()), 
        lr=args.lr
    )
    
    logging.info(f"Arguments: {args}")
    logging.info(f"Encoder params: {sum(p.numel() for p in Encoder.parameters()):,}")
    logging.info(f"Decoder params: {sum(p.numel() for p in Decoder.parameters()):,}")

    train_model(
        encoder=Encoder, 
        decoder=Decoder, 
        train_data=train_dataloader, 
        test_data=test_dataloader,
        optimizer=opt, 
        arg=args
    )
