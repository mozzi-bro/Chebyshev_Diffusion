
import argparse


def tree_args():
    parser = argparse.ArgumentParser(description="TREE Model Arguments")

    parser.add_argument('--model', type=str, default='tree')
    parser.add_argument('--data_path', type=str, default=r'./data/')
    parser.add_argument('--dataset', type=str, default='')

    parser.add_argument('--input_size', default=10, type=int,
                        help='Node feature dimension (10D: xyz + radius + geometry + direction)')

    parser.add_argument('--latent_size', type=int, default=512,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_size', type=int, default=1024,
                        help='Hidden layer dimension')

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', type=int, default=40000)
    parser.add_argument('--checkpoint', default=2000, type=int,
                        help='Evaluation interval (epochs)')

    parser.add_argument('--lr', default=0.0001, type=float,
                        help='Learning rate')

    parser.add_argument('--early_stopping', default=True, type=bool,
                        help='Enable early stopping')
    parser.add_argument('--patience', default=10, type=int,
                        help='Early stopping patience (number of checkpoints without improvement)')
    parser.add_argument('--lr_step_size', default=100, type=float,
                        help='LR scheduler step size')
    parser.add_argument('--lr_gamma', default=1, type=float,
                        help='LR scheduler gamma')

    parser.add_argument('--kl_weight', default=0.001, type=float,
                        help='KL divergence loss weight')

    parser.add_argument('--scale_weight', default=1.0, type=float,
                        help='Scale prediction loss weight (root_latent -> scale)')

    parser.add_argument('--edge_radius_weight', default=1.0, type=float,
                        help='Edge radius prediction loss weight')

    parser.add_argument('--use_edge_radius_loss', default=True, type=bool,
                        help='Whether to use edge radius prediction loss')

    parser.add_argument('--use_hint_nodes', default=True, type=bool,
                        help='Whether to use hint nodes in training data')
    parser.add_argument('--hint_deviation_threshold', default=2.0, type=float,
                        help='Max deviation threshold (mm) for hint node insertion')

    parser.add_argument('--log_dir', type=str, default=r'./logs/')
    parser.add_argument('--output_dir', type=str, default=r'./output/')

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2001)

    parser.add_argument('--mmd_distance', type=str, default='rbf',
                        help='MMD distance kernel type')
    parser.add_argument('--max_subgraph', type=bool, default=True,
                        help='Use max subgraph for evaluation')

    args, _ = parser.parse_known_args()
    return args
