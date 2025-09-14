import argparse
import torch
from data_rater import run_meta_training
from config import DataRaterConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description='Data Rater Meta-Learning Training')

    # Overall parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--train_split_ratio', type=float, default=0.9,
                        help='Train split ratio (default: 0.9)')
    parser.add_argument('--inner_lr', type=float, default=1e-3,
                        help='Inner loop learning rate (default: 1e-3)')
    parser.add_argument('--outer_lr', type=float, default=1e-3,
                        help='Outer loop learning rate (default: 1e-3)')
    parser.add_argument('--meta_steps', type=int, default=1000,
                        help='Number of meta-training steps (default: 1000)')
    parser.add_argument('--inner_steps', type=int, default=2,
                        help='Number of inner loop steps (default: 2)')
    parser.add_argument('--meta_refresh_steps', type=int, default=10,
                        help='Meta refresh steps (default: 10)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                        help='Gradient clipping norm (default: 1.0)')
    parser.add_argument('--num_inner_models', type=int, default=4,
                        help='Number of inner models (default: 4)')

    # Dataset and model parameters
    parser.add_argument('--dataset_name', type=str, required=True, default='mnist',
                        help='Name of the dataset to use')
    parser.add_argument('--inner_model_name', type=str, default='ToyCNN',
                        help='Inner model class to use (default: ToyCNN)')
    parser.add_argument('--data_rater_model_name', type=str, default='DataRater',
                        help='Data rater model class to use (default: DataRater)')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device to use for training (default: cuda if available, else cpu)')

    parser.add_argument('--save_data_rater_checkpoint', type=bool, default=False,
                        help='Whether to save the data rater model checkpoint (default: False)')
    parser.add_argument('--log', type=bool, default=False,
                        help='Whether to log the training (default: False)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = DataRaterConfig(
        dataset_name=args.dataset_name,
        inner_model_class=args.inner_model_name,
        data_rater_model_class=args.data_rater_model_name,
        batch_size=args.batch_size,
        train_split_ratio=args.train_split_ratio,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        meta_steps=args.meta_steps,
        inner_steps=args.inner_steps,
        meta_refresh_steps=args.meta_refresh_steps,
        grad_clip_norm=args.grad_clip_norm,
        num_inner_models=args.num_inner_models,
        device=args.device,
        save_data_rater_checkpoint=args.save_data_rater_checkpoint,
        log=args.log,
    )
    run_meta_training(config)
