import torch

class DataRaterConfig:
    def __init__(self,
                 dataset_name,
                 inner_model_class,
                 data_rater_model_class,
                 batch_size=128,
                 train_split_ratio=0.9,
                 inner_lr=1e-3,
                 outer_lr=1e-3,
                 meta_steps=1000,
                 inner_steps=2,
                 meta_refresh_steps=10,
                 grad_clip_norm=1.0,
                 num_inner_models=4,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 loss_type="mse",
                 save_data_rater_checkpoint=False,
                 log=False,
                 attack_eps=8/255,
                 attack_steps=10,
                 attack_step_size=2/255):
        self.inner_model_class = inner_model_class
        self.data_rater_model_class = data_rater_model_class
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.train_split_ratio = train_split_ratio
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.meta_steps = meta_steps
        self.inner_steps = inner_steps
        self.meta_refresh_steps = meta_refresh_steps
        self.grad_clip_norm = grad_clip_norm
        self.num_inner_models = num_inner_models
        self.device = device
        self.loss_type = loss_type
        self.save_data_rater_checkpoint = save_data_rater_checkpoint
        self.log = log
        self.attack_eps = attack_eps
        self.attack_steps = attack_steps
        self.attack_step_size = attack_step_size