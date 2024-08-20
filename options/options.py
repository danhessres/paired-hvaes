import argparse

def generate_seed():
    import random
    return random.randrange(0xffff_ffff)

class GlobalOptions:
    def __init__(self, *args, **kwargs):
        self.parser = argparse.ArgumentParser(*args, **kwargs)
        self.is_ready = False
    def setup(self):
        self.parser.add_argument('--argfile', help = 'Load arguments from json file. CLAs do NOT overwrite these. Model name and data-directory still required.')
        self.parser.add_argument('--model_name', required = True)
        self.parser.add_argument('--cuda', action = 'store_true')
        self.parser.add_argument('--checkpoint_dir', default = 'artifacts')
        self.parser.add_argument('--random_state', default = generate_seed(), type = int)
        self.parser.add_argument('--data_dir', required = True)
        self.parser.add_argument('--load_all', action = 'store_true')
        self.is_ready = True
    def parse(self):
        if not self.is_ready:
            self.setup()
        return self.parser.parse_args()

class ArchOptions(GlobalOptions):
    def setup(self):
        super().setup()
        self.parser.add_argument('--arch_file')
        self.parser.add_argument('--channels', type = int)
        self.parser.add_argument('--altchannels', type = int)
        self.parser.add_argument('--img_res', type = int)
        self.parser.add_argument('--width', type = int)
        self.parser.add_argument('--depth', type = int)
        self.parser.add_argument('--layers_per_group', type = int)
        self.parser.add_argument('--latent_per_group', type = int)
        self.parser.add_argument('--no_bias_above', type = int)
        self.parser.add_argument('--bottleneck', type = float)
        self.parser.add_argument('--translational', action = 'store_true')
        self.parser.add_argument('--translation_only', action = 'store_true')
        self.parser.add_argument('--dmol_output', action = 'store_true')
        self.parser.add_argument('--dmol_mixtures', type = int)
        self.parser.add_argument('--dmol_bits', type = int)

class TrainOptions(ArchOptions):
    """
    TrainOptions requires ArchOptions because 
    we cannot know whether the model needs to be reloaded until parsed
    """
    def setup(self):
        super().setup()
        #General
        self.parser.add_argument('--reload_model', action = 'store_true')
        self.parser.add_argument('--epochs', type = int, default = 100)
        self.parser.add_argument('--warmup', type = int, default = 0)
        self.parser.add_argument('--batch_size', type = int, default = 1)
        self.parser.add_argument('--num_workers', type = int, default = 0)
        self.parser.add_argument('--step_freq', type = int, default = 5)
        self.parser.add_argument('--save_freq', type = int, default = 1)
        self.parser.add_argument('--eval_freq', type = int, default = 1)
        self.parser.add_argument('--example_img', default = 'example.png')
        #Augmentations
        self.parser.add_argument('--aug_vflip', action = 'store_true')
        self.parser.add_argument('--aug_hflip', action = 'store_true')
        self.parser.add_argument('--aug_shift', type = int, default = 0)
        self.parser.add_argument('--aug_scale', type = float, default = 0.0)
        #Loss evaluation
        self.parser.add_argument('--learning_rate', type = float, default = 0.001)
        self.parser.add_argument('--grad_skip', type = float, default = 5000.0)
        self.parser.add_argument('--grad_thres', type = float, default = 300.0)
        self.parser.add_argument('--no_half_precision', action = 'store_true')
        self.parser.add_argument('--loss_kl_fac',  type = float, default = 1.0)
        self.parser.add_argument('--loss_tra_fac', type = float, default = 1.0)
        self.parser.add_argument('--loss_tkl_fac', type = float, default = 1.0)
        self.parser.add_argument('--best_metric', default = 'psnr_x1t')
        self.parser.add_argument('--best_decreasing', action = 'store_true')

class TestOptions(GlobalOptions):
    def setup(self):
        super().setup()
        self.parser.add_argument('--out_dir', default = 'results')
        self.parser.add_argument('--test_on_val', action = 'store_true')
