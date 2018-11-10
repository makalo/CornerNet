import os
import numpy as np

class Config:
    def __init__(self):
        self._configs = {}
        self._configs["dataset"] = None


        # Training Config
        self._configs["display"]           = 5
        self._configs["decay_step"]          = 12000
        self._configs["epoch_num"]          = 100
        self._configs["stepsize"]          = 450000
        self._configs["learning_rate"]     = 0.00025
        self._configs["decay_rate"]        = 0.95
        self._configs["max_iter"]          = 500000
        self._configs["val_iter"]          = 100
        self._configs["batch_size"]        = 5
        self._configs["snapshot_name"]     = 'corner_net'
        self._configs["prefetch_size"]     = 100
        self._configs["weight_decay"]      = False
        self._configs["weight_decay_rate"] = 1e-5
        self._configs["weight_decay_type"] = "l2"
        self._configs["pretrain"]          = None
        self._configs["opt_algo"]          = "adam"
        self._configs["chunk_sizes"]       = [4, 5, 5, 5, 5, 5, 5, 5, 5, 5]



        # Directories
        self._configs["data_dir"]   = "./data"
        self._configs["cache_dir"]  = "./cache"
        self._configs["config_dir"] = "./config"
        self._configs["result_dir"] = "./results"
        self._configs["debug_dir"]   = "./debug"

        # Split
        self._configs["train_split"] = "trainval"
        self._configs["val_split"]   = "minival"
        self._configs["test_split"]  = "testdev"

        # Rng
        self._configs["data_rng"] = np.random.RandomState(123)
        self._configs["nnet_rng"] = np.random.RandomState(317)

        #data_config
        self._configs["categories"]        =80
        self._configs["rand_scale_min"]    =0.6
        self._configs["rand_scale_max"]    =1.4
        self._configs["rand_scale_step"]   =0.1
        self._configs["rand_scales"]       =[0.5,0.75,1,1.25,1.5]
        self._configs["rand_crop"]         =True
        self._configs["rand_color"]        =True
        self._configs["border"]            =128
        self._configs["gaussian_bump"]     =True
        self._configs["input_size"]        =[511,511]
        self._configs["output_sizes"]      =[[128,128]]
        self._configs["test_scales"]       =[0.5,0.75,1,1.25,1.5]
        self._configs["top_k"]             =100
        self._configs["ae_threshold"]      =0.5
        self._configs["nms_threshold"]     =0.5
        self._configs["merge_bbox"]        =True
        self._configs["weight_exp"]        =10
        self._configs["max_per_image"]     =100
        self._configs["gaussian_radius"]   =-1
        self._configs["gaussian_iou"]      =0.7

    @property
    def gaussian_iou(self):
        return self._configs["gaussian_iou"]
    @property
    def gaussian_radius(self):
        return self._configs["gaussian_radius"]
    @property
    def categories(self):
        return self._configs["categories"]
    @property
    def rand_scale_min(self):
        return self._configs["rand_scale_min"]
    @property
    def rand_scale_max(self):
        return self._configs["rand_scale_max"]
    @property
    def rand_scale_step(self):
        return self._configs["rand_scale_step"]
    @property
    def rand_scales(self):
        return self._configs["rand_scales"]
    @property
    def rand_crop(self):
        return self._configs["rand_crop"]
    @property
    def rand_color(self):
        return self._configs["rand_color"]
    @property
    def border(self):
        return self._configs["border"]
    @property
    def gaussian_bump(self):
        return self._configs["gaussian_bump"]
    @property
    def input_size(self):
        return self._configs["input_size"]
    @property
    def output_sizes(self):
        return self._configs["output_sizes"]
    @property
    def test_scales(self):
        return self._configs["test_scales"]
    @property
    def top_k(self):
        return self._configs["top_k"]
    @property
    def ae_threshold(self):
        return self._configs["ae_threshold"]
    @property
    def nms_threshold(self):
        return self._configs["nms_threshold"]
    @property
    def merge_bbox(self):
        return self._configs["merge_bbox"]
    @property
    def weight_exp(self):
        return self._configs["weight_exp"]
    @property
    def max_per_image(self):
        return self._configs["max_per_image"]
    @property
    def chunk_sizes(self):
        return self._configs["chunk_sizes"]

    @property
    def train_split(self):
        return self._configs["train_split"]

    @property
    def val_split(self):
        return self._configs["val_split"]

    @property
    def test_split(self):
        return self._configs["test_split"]

    @property
    def full(self):
        return self._configs

    @property
    def sampling_function(self):
        return self._configs["sampling_function"]

    @property
    def data_rng(self):
        return self._configs["data_rng"]

    @property
    def nnet_rng(self):
        return self._configs["nnet_rng"]

    @property
    def opt_algo(self):
        return self._configs["opt_algo"]

    @property
    def weight_decay_type(self):
        return self._configs["weight_decay_type"]

    @property
    def prefetch_size(self):
        return self._configs["prefetch_size"]

    @property
    def pretrain(self):
        return self._configs["pretrain"]

    @property
    def weight_decay_rate(self):
        return self._configs["weight_decay_rate"]

    @property
    def weight_decay(self):
        return self._configs["weight_decay"]

    @property
    def result_dir(self):
        result_dir = os.path.join(self._configs["result_dir"], self.snapshot_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        return result_dir
    @property
    def debug_dir(self):
        debug_dir = os.path.join(self.cache_dir, "debug")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        return debug_dir

    @property
    def dataset(self):
        return self._configs["dataset"]

    @property
    def snapshot_name(self):
        return self._configs["snapshot_name"]

    @property
    def snapshot_dir(self):
        snapshot_dir = os.path.join(self.cache_dir, "nnet", self.snapshot_name)

        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        return snapshot_dir

    @property
    def snapshot_file(self):
        snapshot_file = os.path.join(self.snapshot_dir, self.snapshot_name + ".ckpt")
        return snapshot_file

    @property
    def config_dir(self):
        return self._configs["config_dir"]

    @property
    def batch_size(self):
        return self._configs["batch_size"]
    @property
    def epoch_num(self):
        return self._configs["epoch_num"]


    @property
    def max_iter(self):
        return self._configs["max_iter"]

    @property
    def learning_rate(self):
        return self._configs["learning_rate"]

    @property
    def decay_rate(self):
        return self._configs["decay_rate"]
    @property
    def decay_step(self):
        return self._configs["decay_step"]

    @property
    def stepsize(self):
        return self._configs["stepsize"]

    @property
    def snapshot(self):
        return self._configs["snapshot"]

    @property
    def display(self):
        return self._configs["display"]

    @property
    def val_iter(self):
        return self._configs["val_iter"]

    @property
    def data_dir(self):
        return self._configs["data_dir"]

    @property
    def cache_dir(self):
        if not os.path.exists(self._configs["cache_dir"]):
            os.makedirs(self._configs["cache_dir"])
        return self._configs["cache_dir"]

    def update_config(self, new):
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key]

cfg = Config()
