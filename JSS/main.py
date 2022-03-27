import time

import ray
import wandb

import random

import numpy as np

import ray.tune.integration.wandb as wandb_tune

from ray.rllib.agents.ppo import PPOTrainer

from CustomCallbacks import *
from models import *

from typing import Dict, Tuple

import multiprocessing as mp
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog

from ray.tune.utils import flatten_dict

_exclude_results = ["done", "should_checkpoint", "config"]

# Use these result keys to update `wandb.config`
_config_results = [
    "trial_id",
    "experiment_tag",
    "node_ip",
    "experiment_id",
    "hostname",
    "pid",
    "date",
]


def _handle_result(result: Dict) -> Tuple[Dict, Dict]:
    config_update = result.get("config", {}).copy()
    log = {}
    flat_result = flatten_dict(result, delimiter="/")

    for k, v in flat_result.items():
        if any(k.startswith(item + "/") or k == item for item in _config_results):
            config_update[k] = v
        elif any(k.startswith(item + "/") or k == item for item in _exclude_results):
            continue
        elif not wandb_tune._is_allowed_type(v):
            continue
        else:
            log[k] = v

    config_update.pop("callbacks", None)  # Remove callbacks
    return log, config_update


def train_func():
    default_config = {
        "env": "JSSEnv:jss-v1",  # 强化学习环境的名字，通过gym包创建
        "seed": 0,  # 随机数种子
        "framework": "torch",  # 使用的框架,i.e. tf,torch
        "log_level": "WARN",  # 日志等级
        "num_gpus": 1,  # GPU数量
        "instance_path": "instances/ta41",  # JSSP实例的路径
        "evaluation_interval": None,
        "metrics_num_episodes_for_smoothing": 2000,
        "gamma": 1.0,  # 强化学习中的折扣因子
        "num_workers": mp.cpu_count() // 2,  # 多线程读取数据的cpu
        "train_batch_size": mp.cpu_count() // 2 * 4 * 704,  # 训练的批次大小
        "num_envs_per_worker": 4,
        "rollout_fragment_length": 704,  # TO TUNE
        "sgd_minibatch_size": 33000,
        "layer_nb": 2,  # 自定义模型的层数
        "layer_size": 319,  # 自定义模型的隐藏层大小
        "lr": 0.0006861,  # TO TUNE 学习率
        "lr_start": 0.0006861,  # TO TUNE 起始学习率？
        "lr_end": 0.00007783,  # TO TUNE 最终学习率？
        "clip_param": 0.541,  # TO TUNE
        "vf_clip_param": 26,  # TO TUNE
        "num_sgd_iter": 12,  # TO TUNE
        "vf_loss_coeff": 0.7918,
        "kl_coeff": 0.496,
        "kl_target": 0.05047,  # TO TUNE
        "lambda": 1.0,
        "entropy_coeff": 0.0002458,  # TUNE LATER
        "entropy_start": 0.0002458,
        "entropy_end": 0.002042,
        "entropy_coeff_schedule": None,
        "batch_mode": "truncate_episodes",
        "grad_clip": None,
        "use_critic": True,
        "use_gae": True,
        "shuffle_sequences": True,
        "observation_filter": "NoFilter",
        "_fake_gpus": False,
    }

    wandb.init(config=default_config)
    ray.init()
    tf.random.set_seed(default_config["seed"])
    torch.manual_seed(default_config["seed"])
    torch.cuda.manual_seed(default_config["seed"])
    np.random.seed(default_config["seed"])
    random.seed(default_config["seed"])
    # wandb.log(default_config)
    config = wandb.config

    # 自定义的模型
    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)
    ModelCatalog.register_custom_model("fc_masked_model_torch", FCMaskedActionsModelTorch)

    config["model"] = {
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model_torch",  # i.e. fc_masked_model_tf, fc_masked_model_torch
        "fcnet_hiddens": [config["layer_size"] for _ in range(config["layer_nb"])],
        "vf_share_layers": False,
    }
    config["env_config"] = {"env_config": {"instance_path": config["instance_path"]}}

    config = with_common_config(config)
    config["seed"] = 0
    config["callbacks"] = CustomCallbacks
    config["train_batch_size"] = config["sgd_minibatch_size"]

    config["lr"] = config["lr_start"]
    config["lr_schedule"] = [[0, config["lr_start"]], [15000000, config["lr_end"]]]

    config["entropy_coeff"] = config["entropy_start"]
    config["entropy_coeff_schedule"] = [[0, config["entropy_start"]], [15000000, config["entropy_end"]]]

    config.pop("instance_path", None)
    config.pop("layer_size", None)
    config.pop("layer_nb", None)
    config.pop("lr_start", None)
    config.pop("lr_end", None)
    config.pop("entropy_start", None)
    config.pop("entropy_end", None)

    stop = {
        "time_total_s": 10 * 60,  # 训练时间多少秒
    }
    # wandb.log(config)
    start_time = time.time()
    trainer = PPOTrainer(config=config)
    while start_time + stop["time_total_s"] > time.time():
        result = trainer.train()
        result = wandb_tune._clean_log(result)
        log, config_update = _handle_result(result)
        wandb.log(log)
        # wandb.config.update(config_update, allow_val_change=True)
    # trainer.export_policy_model("/home/jupyter/JSS/JSS/models/")

    ray.shutdown()


if __name__ == "__main__":
    train_func()
