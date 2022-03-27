import gym

# from ray.rllib.utils.framework import try_import_tf,try_import_torch
import torch
import torch.nn as nn
import tensorflow as tf

from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN

# tf1, tf, tfv = try_import_tf()
# torch, nn = try_import_torch()


class FCMaskedActionsModelTF(DistributionalQTFModel, TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super(FCMaskedActionsModelTF, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        true_obs_space = gym.spaces.MultiBinary(n=obs_space.shape[0] - action_space.n)
        self.action_embed_model = FullyConnectedNetwork(
            obs_space=true_obs_space,
            action_space=action_space,
            num_outputs=action_space.n,
            model_config=model_config,
            name=name + "action_model",
        )
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        raw_actions, _ = self.action_embed_model({"obs": input_dict["obs"]["real_obs"]})
        # inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        logits = tf.where(tf.math.equal(action_mask, 1), raw_actions, tf.float32.min)
        return logits, state

    def value_function(self):
        return self.action_embed_model.value_function()


class FCMaskedActionsModelTorch(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        true_obs_space = gym.spaces.MultiBinary(n=obs_space.shape[0] - action_space.n)
        self.action_embed_model = TorchFC(
            obs_space=true_obs_space,
            action_space=action_space,
            num_outputs=action_space.n,
            model_config=model_config,
            name=name + "action_model",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        raw_actions, _ = self.action_embed_model({"obs": input_dict["obs"]["real_obs"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return raw_actions, state

        masked_logits = torch.where(action_mask == 1, raw_actions, torch.Tensor([FLOAT_MIN]).to(raw_actions.device))

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.action_embed_model.value_function()
