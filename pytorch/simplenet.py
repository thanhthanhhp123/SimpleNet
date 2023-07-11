import logging
import os
import pickle
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.tensorboard import SummaryWriter
import common
import metrics

LOGGER = logging.getLogger(__name__)

def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, torch.nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)

class Discriminator(nn.Module):
    def __init__(self, inplanes, n_layers = 1, hidden = None):
        super(Discriminator, self).__init__()
        _hidden = inplanes if hidden is None else hidden
        self.body = nn.Sequential()
        for i in range(n_layers - 1):
            _in = inplanes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d'%(i+1),
                                 nn.Sequential(
                                        nn.Linear(_in, _hidden),
                                        nn.BatchNorm1d(_hidden),
                                        nn.LeakyReLU(0.2, inplace = True)
                                 ))
            self.tail = nn.Linear(_hidden, 1, bias = False)
            self.apply(init_weight)
    
    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x

class Projection(nn.Module):
    def __init__(self, inplanes, outplanes = None, n_layers = 1, layer_type = 0):
        super(Projection, self).__init__()

        if outplanes is None:
            outplanes = inplanes
        self.layers = nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = inplanes if i == 0 else _out
            _out = outplanes
            self.layers.add_module(f'{i}fc',
                                   nn.Linear(_in, _out))
            if i < n_layers -1:
                if layer_type > 1:
                    self.layers.add_module(f'{i}relu',
                                           nn.LeakyReLU(0.2, inplace = True))
        self.apply(init_weight)
    def forward(self, x):
        x = self.layers(x)
        return x


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir)
    
    def step(self):
        self.g_iter += 1


class SimpleNet(nn.Module):
    def __init__(self, device):
        super(SimpleNet, self).__init__()
        self.device = device
    
    def load(self,
             backbone,
             layers_to_extract_from,
             device,
             input_shape,
             pretrain_embed_dimension,
             target_embed_dimension,
             patchsize = 3,
             patchstride = 1,
             embedding_size = None,
             meta_epochs = 1,
             aed_meta_epochs = 1,
             gan_epochs = 1,
             noise_std = 0.05,
             mix_noise = 1,
             noise_type = 'GAU',
             dsc_layers = 2,
             dsc_hidden = None,
             dsc_margin = .8,
             dsc_lr = 2e-4,
             train_backbone = False,
             auto_noise = 0,
             cos_lr = False,
             lr = 1e-4,
             pre_proj = 0,
             proj_layer_type=0,
             **kwargs):
        pid = os.getpid()

        self.backbone = backbone
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride = patchstride)

        self.forward_modules = nn.ModuleList()
        feature_aggregator = common.NetwotkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules['feature_aggregator'] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension)
        self.forward_modules['preprocessing'] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggretor = common.Aggretor(
            target_dim = target_embed_dimension
        )

        _ = preadapt_aggretor.to(self.device)

        self.forward_modules['preadapt_aggretor'] = preadapt_aggretor

        self.anomaly_segmentor = common.RescaleSegmentor(
            self.device, target_size=input_shape[-2:]
        )

        self.embedding_size = embedding_size if embedding_size is not None else self.target_embed_dimension
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(
                self.forward_modules['feature_aggregator'].backbone.parameters(),
                self.lr
            )
        
        self.aed_meta_epochs = aed_meta_epochs
        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(
                self.target_embed_dimension,
                self.target_embed_dimension,
                pre_proj, 
                proj_layer_type
            )
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), self.lr)

            self.auto_noise =[auto_noise, None]
            self.dsc_lr = dsc_lr
            self.gan_epochs = gan_epochs
            self.mix_noise = mix_noise
            self.noise_type = noise_type
            self.noise_std = noise_std
            self.disciminator = Discriminator(
                self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden
            )
            self.disciminator.to(self.device)
            self.dsc_opt = torch.optim.Adam(self.disciminator.parameters(), self.dsc_lr, weight_decay=1e-5)
            self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.dsc_opt, (meta_epochs - aed_meta_epochs) * gan_epochs, self.dcs_lr * .4)
            self.dsc_margin = dsc_margin

            self.model_dir = ''
            self.dataset_name = ''
            self.tau = 1
            self.logger = None

        def set_model_dir(self, model_dir, dataset_name):
            self.model_dir = model_dir
            os.makedirs(self.model_dir, exist_ok=True)
            self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.tb_dir = os.path.join(self.ckpt_dir, 'tb')
            os.makedirs(self.tb_dir, exist_ok=True)
            self.logger = TBWrapper(self.tb_dir)
        
        def embed(self, data):
            if isinstance(data, torch.utils.data.DataLoader):
                features = []
                for image in data:
                    if isinstance(image, dict):
                        image = image['image']
                        input_image = image.to(torch.float).to(self.device)
                    with torch.no_grad():
                        features.append(self._embed(input_image))
                return features
            return self._embed(data)
