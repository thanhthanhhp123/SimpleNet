import logging
import os
import pickle
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.nn as nn
import tqdm
import torch.functional as F
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
        
        def _embed(self, images, detach=True, provide_patch_shapes = False, evaluation = False):
            B = len(images)
            if not evaluation and self.train_backbone:
                self.forward_modules['feature_aggerator'](images, eval = evaluation)
                features = self.forward_modules['feature_aggregator'].train()
            else:
                _ = self.forward_modules['feature_aggregator'].eval()
                with torch.no_grad():
                    features = self.forward_modules['feature_aggregator'](images)
            
            features = [features[layer] for layer in self.layers_to_extract_from]

            for i, feat in enumerate(features):
                if len(feat.shape) == 3:
                    B, L, C = feat.shape
                    features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)
            
            features = [
                self.patch_maker.patchify(x,
                                          return_spatial_infor=True) for x in features
            ]
            patch_shapes = [x[1] for x in features]
            features = [x[0] for x in features]
            ref_num_patches = patch_shapes[0]

            for i in range(1, len(features)):
                _features = features[i]
                patch_dims = patch_shapes[i]

                _features = _features.permute(0, -3, -2, -1, 1, 2)
                perm_base_shape = _features.shape
                _features = _features.reshape(
                    -1, *_features.shape[-2:]
                )
                _features = F.interpolate(
                    _features.unsqueeze(1),
                    size = (ref_num_patches[0], ref_num_patches[1]),
                    mode = 'bilinear',
                    align_corners = False
                )
                _features = _features.squeeze(1)
                _features = _features.reshape(
                    *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                )
                _features = _features.permute(0, -2, -1, 1, 2, 3)
                _features = _features.reshape(
                    len(_features), -1, *_features.shape[-3:]
                )
                features[i] = _features
            features = [x.reshape(-1, *x.shape[-3:]) for x in features]

            features = self.forward_modules['preprocessing'](features)
            features = self.forward_modules['preadapt_aggretor'](features)

            return features, patch_shapes
    
    def test(self, training_data, test_data):
        ckpt_path = os.path.join(self.ckpt_dir, 'models.ckpt')
        if os.path.exists(ckpt_path):
            state_dicts = torch.load(ckpt_path)
            if 'pretrained_enc' in state_dicts:
                self.features_enc.load_state_dict(state_dicts['pretrained_enc'])
            if 'pretrained_dec' in state_dicts:
                self.features_dec.load_state_dict(state_dicts['pretrained_dec'])
        aggerator = {'scores': [],
                     'segmentations': [],
                     'features': []}
        scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
        aggerator['scores'].append(scores)
        aggerator['segmentations'].append(segmentations)
        aggerator['features'].append(features)

        scores = np.array(aggerator['scores'])
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scroes = np.mean(scores, axis=0)

        segmentations = np.array(aggerator['segmentations'])
        min_scores = (
            segmentations.reshape(
            len(segmentations), -1
            ).min(axis = -1).reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(
            len(segmentations), -1
            ).max(axis = -1).reshape(-1, 1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        segmentations = np.mean(segmentations, axis=0)

        anomaly_labels = [
            x[1] != 'good' for x in test_data.dataset.data_to_iterate
        ]

        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, anomaly_labels
        )['auroc']

        pixel_scores = metrics.compute_imagewise_retrieval_metrics(
            segmentations, masks_gt
        )

        full_pixel_auroc = pixel_scores['auroc']

        return auroc, full_pixel_auroc
    
    def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt):
        scores = np.squeeze(np.array(scores))
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)

        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt
        )['auroc']

        if len(masks_gt) > 0:
            pass