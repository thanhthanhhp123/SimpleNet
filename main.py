import logging
import os
import sys

import numpy as np
import torch
import backbones
import common
import metrics
import simplenet
import utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {
    'mvtec': ['datasets.mvtec', 'MVTecDataset'],
}

def main(**kwargs):
    pass

def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    run_name,
    test
):
    methods = {key: item for (key, item) in methods}
    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name
        )
    pid = os.getpid()
    list_of_dataloaders = methods['get_dataloaders'](seed)
    device = utils.set_torch_device(gpu)

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        utils.fix_seeds(seed, device)
        
        dataset_name = dataloaders["training"].name

        imagesize = dataloaders["training"].dataset.imagesize
        simplenet_list = methods['get_simplenet'](imagesize, device)

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        for i, SimpleNet in enumerate(simplenet_list):
            if SimpleNet.backbone.seed is not None:
                utils.fix_seeds(SimpleNet.backbone.seed, device)
            LOGGER.info(
                'Training Models ({}/{})'.format(i+1, len(simplenet_list))
            )

            SimpleNet.set_model_dir(os.path.join(models_dir, f'{i}'), dataset_name)
            if not test:
                i_auroc, p_auroc, pro_auroc = SimpleNet.train(dataloaders['training'], dataloaders['testing'])
            else:
                i_auroc, p_auroc, pro_auroc = SimpleNet.test(dataloaders['testing'])


            result_collect.append(
                {
                    'dataset_name': dataset_name,
                    'instance_auroc': i_auroc,
                    'full_pixel_auroc': p_auroc,
                    'anomaly_pixel_auroc': pro_auroc,
                }
            )
            for key, item in result_collect[-1].items():
                if key != 'dataset_name':
                    LOGGER.info('{0}: {1:3.3f}'.format(key, item))
        LOGGER.info('\n\n------\n')

    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results['dataset_name'] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )

def net(backbone_names,
        layers_to_extract_from, pretrain_embed_dimension,
        target_embed_dimension,
        patchsize, embedding_size,
        meta_epochs,
        aed_meta_epochs,
        gan_epochs,
        noise_std,
        dsc_layers,
        dsc_hidden,
        dsc_margin,
        dsc_lr,
        auto_noise,
        train_backbone,
        cos_lr,
        pre_proj,
        proj_layer_type,
        mix_noise
        ):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split('.')[0])
            layer = '.'.join(layer.split('.')[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]
    
    def get_simplenet(input_shape, device):
        simplenets = []
        for backbone_name, layers_to_extract_from in zip(backbone_names, layers_to_extract_from_coll):
            backbone_seed = None
            if '.seed-' in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
                backbone = backbones.load(backbone_name)
                backbone.name, backbone.seed = backbone_name, backbone_seed

                simplenet_inst = simplenet.SimpleNet(device)
                simplenet_inst.load(
                    backbone = backbone,
                    layers_to_extract_from = layers_to_extract_from,
                    device = device,
                    input_shape = input_shape,
                    pretrain_embed_dimension=pretrain_embed_dimension,
                    target_embed_dimension=target_embed_dimension,
                    patchsize=patchsize,
                    embedding_size=embedding_size,
                    meta_epochs=meta_epochs,
                    aed_meta_epochs=aed_meta_epochs,
                    gan_epochs=gan_epochs,
                    noise_std=noise_std,
                    dsc_layers=dsc_layers,
                    dsc_hidden=dsc_hidden,
                    dsc_margin=dsc_margin,
                    dsc_lr=dsc_lr,
                    auto_noise=auto_noise,
                    train_backbone=train_backbone,
                    cos_lr=cos_lr,
                    pre_proj=pre_proj,
                    proj_layer_type=proj_layer_type,
                    mix_noise=mix_noise
                )
                simplenets.append(simplenet_inst)
        return simplenets
    
    return ('get_simplenet', get_simplenet)

def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    rotate_degrees,
    translate,
    scale,
    brightness,
    contrast,
    saturation,
    gray,
    hflip,
    vflip,
    augment
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname = subdataset,
                resize = resize,
                train_val_split = train_val_split,
                imagesize = imagesize,
                split = dataset_library.DatasetSplit.TRAIN,
                seed = seed,
                rotate_degrees = rotate_degrees,
                translate = translate,
                brightness_factor = brightness,
                contrast_factor = contrast,
                saturation_factor = saturation,
                gray_p = gray,
                h_flip_p = hflip,
                v_flip_p = vflip,
                scale = scale,
                augment = augment
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname = subdataset,
                resize = resize,
                imagesize = imagesize,
                split = dataset_library.DatasetSplit.TEST,
                seed = seed,
            )

            LOGGER.info(f'Dataset: train = {len(train_dataset)}, test = {len(test_dataset)}')
            
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size = batch_size,
                shuffle = True,
                num_workers = num_workers,
                prefetch_factor = 2,
                pin_memory = 2,)
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size = batch_size,
                shuffle = False,
                num_workers = num_workers,
                prefetch_factor = 2,
                pin_memory = True
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += '_' + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname = subdataset,
                    resize = resize,
                    train_val_split = train_val_split,
                    imagesize = imagesize,
                    split = dataset_library.DatasetSplit.VAL,
                    seed = seed,
                )
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size = batch_size,
                    shuffle = False,
                    num_workers = num_workers,
                    prefetch_factor = 2,
                    pin_memory = True
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                'training': train_dataloader,
                'testing': test_dataloader,
                'validation': val_dataloader
            }
            dataloaders.append(dataloader_dict)
        return dataloaders
    return ('get_dataloaders', get_dataloaders)
                                                                                     