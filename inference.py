"""
The inference results are saved to a csv file.
Usage:
    Inference with model on wandb:
        python inference.py \
        --timm_name {timm model name} \
        --wandb_run_path {wandb_run_path} \
        --model_name {model name storaged in wandb} \
        --data_ver {data version} \
        --image_size {image size default: 224}
"""
import pandas as pd

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import timm

import wandb

from src import set_device, SubstrateDataset, TestTransforms, predict_classes


def main(timm_name, model_name, wandb_run_path, data_ver, image_size):

    # gpu or cpu
    device = set_device()

    # read data
    test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/substrate/input/sample_submit.tsv', sep='\t', header=None)

    # image name list & dummy label list
    x_test = test[0].values
    dummy = test[1].values

    # dataset
    test_dataset = SubstrateDataset(
        x_test,
        dummy,
        img_dir=f'/content/drive/MyDrive/Colab Notebooks/substrate/input/{data_ver}/test',
        transform=TestTransforms(image_size=image_size),
        phase='test'
    )

    # dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # load model
    model = timm.create_model(timm_name, pretrained=False, num_classes=2)

    # restore model in wandb
    best_model = wandb.restore(f'{model_name}.ckpt', run_path=wandb_run_path)

    # load state_dict from ckpt with 'model.' deleted
    state_dict = torch.load(best_model.name, map_location=torch.device(device))['state_dict']
    new_state_dict = { k.lstrip('model.') : v for k, v in state_dict.items() }
    model.load_state_dict(new_state_dict, strict=True)

    # inference
    preds = predict_classes(model, test_dataloader, device)

    # submit
    test[1] = preds
    test.to_csv(f'/content/drive/MyDrive/Colab Notebooks/substrate/submit/submission_{model_name}.tsv', sep='\t', header=None, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--timm_name', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--wandb_run_path', type=str)
    parser.add_argument('--data_ver', type=str)
    parser.add_argument('--image_size', type=int, default=224)
    
    args = parser.parse_args()

    main(
        timm_name=args.timm_name,
        model_name=args.model_name,
        wandb_run_path=args.wandb_run_path,
        data_ver=args.data_ver,
        image_size=args.image_size,
    )