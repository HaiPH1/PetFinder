import gc
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from utils.utils import seed_all, count_parameters
from utils import constants
import pandas as pd
from supervised.models import PetFinderModel
from supervised.datasets import PetFinderDataset
from torch.optim import Adam
from supervised.engine import train_fn, validation_fn
from sklearn.model_selection import KFold


parser = argparse.ArgumentParser()


parser.add_argument(
    "--image_model",
    default="resnet50",
    type=str,
    help="model name or model path of pretrained visual models",
)

parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="num examples per batch",
)

parser.add_argument(
    "--lr",
    default=0.0001,
    type=float,
    help="learning rate",
)

parser.add_argument(
    "--n_epochs",
    default=10,
    type=int,
    help="num epochs required for training",
)

parser.add_argument(
    "--seed",
    default=1996,
    type=int,
    help="seed for reproceduce",
)

parser.add_argument(
    "--accu_step",
    default=32,
    type=int,
    help="accu_grad_step",
)


args = parser.parse_args()

if __name__ == "__main__":

    seed_all(seed_value=args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = pd.read_csv(constants.TRAINING_LABEL)
    kf = KFold(n_splits=20, shuffle=True, random_state=2111)
    for i, (train_index, valid_index) in enumerate(kf.split(samples)):
        training_samples = samples.iloc[train_index]
        valid_samples = samples.iloc[valid_index]

        train_dataset = PetFinderDataset(
            training_samples, constants.TRAINING_IMAGE_DIR
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )

        valid_dataset = PetFinderDataset(
            valid_samples,
            constants.TRAINING_IMAGE_DIR,
            mode="valid",
        )
        valid_sampler = SequentialSampler(valid_dataset)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=valid_sampler
        )

        model = PetFinderModel(args.image_model)
        print("The number of parameters of the model: ", count_parameters(model))
        model.to(device)

        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = None
        criterion = nn.MSELoss()

        min_mse = 1000
        for epoch in range(args.n_epochs):
            print("Training on epoch", epoch + 1)
            train_loss, train_rmse = train_fn(
                dataloader=train_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scheduler=scheduler,
                accu_step=args.accu_step,
            )

            validation_loss, valid_rmse = validation_fn(
                valid_loader, model, criterion, device
            )
            if min_mse > valid_rmse:
                min_mse = valid_rmse
                torch.save(model.state_dict(), "../weights/model_%s_%0.4f.pth" % (i, np.sqrt(min_mse)))

            print("Train loss: %.4f, Validation loss: %.4f" % (train_loss, validation_loss))
            print("Train RMSE: %.4f, Validation RMSE: %.4f" % (np.sqrt(train_rmse), np.sqrt(valid_rmse)))
            print("*" * 100)
        print("Min RMSE at %s fold: %s" % (i, round(np.sqrt(min_mse), 4)))
        print("#"*100)
