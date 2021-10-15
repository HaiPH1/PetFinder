import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, image_model="resnet50"):
        super().__init__()
        if image_model == "resnet50":
            self.model = models.resnet34(pretrained=True)
            # self.model.load_state_dict(torch.load("../model.pth"), strict=False)
        self.image_model = nn.Sequential(*list(self.model.children())[:-1])
        self.num_feat = self.model.fc.in_features

    def forward(
        self,
        images=None,
    ):
        img_feat = self.image_model(images)
        img_feat = torch.flatten(img_feat, start_dim=2)
        img_feat = img_feat[:, :, 0]

        return img_feat


class PetFinderModel(nn.Module):
    def __init__(
        self,
        image_model="resnet50",
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(image_model=image_model)
        self.num_feat = self.image_encoder.num_feat
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.hidden_layer = nn.Linear(self.num_feat, self.num_feat)
        self.regression_layer = nn.Linear(self.num_feat, 1)

    def forward(
        self,
        images=None,
    ):
        image_features = self.image_encoder(images)
        x = self.dropout(image_features)
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.regression_layer(x)
        x = torch.squeeze(x)
        x = torch.sigmoid(x)
        return x
