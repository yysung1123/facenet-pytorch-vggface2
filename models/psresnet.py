import torch.nn as nn
import torch
from torch.nn import functional as F
from .utils_psresnet import psresnet18

class PSResnet18_11111_CBAMTriplet(nn.Module):
    """Constructs a ResNet-18 model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 512.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=512, pretrained=False):
        super(PSResnet18_11111_CBAMTriplet, self).__init__()
        self.model = psresnet18(pretrained=pretrained, in_channels=6, groups=[1,1,1,1,1])
        self.cbam = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.tanh = nn.Tanh()

        # Output embedding
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        images, attention_map = images[:, :3, :, :], images[:, 3:4, :, :]
        att = self.cbam(attention_map)
        images = torch.cat((images, images * (self.tanh(att) + 1)), dim=1)

        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding
    
class PSResnet18_21111_CBAMTriplet(nn.Module):
    """Constructs a ResNet-18 model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 512.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=512, pretrained=False):
        super(PSResnet18_21111_CBAMTriplet, self).__init__()
        self.model = psresnet18(pretrained=pretrained, in_channels=6, groups=[2,1,1,1,1])
        self.cbam = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.tanh = nn.Tanh()

        # Output embedding
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        images, attention_map = images[:, :3, :, :], images[:, 3:4, :, :]
        att = self.cbam(attention_map)
        images = torch.cat((images, images * (self.tanh(att) + 1)), dim=1)

        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding
    
class PSResnet18_22111_CBAMTriplet(nn.Module):
    """Constructs a ResNet-18 model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 512.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=512, pretrained=False):
        super(PSResnet18_22111_CBAMTriplet, self).__init__()
        self.model = psresnet18(pretrained=pretrained, in_channels=6, groups=[2,2,1,1,1])
        self.cbam = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.tanh = nn.Tanh()

        # Output embedding
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        images, attention_map = images[:, :3, :, :], images[:, 3:4, :, :]
        att = self.cbam(attention_map)
        images = torch.cat((images, images * (self.tanh(att) + 1)), dim=1)

        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class PSResnet18_22211_CBAMTriplet(nn.Module):
    """Constructs a ResNet-18 model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 512.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=512, pretrained=False):
        super(PSResnet18_22211_CBAMTriplet, self).__init__()
        self.model = psresnet18(pretrained=pretrained, in_channels=6, groups=[2,2,2,1,1])
        self.cbam = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.tanh = nn.Tanh()

        # Output embedding
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        images, attention_map = images[:, :3, :, :], images[:, 3:4, :, :]
        att = self.cbam(attention_map)
        images = torch.cat((images, images * (self.tanh(att) + 1)), dim=1)

        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding
