
import torch
import torch.nn as nn
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


class DRL_Simsiam(nn.Module):
    """
    Build a DRL_Simsiam model.
    """
    def __init__(self, dim=2048, pred_dim=512, args=None):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(DRL_Simsiam, self).__init__()

        self.args = args

        # create the encoder
        self.encoder = DRL_Simsiam.get_backbone(args.arch, args)
        if self.args.last_dim == -1:
            prev_dim = self.encoder.fc.weight.shape[1]
        else:
            prev_dim = self.args.last_dim
        self.size1 = int(round(prev_dim * self.args.hp1)) ## out_dim 512
        self.size2 = prev_dim - self.size1
        print("prev_dim, self.size1, self.size2", prev_dim, self.size1, self.size2)

        self.encoder.fc = nn.Identity()
        self.projector_img = nn.Sequential(nn.Linear(self.size1, dim, bias=False),
                                        nn.BatchNorm1d(dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(dim, dim, bias=False),
                                        nn.BatchNorm1d(dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(dim, dim),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        # build a 2-layer predictor
        self.predictor_img = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.projector_aug = nn.Sequential(nn.Linear(self.size2, dim, bias=False),
                                        nn.BatchNorm1d(dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(dim, dim, bias=False),
                                        nn.BatchNorm1d(dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(dim, dim),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        # build a 2-layer predictor
        self.predictor_aug = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    @staticmethod
    def get_backbone(backbone_name, args = None):
        return {'resnet18': ResNet18(args=args),
                'resnet34': ResNet34(args=args),
                'resnet50': ResNet50(args=args),
                'resnet101': ResNet101(args=args),
                'resnet152': ResNet152(args=args)}[backbone_name]

    def forward(self, im_aug1, im_aug2):

        z1 = self.encoder(im_aug1)
        z1_sep_img = z1[:,:self.size1]
        z1_sep_aug = z1[:,self.size1:]

        z2 = self.encoder(im_aug2)
        z2_sep_img = z2[:,:self.size1]
        z2_sep_aug = z2[:,self.size1:]

        z1_feature_img = self.projector_img(z1_sep_img)
        z2_feature_img = self.projector_img(z2_sep_img)

        z1_feature_aug = self.projector_aug(z1_sep_aug)
        z2_feature_aug = self.projector_aug(z2_sep_aug)

        p1_img = self.predictor_img(z1_feature_img)
        p2_img = self.predictor_img(z2_feature_img)

        p1_aug = self.predictor_aug(z1_feature_aug)
        p2_aug = self.predictor_aug(z2_feature_aug)

        #return z1_feature_img.detach(), z2_feature_img.detach(), z1_feature_aug.detach(), z2_feature_aug.detach(), p1_img, p2_img, p1_aug, p2_aug
        return z1_feature_img.detach(), z2_feature_img.detach(), z1_feature_aug.detach(), z2_feature_aug.detach(), p1_img, p2_img, p1_aug, p2_aug, z1_sep_img, z1_sep_aug, z2_sep_img, z2_sep_aug




















