import torch
import torch.nn as nn
from monai.losses import DiceLoss
import segmentation_models_pytorch as smp





class Binary_Segmentation_Loss(nn.Module):
    def __init__(self, DICE_weight=1,BCE_weight=1,skeleton_weight=1,pos_rate=0.0003015102702192962):
        """
        Initialize the combined loss function with a weighting factor for the SDM product loss.

        Args:
        lambda_val (float): Weight for the product loss component.
        """
        super(Binary_Segmentation_Loss, self).__init__()
        self.dice_loss=smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # self.dice_loss=DiceLoss()
        self.dice_weight=DICE_weight
        self.pos_rate=pos_rate
        # self.bce_loss=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1-self.pos_rate)/self.pos_rate]).to("cuda"))
        # self.bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(81).to("cuda"))
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.bce_weight=BCE_weight
        self.skeleton_weight=skeleton_weight

    def forward(self,predictions,labels,skeletons):
        B, C, H, W = predictions.shape
        skeleton_loss = (
                    (torch.sigmoid(predictions) * skeletons).view(B, -1).sum(1) / skeletons.view(B, -1).sum(1)).mean()
        return self.dice_weight*self.dice_loss(predictions,labels)+self.bce_weight*self.bce_loss(predictions,labels)-self.skeleton_weight*skeleton_loss





# Implementation of ProductLoss as previously defined
class ProductLoss(nn.Module):
    def __init__(self):
        super(ProductLoss, self).__init__()

    def forward(self, predictions, targets):
        prod = targets * predictions
        denom = prod + predictions.pow(2) + targets.pow(2)
        denom = torch.clamp(denom, min=1e-8)
        loss_product = -torch.mean(prod / denom)
        return loss_product


# Example usage:
