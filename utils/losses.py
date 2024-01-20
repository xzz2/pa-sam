import torch
from torch.nn import functional as F
from typing import List, Optional
import utils.misc as misc
from scipy.ndimage.morphology import distance_transform_edt
import os

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords

def get_uncertain_point_coords_with_randomness_with_uncertain_map(
    uncertainty_map, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = uncertainty_map.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=uncertainty_map.device)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = point_sample(uncertainty_map, point_coords, align_corners=False)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=uncertainty_map.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=uncertainty_map.device),
            ],
            dim=1,
        )
    return point_coords

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        mask = None
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    if mask is not None:
        inputs = inputs * mask
        targets = targets * mask
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks



def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        no_reduction: bool,
        mask = None
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    if no_reduction:
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        if mask is not None:
            return (torch.sum(loss * mask, dim=(-2,-1)) / (torch.sum(mask, dim=(-2,-1))+1e-8)).sum() / num_masks
        else:
            return loss.mean(1).sum() / num_masks
    else:
        loss = F.binary_cross_entropy_with_logits(inputs, targets)
        return loss



def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

def loss_masks(src_masks, target_masks, num_masks, oversample_ratio=3.0):
    """Compute the losses related to the masks: the focal loss and the dice loss.
    targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    """

    # No need to upsample predictions as we are using normalized coordinates :)

    with torch.no_grad():
        # sample point_coords
        point_coords = get_uncertain_point_coords_with_randomness(
            src_masks,
            lambda logits: calculate_uncertainty(logits),
            112 * 112,
            oversample_ratio,
            0.75,
        )
        # get gt labels
        point_labels = point_sample(
            target_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

    point_logits = point_sample(
        src_masks,
        point_coords,
        align_corners=False,
    ).squeeze(1)

    loss_mask = sigmoid_ce_loss(point_logits, point_labels, num_masks, no_reduction=True)
    loss_dice = dice_loss(point_logits, point_labels, num_masks)

    del src_masks
    del target_masks
    return loss_mask, loss_dice

def loss_masks_whole(src_masks, target_masks, num_masks):
    up_src_masks = F.interpolate(src_masks,target_masks.shape[-2:],mode="bilinear", align_corners=False)
    loss_mask = sigmoid_ce_loss(up_src_masks, target_masks, num_masks, no_reduction=False)
    loss_dice = dice_loss(up_src_masks, target_masks, num_masks)
    del src_masks
    del target_masks
    return loss_mask, loss_dice

def loss_masks_whole_uncertain(coarse_masks, refined_masks, target_masks, uncertain_map, num_masks):
    uncertain_map = F.interpolate(uncertain_map,target_masks.shape[-2:],mode="bilinear", align_corners=False).detach()
    up_coarse_masks = F.interpolate(coarse_masks,target_masks.shape[-2:],mode="bilinear", align_corners=False)
    loss_mask_coarse = sigmoid_ce_loss(up_coarse_masks, target_masks, num_masks, no_reduction=True, mask=(uncertain_map<0.5))
    loss_dice_coarse = dice_loss(up_coarse_masks, target_masks, num_masks, mask=(uncertain_map<0.5).flatten(1))
    up_refined_masks = F.interpolate(refined_masks,target_masks.shape[-2:],mode="bilinear", align_corners=False)
    loss_mask_refined = sigmoid_ce_loss(up_refined_masks, target_masks, num_masks, no_reduction=True, mask=(uncertain_map>=0.5))
    loss_dice_refined = dice_loss(up_refined_masks, target_masks, num_masks, mask=(uncertain_map>=0.5).flatten(1))
    del coarse_masks
    del refined_masks
    del target_masks
    return loss_mask_coarse+loss_mask_refined, loss_dice_coarse+loss_dice_refined

#--------------------------------------------------------------------------------------------------------------------
def loss_boxes(outputs, targets):
    """ 
    Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
    targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
    The target boxes are expected in format (x1, y1, x2, y2), normalized by the image size.
    When the box is irregular, loss_giou is None
    outputs: b*4
    targets: b*4
    """
    num_boxes = outputs.shape[0]

    loss_bbox = F.l1_loss(outputs, targets)
    
    if (outputs[:, 2:] >= outputs[:, :2]).all() and (targets[:, 2:] >= targets[:, :2]).all():
        loss_giou = 1 - torch.diag(misc.generalized_box_iou(outputs,targets))
        loss_giou = loss_giou.sum() / num_boxes
    else:
        loss_giou = torch.Tensor([-1]).to(outputs.device).squeeze(0)

    return loss_bbox, loss_giou

def distance_to_mask_label(distance_map, 
                    seg_label_map, 
                    return_tensor=False):

    max_distance = int(os.environ.get('dt_max_distance', 1)) # 5
    min_distance = int(os.environ.get('dt_min_distance', 0))
    if return_tensor:
        assert isinstance(distance_map, torch.Tensor)
        assert isinstance(seg_label_map, torch.Tensor)
    else:
        assert isinstance(distance_map, np.ndarray)
        assert isinstance(seg_label_map, np.ndarray)

    if return_tensor:
        mask_label_map = torch.zeros_like(seg_label_map).long().to(distance_map.device)
    else:
        mask_label_map = np.zeros(seg_label_map.shape, dtype=np.int)

    keep_mask = (distance_map <= max_distance) & (distance_map >= min_distance)
    mask_label_map[keep_mask] = 1
    mask_label_map[seg_label_map == -1] = -1

    return mask_label_map

def calc_weights(label_map, num_classes):

    weights = []
    for i in range(num_classes):
        weights.append((label_map == i).sum().data)
    weights = torch.FloatTensor(weights)
    weights_sum = weights.sum()
    return (1 - weights / weights_sum).cuda() 

def loss_uncertain(pred_mask,target):
    """
    pred_mask: b*h*w
    target: b*h*w
    """

    fg = (target >= 128).int()
    bg = (target < 128).int()
    gt_mask = []
    for i in range(target.shape[0]):
        gt_mask.append(torch.from_numpy(misc.mask_to_boundary(fg[i][0].cpu().numpy(), dilation_ratio=0.005, value=0)).cuda().unsqueeze(0).unsqueeze(0)+
                                   torch.from_numpy(misc.mask_to_boundary(bg[i][0].cpu().numpy(), dilation_ratio=0.005, value=1)).cuda().unsqueeze(0).unsqueeze(0))
    gt_mask = torch.cat(gt_mask,0)
    gt_size = gt_mask.shape[2:]
    mask_weights = calc_weights(gt_mask, 2)
    mask_weights = torch.where(gt_mask == 0, mask_weights[0], mask_weights[1])
    pred_mask = F.interpolate(pred_mask, size=gt_size, mode="bilinear", align_corners=True)
    mask_loss = F.binary_cross_entropy(pred_mask, gt_mask.float())

    return mask_loss, gt_mask

def l2_loss(input, target):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    pos_inds = torch.nonzero(target > 0.0).squeeze(1)
    if pos_inds.shape[0] > 0:
        cond = torch.abs(input[pos_inds] - target[pos_inds])
        loss = 0.5 * cond**2 / pos_inds.shape[0]
    else:
        loss = input * 0.0
    return loss.sum()

def loss_iou(pred_iou, src_masks, target_masks):
    """
    pred_iou: b*4
    src_masks:b,1,h,w
    target_masks:b,1,H,W
    """
    src_masks = F.interpolate(src_masks, size=target_masks.size()[2:], mode='bilinear', align_corners=False)
    target_iou = []
    for i in range(0,len(src_masks)):
        target_iou.append(misc.mask_iou(src_masks[i],target_masks[i]).unsqueeze(0))
    target_iou = torch.cat(target_iou, dim=0)
    loss_iou = l2_loss(pred_iou[:,0], target_iou.detach())
    
    return loss_iou



