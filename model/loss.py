import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)


def mtl_loss_naive(w, task_loss):
    """
    naive weighted sum
    :param w:
    :param task_loss:
    :return:
    """
    return w * task_loss
                                                            ##sqrt?

def mtl_loss_kendall(uncertainty, task_loss):
    """
    kendall loss
    :param uncertainties:
    :param task_losses:
    :return:
    """
    return (1.0 / (torch.exp(uncertainty))) * task_loss + uncertainty
                                                            ##sqrt?




def mtl_loss_liebel(uncertainty, task_loss):
    """
    liebel loss
    :param uncertainties:
    :param task_losses:
    :return:
    """
    return (1.0/ (torch.exp(uncertainty))) * task_loss + torch.log(1.0 + torch.exp(uncertainty))

