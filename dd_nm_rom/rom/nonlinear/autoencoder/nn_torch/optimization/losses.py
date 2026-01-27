import torch

from torch.nn.modules import loss
from dd_nm_rom import backend as bkd


_LOSS_IDS = ("mae", "mare", "mse", "msre")

def get(
  identifier="mse",
  reduction="mean"
):
  if (isinstance(identifier, str) and (identifier.lower() in _LOSS_IDS)):
    return {
      "mae":   loss.L1Loss,
      "mare":  MARELoss,
      "mse":   loss.MSELoss,
      "msre":  MSRELoss
    }[identifier.lower()](reduction=reduction)
  else:
    raise ValueError(
      f"Could not interpret loss function identifier: '{identifier}'."
    )


class MARELoss(loss._Loss):

  def __init__(
    self,
    reduction: str = "mean"
  ) -> None:
    super(MARELoss, self).__init__(reduction=reduction)

  def forward(
    self,
    input: torch.Tensor,
    target: torch.Tensor
  ) -> torch.Tensor:
    target = target.view_as(input)
    num = torch.abs(target - input)
    den = torch.abs(target)
    return self.compute(num, den)

  def compute(self, num, den):
    num = torch.sum(num, dim=-1)
    den = torch.sum(den, dim=-1) + bkd.epsilon()
    loss = num / den
    if (self.reduction == "mean"):
      return torch.mean(loss)
    elif (self.reduction == "sum"):
      return torch.sum(loss)
    else:
      return loss


class MSRELoss(MARELoss):

  def __init__(
    self,
    reduction: str = "mean"
  ) -> None:
    super(MSRELoss, self).__init__(reduction=reduction)

  def forward(
    self,
    input: torch.Tensor,
    target: torch.Tensor
  ) -> torch.Tensor:
    target = target.view_as(input)
    num = torch.square(target - input)
    den = torch.square(target)
    return self.compute(num, den)
