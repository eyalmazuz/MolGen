import torch
import torch.nn.functional as F


# FAMO class definition (adapted from the paper's pseudocode)
class FAMO:
    def __init__(self, num_tasks, min_losses, lr=0.025, gamma=0.001):
        # min_losses: a tensor of shape (num_tasks,) representing a lower bound on each task's loss.
        self.min_losses = min_losses
        self.xi = torch.zeros(num_tasks, device=min_losses.device, requires_grad=True)
        self.xi_opt = torch.optim.Adam([self.xi], lr=lr, weight_decay=gamma)
        self.prev_losses = None  # To store previous losses for the update

    def get_weighted_loss(self, losses):
        """
        Compute the FAMO balanced loss given per-task losses.
        Args:
            losses (torch.Tensor): Tensor of shape (num_tasks,) with current per-task losses.
        Returns:
            torch.Tensor: A scalar loss that is the weighted combination of per-task losses.
        """
        # Compute the softmax over task logits: z_t = softmax(xi)
        z = F.softmax(self.xi, dim=-1)
        # Adjust losses by subtracting the lower bounds (with a small constant for numerical stability)
        D = losses - self.min_losses + 1e-8
        # Compute constant c to re-normalize the gradient magnitude
        c = 1 / (z / D).sum().detach()
        # The balanced loss is a weighted sum of the log-transformed (adjusted) losses
        balanced_loss = (c * torch.log(D) * z).sum()
        return balanced_loss

    def update(self, prev_losses, curr_losses):
        """
        Update the task logits xi using the difference in log losses.
        Args:
            prev_losses (torch.Tensor): Previous per-task losses.
            curr_losses (torch.Tensor): Current per-task losses.
        """
        # Compute the change in log losses for each task
        delta = torch.log(prev_losses - self.min_losses + 1e-8) - torch.log(curr_losses - self.min_losses + 1e-8)
        # Compute the softmax probabilities from xi
        z = F.softmax(self.xi, dim=-1)
        # Compute the vector-Jacobian product: delta_t = ∇_{xi} z^T * delta
        d = torch.autograd.grad(outputs=z, inputs=self.xi, grad_outputs=delta.detach(), retain_graph=True)[0]
        # Update xi with the computed gradient
        self.xi_opt.zero_grad()
        self.xi.grad = d
        self.xi_opt.step()
