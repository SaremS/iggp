from dataclasses import dataclass

import torch

@dataclass
class GPPosteriors:
    """
    Holds the necessary objects to calculate the mean integrated gradients for
    Gaussian Processes

    gp_mus (torch.Tensor): Posterior means of the Gaussian Process
    gp_dx_mus (torch.Tensor): Posterior means of the corresponding
                                derivative Gaussian Process 

    gp_sigma_sqs (torch.Tensor): Posterior variances of the Gaussian Process
    gp_dx_sigma_sqs (torch.Tensor): Posterior variances of the corresponding
                                    derivative Gaussian Process

    gp_xdx_cross_covs (torch.Tensor): Posterior covariances between Gaussian Process
                                    and its corresponding derivative GP
    """
    gp_mus: torch.Tensor
    gp_dx_mus: torch.Tensor

    gp_sigma_sqs: torch.Tensor
    gp_dx_sigma_sqs: torch.Tensor

    gp_xdx_cross_covs: torch.Tensor


@dataclass
class GPPosteriorsMultiOutput:
    """
    Holds the necessary objects to calculate the mean integrated gradients for
    Gaussian Processes with multivariate output 

    mu_X (torch.Tensor): Posterior means of the Gaussian Process
    mu_Y (torch.Tensor): Posterior means of the corresponding
                                derivative Gaussian Process 

    K_XX (torch.Tensor): Posterior covariance matrix of the Gaussian Process
    K_XY (torch.Tensor): Posterior cross-covariance vector (matrix form) of the corresponding
                                    derivative Gaussian Process (i.e. the covariance of each
                                    GP with the partial derivative GP of the target output
                                    dimension)
    """
    mu_X: torch.Tensor
    mu_Y: torch.Tensor

    K_XX: torch.Tensor
    K_XY: torch.Tensor
