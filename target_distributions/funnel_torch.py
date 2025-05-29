import torch
import torch.nn.functional as F
import math
from interfaces.target_torch import TorchTargetDistribution

class NealFunnelTorch(TorchTargetDistribution):
    """
    Neal's Standard Funnel distribution, implemented in PyTorch.

    log p(v, z_1, ..., z_{D-1}) = log N(v | mu_v, sigma_v^2) + sum_{k=1}^{D-1} log N(z_k | mu_z, exp(v))
    where exp(v) is the variance of z_k.

    Total log-density:
    log p(x) = -0.5 * D * log(2pi) - 0.5 * log(sigma_v^2) - 0.5 * (v - mu_v)^2 / sigma_v^2
               - 0.5 * (D-1) * v - 0.5 * exp(-v) * sum_{k=1}^{D-1} (z_k - mu_z)^2
    """

    def __init__(self, dim, mu_v=0.0, sigma_v_sq=9.0, mu_z=0.0, device=None):
        super().__init__(dim, device)
        if dim < 1:
            raise ValueError("dim must be at least 1 for Neal's Funnel (v variable).")

        self.mu_v = torch.tensor(mu_v, device=self.device, dtype=torch.float32)
        self.sigma_v_sq = torch.tensor(sigma_v_sq, device=self.device, dtype=torch.float32)
        if self.sigma_v_sq <= 0:
            raise ValueError("sigma_v_sq must be positive.")
        self.mu_z = torch.tensor(mu_z, device=self.device, dtype=torch.float32)

        self.log_sigma_v_sq = torch.log(self.sigma_v_sq)
        self.log_2_pi = torch.tensor(2.0 * math.pi, device=self.device, dtype=torch.float32).log()
        
        self.D_tensor = torch.tensor(float(self.dim), device=self.device, dtype=torch.float32)
        if self.dim > 1:
            self.D_minus_1_tensor = torch.tensor(float(self.dim - 1), device=self.device, dtype=torch.float32)
        else: # Only v variable
            self.D_minus_1_tensor = torch.tensor(0.0, device=self.device, dtype=torch.float32)


    def log_density(self, x_tensor):
        """
        Compute the log density of Neal's Funnel.
        x_tensor: (batch_size, D) or (D,)
        """
        # Handle device mismatch by moving input to the correct device
        if x_tensor.device != self.device:
            x_tensor = x_tensor.to(self.device)
            
        if x_tensor.ndim == 1:
            x_tensor_internal = x_tensor.unsqueeze(0)
        else:
            x_tensor_internal = x_tensor
        
        batch_size = x_tensor_internal.shape[0]
        v = x_tensor_internal[:, 0]

        # Log prior for v
        log_prior_v = -0.5 * self.log_2_pi - 0.5 * self.log_sigma_v_sq - 0.5 * (v - self.mu_v)**2 / self.sigma_v_sq
        
        if self.dim == 1: # Only v variable, no z variables
            total_log_dens = log_prior_v
        else:
            zs = x_tensor_internal[:, 1:]
            sum_sq_zs = torch.sum((zs - self.mu_z)**2, dim=1)
            
            # Log likelihood for z_k's
            # log p(z_k|v) = -0.5 * log(2pi) - 0.5 * log(exp(v)) - 0.5 * (z_k - mu_z)^2 / exp(v)
            #                = -0.5 * log(2pi) - 0.5 * v - 0.5 * (z_k - mu_z)^2 * exp(-v)
            # Sum over (D-1) z_k's:
            log_likelihood_zs = (-0.5 * self.D_minus_1_tensor * self.log_2_pi 
                               - 0.5 * self.D_minus_1_tensor * v 
                               - 0.5 * torch.exp(-v) * sum_sq_zs)
            total_log_dens = log_prior_v + log_likelihood_zs

        if x_tensor.ndim == 1:
            return total_log_dens.squeeze(0)
        return total_log_dens

    def density(self, x):
        return torch.exp(self.log_density(x))

    def get_name(self):
        return f"NealFunnelTorch_D{self.dim}"

    def draw_sample(self, beta=1.0):
        """
        Draw a sample from Neal's Funnel.
        This method should not be necessary but is here for completeness.
        """
        raise NotImplementedError("NealFunnelTorch.draw_sample is not implemented.")
        current_beta = torch.tensor(beta, device=self.device, dtype=torch.float32)
        
        # Sample v
        tempered_sigma_v_sq = self.sigma_v_sq / current_beta
        v_sample = torch.normal(mean=self.mu_v, std=torch.sqrt(tempered_sigma_v_sq), size=(1,)).to(self.device)

        if self.dim == 1:
            sample_torch = v_sample
        else:
            # Sample z_k's
            # Original variance of z_k is exp(v_sample_unadjusted_for_beta_in_its_own_sampling)
            # Here, v_sample is already drawn from its beta-adjusted distribution.
            # The heuristic is that var(z_k | v_from_tempered_prior_v) = exp(v_from_tempered_prior_v) / beta
            tempered_var_z = torch.exp(v_sample) / current_beta
            std_z_sample = torch.sqrt(tempered_var_z)
            
            zs_sample = self.mu_z + std_z_sample * torch.randn(int(self.D_minus_1_tensor.item()), device=self.device)
            sample_torch = torch.cat((v_sample, zs_sample.flatten())).to(self.device)
        
        return sample_torch.cpu().numpy()


class SuperFunnelTorch(TorchTargetDistribution):
    """
    Hierarchical Logistic Regression "Super-Funnel" implemented in PyTorch.
    The state vector Theta consists of:
    (alphas (J), betas (J*K), mu_alpha (1), mu_beta (K), tau_alpha (1), tau_beta (1))
    Priors:
        alpha_j ~ N(mu_alpha, tau_alpha^2)
        beta_jk ~ N(mu_beta_k, tau_beta^2)
        mu_alpha ~ N(0, prior_hypermean_std^2)
        mu_beta_k ~ N(0, prior_hypermean_std^2)
        tau_alpha ~ HalfCauchy(0, prior_tau_scale)
        tau_beta ~ HalfCauchy(0, prior_tau_scale)
    Likelihood:
        y_ij ~ Bernoulli(logit_inv(alpha_j + x_ij^T beta_j))
    """

    def __init__(self, J, K, X_data, Y_data, 
                 prior_hypermean_std=10.0, prior_tau_scale=2.5, device=None):
        
        self.J = J
        self.K = K
        # Dim: J (alphas) + J*K (betas) + 1 (mu_alpha) + K (mu_beta) + 1 (tau_alpha) + 1 (tau_beta)
        dim = J + J * K + 1 + K + 1 + 1
        super().__init__(dim, device)

        if not (isinstance(X_data, list) and len(X_data) == J):
            raise ValueError(f"X_data must be a list of J={J} tensors.")
        if not (isinstance(Y_data, list) and len(Y_data) == J):
            raise ValueError(f"Y_data must be a list of J={J} tensors.")

        self.X_data = []
        self.Y_data = []
        self.n_j_array = torch.empty(J, device=self.device, dtype=torch.long)
        for j in range(J):
            if not isinstance(X_data[j], torch.Tensor) or not isinstance(Y_data[j], torch.Tensor):
                 raise ValueError(f"X_data[{j}] and Y_data[{j}] must be PyTorch tensors.")
            if X_data[j].ndim != 2 or X_data[j].shape[1] != K:
                raise ValueError(f"X_data[{j}] must have shape (n_j, K={K}). Got {X_data[j].shape}")
            if Y_data[j].ndim != 1 or Y_data[j].shape[0] != X_data[j].shape[0]:
                raise ValueError(f"Y_data[{j}] must have shape (n_j,). Got {Y_data[j].shape}, X_data had {X_data[j].shape[0]} samples.")
            
            self.X_data.append(X_data[j].to(self.device).to(torch.float32))
            self.Y_data.append(Y_data[j].to(self.device).to(torch.float32)) # Ensure Y is float for calcs
            self.n_j_array[j] = Y_data[j].shape[0]

        self.prior_hypermean_std = torch.tensor(prior_hypermean_std, device=self.device, dtype=torch.float32)
        self.prior_hypermean_var = self.prior_hypermean_std**2
        self.log_prior_hypermean_var = torch.log(self.prior_hypermean_var)

        self.prior_tau_scale = torch.tensor(prior_tau_scale, device=self.device, dtype=torch.float32)
        self.log_prior_tau_scale = torch.log(self.prior_tau_scale)

        self.log_2_pi = torch.tensor(2.0 * math.pi, device=self.device, dtype=torch.float32).log()
        self.log_2 = torch.tensor(2.0, device=self.device, dtype=torch.float32).log()
        self.log_pi = torch.tensor(math.pi, device=self.device, dtype=torch.float32).log()
        self.K_tensor = torch.tensor(float(self.K), device=self.device, dtype=torch.float32)

    def _parse_theta(self, Theta_tensor):
        batch_size = Theta_tensor.shape[0]
        idx = 0
        
        alphas = Theta_tensor[:, idx : idx + self.J]
        idx += self.J
        
        betas_flat = Theta_tensor[:, idx : idx + self.J * self.K]
        betas = betas_flat.view(batch_size, self.J, self.K)
        idx += self.J * self.K
        
        mu_alpha = Theta_tensor[:, idx]
        idx += 1
        
        mu_beta = Theta_tensor[:, idx : idx + self.K]
        idx += self.K
        
        tau_alpha = Theta_tensor[:, idx]
        idx += 1
        
        tau_beta = Theta_tensor[:, idx]
        
        return alphas, betas, mu_alpha, mu_beta, tau_alpha, tau_beta

    def log_density(self, Theta_tensor):
        # Handle device mismatch by moving input to the correct device
        if Theta_tensor.device != self.device:
            Theta_tensor = Theta_tensor.to(self.device)
            
        if Theta_tensor.ndim == 1:
            Theta_tensor_internal = Theta_tensor.unsqueeze(0)
        else:
            Theta_tensor_internal = Theta_tensor

        batch_size = Theta_tensor_internal.shape[0]
        log_p = torch.zeros(batch_size, device=self.device, dtype=torch.float32)

        alphas, betas, mu_alpha, mu_beta, tau_alpha, tau_beta = self._parse_theta(Theta_tensor_internal)

        # Constraint check for taus
        valid_mask = (tau_alpha > 1e-9) & (tau_beta > 1e-9) # Add small epsilon for stability if needed near zero
        log_p[~valid_mask] = -torch.inf
        
        # Reduce computation to only valid samples if there are invalid ones
        # This can be complex to refactor all subsequent ops. A simpler way:
        # compute all terms, and for terms involving log(tau) or 1/tau, ensure they result in -inf if tau is invalid.
        # The initial log_p[~valid_mask] = -torch.inf will dominate if any tau is bad.

        # Log-Likelihood LL = sum_j sum_i [y_ij * logsigmoid(eta_ij) + (1-y_ij) * logsigmoid(-eta_ij)]
        current_LL = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        for j in range(self.J):
            alpha_j_b = alphas[:, j]  # (batch_size,)
            beta_j_b = betas[:, j, :]  # (batch_size, K)
            X_j = self.X_data[j]      # (n_j, K)
            Y_j = self.Y_data[j]      # (n_j,)

            # eta_j_b = alpha_j_b.unsqueeze(1) + torch.matmul(X_j, beta_j_b.T).T 
            # X_j (n_j,K), beta_j_b.T (K,batch) -> (n_j,batch) -> .T (batch,n_j)
            eta_j_b = alpha_j_b.unsqueeze(1) + torch.einsum('nk,bk->bn', X_j, beta_j_b)
            
            log_p_y_j = (Y_j.unsqueeze(0) * F.logsigmoid(eta_j_b) + 
                         (1 - Y_j.unsqueeze(0)) * F.logsigmoid(-eta_j_b)).sum(dim=1)
            current_LL += log_p_y_j
        log_p += current_LL

        # Log-Prior for alphas: sum_j N(alpha_j | mu_alpha, tau_alpha^2)
        # log_tau_alpha_safe = torch.where(tau_alpha > 0, torch.log(tau_alpha), -torch.inf)
        # log_prior_alpha_terms = -0.5 * self.log_2_pi - log_tau_alpha_safe.unsqueeze(1) - \ 
        #                         0.5 * (alphas - mu_alpha.unsqueeze(1))**2 / (tau_alpha.unsqueeze(1)**2 + 1e-30) # Add epsilon for stability
        # log_p += log_prior_alpha_terms.sum(dim=1)
        # More direct way using valid_mask:
        if torch.any(valid_mask):
            safe_tau_alpha = torch.where(valid_mask, tau_alpha, torch.ones_like(tau_alpha))
            log_prior_alpha_terms = -0.5 * self.log_2_pi - torch.log(safe_tau_alpha).unsqueeze(1) - \
                                    0.5 * (alphas - mu_alpha.unsqueeze(1))**2 / (safe_tau_alpha.unsqueeze(1)**2)
            log_p[valid_mask] += log_prior_alpha_terms[valid_mask].sum(dim=1)

        # Log-Prior for betas: sum_j sum_k N(beta_jk | mu_beta_k, tau_beta^2)
        # Which is sum_j N_K(beta_j | mu_beta, tau_beta^2 I)
        # log_tau_beta_safe = torch.where(tau_beta > 0, torch.log(tau_beta), -torch.inf)
        # diff_betas = betas - mu_beta.unsqueeze(1)  # (batch, J, K)
        # sum_sq_diff_betas_per_j = (diff_betas**2).sum(dim=2)  # (batch, J)
        # log_prior_beta_terms_per_j = -0.5 * self.K_tensor * self.log_2_pi - \
        #                              self.K_tensor * log_tau_beta_safe.unsqueeze(1) - \
        #                              0.5 * sum_sq_diff_betas_per_j / (tau_beta.unsqueeze(1)**2 + 1e-30)
        # log_p += log_prior_beta_terms_per_j.sum(dim=1)
        if torch.any(valid_mask):
            safe_tau_beta = torch.where(valid_mask, tau_beta, torch.ones_like(tau_beta))
            diff_betas = betas - mu_beta.unsqueeze(1)  # (batch, J, K)
            sum_sq_diff_betas_per_j = (diff_betas**2).sum(dim=2)  # (batch, J)
            log_prior_beta_terms_per_j = -0.5 * self.K_tensor * self.log_2_pi - \
                                     self.K_tensor * torch.log(safe_tau_beta).unsqueeze(1) - \
                                     0.5 * sum_sq_diff_betas_per_j / (safe_tau_beta.unsqueeze(1)**2)
            log_p[valid_mask] += log_prior_beta_terms_per_j[valid_mask].sum(dim=1)


        # Log-Prior for mu_alpha: N(mu_alpha | 0, prior_hypermean_var)
        log_prior_mu_alpha_val = -0.5 * self.log_2_pi - 0.5 * self.log_prior_hypermean_var - \
                                 0.5 * mu_alpha**2 / self.prior_hypermean_var
        log_p += log_prior_mu_alpha_val # Applies to all, invalid already -inf

        # Log-Prior for mu_beta: N_K(mu_beta | 0, prior_hypermean_var I)
        sum_sq_mu_beta = (mu_beta**2).sum(dim=1)
        log_prior_mu_beta_val = -0.5 * self.K_tensor * self.log_2_pi - \
                                0.5 * self.K_tensor * self.log_prior_hypermean_var - \
                                0.5 * sum_sq_mu_beta / self.prior_hypermean_var
        log_p += log_prior_mu_beta_val

        # Log-Prior for tau_alpha, tau_beta: HalfCauchy(0, prior_tau_scale)
        # log p(tau) = log(2) - log(pi) - log(scale) - log(1 + (tau/scale)^2)
        # Must have tau > 0. This is handled by valid_mask. Compute for all, then select.
        if torch.any(valid_mask):
            term_tau_alpha = (tau_alpha / self.prior_tau_scale)**2
            log_prior_tau_alpha_val = self.log_2 - self.log_pi - self.log_prior_tau_scale - torch.log1p(term_tau_alpha)
            log_p[valid_mask] += log_prior_tau_alpha_val[valid_mask]

            term_tau_beta = (tau_beta / self.prior_tau_scale)**2
            log_prior_tau_beta_val = self.log_2 - self.log_pi - self.log_prior_tau_scale - torch.log1p(term_tau_beta)
            log_p[valid_mask] += log_prior_tau_beta_val[valid_mask]

        if Theta_tensor.ndim == 1:
            return log_p.squeeze(0)
        return log_p

    def density(self, x):
        return torch.exp(self.log_density(x))

    def get_name(self):
        return f"SuperFunnelTorch_J{self.J}_K{self.K}"

    def draw_sample(self, beta=1.0):
        """
        NOTE: This should not be a required method for the interface. Just here for completeness.
        I don't think this is used in the code.
        
        Draw a sample from the prior distributions, with heuristic tempering for beta != 1.0.
        Normal prior stddevs are scaled by 1/sqrt(beta).
        HalfCauchy prior scales are scaled by 1/sqrt(beta).
        This is a rough heuristic for PT ladder setup and not a true tempered sample.
        Returns a numpy.ndarray.
        """
        raise NotImplementedError("SuperFunnelTorch.draw_sample is not implemented.")
        current_beta_tensor = torch.tensor(beta, device=self.device, dtype=torch.float32)
        sqrt_beta = torch.sqrt(current_beta_tensor)

        # Hypermeans
        tempered_hypermean_std = self.prior_hypermean_std / sqrt_beta
        mu_alpha_sample = torch.normal(0.0, tempered_hypermean_std, device=self.device)
        mu_beta_sample = torch.normal(0.0, tempered_hypermean_std, size=(self.K,), device=self.device)

        # Hyperprecisions (taus)
        tempered_tau_scale = self.prior_tau_scale / sqrt_beta
        # Sample from Cauchy(0, tempered_tau_scale) and take abs for HalfCauchy
        # Cauchy samples: scale * tan(pi * (rand - 0.5))
        tau_alpha_sample = tempered_tau_scale * torch.abs(torch.tan(math.pi * (torch.rand((), device=self.device) - 0.5)))
        tau_beta_sample = tempered_tau_scale * torch.abs(torch.tan(math.pi * (torch.rand((), device=self.device) - 0.5)))
        # Ensure they are not exactly zero if scale is very small or due to randomness
        tau_alpha_sample = torch.clamp(tau_alpha_sample, min=1e-6)
        tau_beta_sample = torch.clamp(tau_beta_sample, min=1e-6)

        # Alphas
        alphas_sample = torch.normal(mu_alpha_sample, tau_alpha_sample, size=(self.J,), device=self.device)

        # Betas
        # mu_beta_sample is (K,), tau_beta_sample is scalar. Need mu_beta_sample for each J.
        betas_sample = torch.normal(mu_beta_sample.unsqueeze(0).expand(self.J, -1), 
                                  tau_beta_sample, 
                                  size=(self.J, self.K), device=self.device)

        # Flatten and concatenate
        theta_list = [
            alphas_sample.flatten(),
            betas_sample.flatten(),
            mu_alpha_sample.unsqueeze(0),
            mu_beta_sample.flatten(),
            tau_alpha_sample.unsqueeze(0),
            tau_beta_sample.unsqueeze(0)
        ]
        sample_torch = torch.cat(theta_list).to(self.device)
        return sample_torch.cpu().numpy() 
