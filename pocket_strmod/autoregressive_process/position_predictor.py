import torch
from torch.nn import Module, Sequential
from torch.nn import functional as F
from .layers import GBPerceptronVN, GBLinear
import math

GAUSSIAN_COEF = 1.0 / math.sqrt(2 * math.pi)

class PositionPredictor(Module):
    def __init__(self, in_sca, in_vec, num_filters, n_component, bottleneck=1, use_conv1d=False):
        super(PositionPredictor, self).__init__()
        self.n_component = n_component
        self.gvp = Sequential(
            GBPerceptronVN(
                in_sca*2, in_vec, num_filters[0], num_filters[1], bottleneck=bottleneck, use_conv1d=use_conv1d
                ),
            GBLinear(
                num_filters[0], num_filters[1], num_filters[0], num_filters[1], bottleneck=bottleneck,
                use_conv1d=use_conv1d
                )
        )
        self.mu_net = GBLinear(
            num_filters[0], num_filters[1], n_component, n_component, bottleneck=bottleneck, use_conv1d=use_conv1d
            )
        self.logsigma_net= GBLinear(
            num_filters[0], num_filters[1], n_component, n_component, bottleneck=bottleneck, use_conv1d=use_conv1d
            )
        self.pi_net = GBLinear(
            num_filters[0], num_filters[1], n_component, 1, bottleneck=bottleneck, use_conv1d=use_conv1d
            )

    def forward(self, h_compose, idx_pivotal, pos_compose, atom_type_emb=None):
        h_pivotal = [h[idx_pivotal] for h in h_compose]
        pos_pivotal = pos_compose[idx_pivotal]
        if isinstance(atom_type_emb, torch.Tensor):
            h_pivotal[0] = torch.cat([h_pivotal[0], atom_type_emb], dim=1)

        feat_pivotal = self.gvp(h_pivotal)
        relative_mu = self.mu_net(feat_pivotal)[1] 
        logsigma = self.logsigma_net(feat_pivotal)[1] 
        sigma = torch.exp(logsigma)
        pi = self.pi_net(feat_pivotal)[0]
        pi = F.softmax(pi, dim=1)
        
        abs_mu = relative_mu + pos_pivotal.unsqueeze(dim=1).expand_as(relative_mu)
        return relative_mu, abs_mu, sigma, pi

    def get_mdn_probability(self, mu, sigma, pi, pos_target):
        prob_gauss = self._get_gaussian_probability(mu, sigma, pos_target)
        prob_mdn = pi * prob_gauss
        prob_mdn = torch.sum(prob_mdn, dim=1)
        return prob_mdn

    def _get_gaussian_probability(self, mu, sigma, pos_target):
        target = pos_target.unsqueeze(1).expand_as(mu)
        errors = target - mu
        sigma = sigma + 1e-16
        p = GAUSSIAN_COEF * torch.exp(-0.5 * (errors / sigma)**2) / sigma
        p = torch.prod(p, dim=2)
        return p 

    def sample_batch(self, mu, sigma, pi, num):
        index_cats = torch.multinomial(pi, num, replacement=True)
        index_batch = torch.arange(len(mu)).unsqueeze(-1).expand(-1, num) 
        mu_sample = mu[index_batch, index_cats]  
        sigma_sample = sigma[index_batch, index_cats]
        values = torch.normal(mu_sample, sigma_sample) 
        return values

    def get_maximum(self, mu, sigma, pi):
        return mu