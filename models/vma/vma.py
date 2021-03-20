import torch
from math import log
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical as Cat, kl_divergence as KL
from .decoder import GenerativeModel
from .encoder import RecognitionModel, AddressRecognition
from .prior import ConditionalPrior


class VMA(nn.Module):
    def __init__(self, encoder=None, decoder=None, latent_dim=50,
                 matching_dim=128, hidden_size=256,
                 data_dim=26 * 26, importance_num=4, likelihood='bernoulli'):
        """
        Total loss for training model is
        L = E_q log p(x | z, m_a) + KL(q(z | x, m_a) || p(z | m_a)) +
                                    KL(q(a | M) || p(a))
        It's intractable since a is discrete variable, so instead we
        minimize next unbiased estimator
        L = E_{a, k ~ q}log[1/K \\sum p(x, z, m_a)/q(a, z | x) ]
        """
        super().__init__()
        self.importance = importance_num
        # rec and gen encoders predict vectors that will be projected to
        # mean and variance by additional layers
        self.rec_encoder = encoder(latent_dim=2 * latent_dim)
        self.gen_encoder = self.rec_encoder#encoder(latent_dim=2 * latent_dim)
        # gen decoder accepts latent sample and output of the gen_encoder
        self.gen_decoder = decoder(
            latent_dim=2 * latent_dim,
            hidden_size=hidden_size,
            skip=True
        )
        self.gen = GenerativeModel(encoder=self.gen_encoder, decoder=self.gen_decoder,
                                   distribution=likelihood, latent_dim=latent_dim)
        self.rec = RecognitionModel(feature_encoder=self.rec_encoder, latent_dim=latent_dim)
        self.addr = AddressRecognition(
            encoder=encoder, matching_dim=matching_dim, data_dim=data_dim,
            num_dummies=2
        )
        self.prior = ConditionalPrior(encoder=self.gen_encoder, latent_dim=latent_dim)

    def _select(self, tensor, a):
        if tensor.dim() == 3:
            if a.dim() != 2:
                a = a.unsqueeze(1)
            a = a.unsqueeze(2).repeat(1, 1, tensor.shape[2])
        return tensor.gather(1, a)

    def forward(self, data, iw=None, test=False, mean=True, **kwargs):
        history = {}
        (M, x), _ = data
        if M is None:
            # threat M as empty condition
            M = x.new_zeros((x.size(0), 0, x.size(1)))
        timesteps = M.size(1)
        batch = M.size(0)
        # prior address distribution - unnormalized uniform
        M = self.addr.add_dummy(M)
        p_a = Cat(M.new_ones(M.shape[:-1]))
        q_a = self.addr(x, M)
        if iw is None:
            iw = self.importance
        a = q_a.sample((iw,)).t()
        m_a = self._select(M, a)

        z_rec = self.rec(x, m_a)
        z_pri = self.prior(m_a)

        z_sample = z_rec.rsample()
        x_rec = self.gen(z_sample, m_a)
        # log p(x | z, a)
        x = x.unsqueeze(1).repeat(1, iw, 1)
        gen_llh_x = x_rec.log_prob(x).sum(-1)
        # log p(z | a)
        gen_llh_z = z_pri.log_prob(z_sample).sum(-1)
        # log p(a)
        gen_llh_a = self._select(p_a.logits, a)
        # log q(z | x, a)
        rec_llh_z = z_rec.log_prob(z_sample).sum(-1)
        # log q(a | x)
        rec_llh_a = self._select(q_a.logits, a)
        rec_llh_a_d = rec_llh_a.detach()
        bounds = gen_llh_x + gen_llh_z + gen_llh_a - (rec_llh_z + rec_llh_a_d)
        if history is not None:
            history['KL_a'] = KL(q_a, p_a)
            history['KL_z'] = KL(z_rec, z_pri).sum(-1).mean(-1)
            history['NLL'] = -gen_llh_x.logsumexp(1)
            history['ELBO'] = (history['NLL'] + history['KL_z']
                                              + history['KL_a'])
            if mean:
                for k, v in history.items():
                    history[k] = v.mean()
        elbo = bounds.logsumexp(1) - log(iw)
        if test:
            if mean:
                return -elbo.mean(), history
            else:
                return -elbo, history
        L = self.vimco(bounds)
        loss = -(elbo + (L * rec_llh_a).sum(1))
        if mean:
            return loss.mean(), history
        else:
            return loss, history

    def generate_batch(self, data, history=None):
        (M, x), _ = data
        if M is None:
            # threat M as empty condition
            M = x.new_zeros((x.size(0), 0, x.size(1)))
        timesteps = M.size(1)
        batch = M.size(0)
        p_a = Cat(M.new_ones((batch, timesteps)))
        a = p_a.sample()
        m_a = self._select(M, a)
        z_pri = self.prior(m_a)
        return self.gen(z_pri.sample(), m_a).probs

    @staticmethod
    def logdiff(a, b):
        """
        assumes that a > b
        log(exp(a) - exp(b)) = a + log(1 - exp(b - a))
        """
        # TODO: this formula does not work in fp16 case, think about
        # taylor series
        return a + (1 - (b - a).exp().clamp(1e-10, 1 - 1e-5)).log()

    def vimco(self, bounds):
        # L - importance weighted lower bound estimate
        # L = log (1/K sum((p(x, z, a) / q(z, a | x))))
        iw = bounds.shape[1]
        logsumexp = bounds.logsumexp(1, keepdim=True)
        L = logsumexp - log(iw)
        w = F.softmax(bounds, dim=1)
        # L_mi = L^-i
        L_mi = self.logdiff(logsumexp, bounds) - log(iw - 1)
        return (L - L_mi - w).detach()
