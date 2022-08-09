import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

from molecule_optimizer.externals.fast_jtnn import JTNNVAE

HIDDEN_SIZE = 450
LATENT_SIZE = 56
DEPTH_T = 20
DEPTH_G = 3


class SemiJTVAE(JTNNVAE):
    def __init__(
        self,
        vocab,
        hidden_size,
        latent_size,
        depthT,
        depthG,
        label_size,
        label_mean,
        label_var,
    ):
        super().__init__(vocab, hidden_size, latent_size, depthT, depthG)

        self.label_size = label_size
        self.scaler = self.setup_scaler(label_mean, label_var)

        self.T_mean = nn.Linear(
            self.hidden_size + self.label_size, self.latent_size
        )
        self.T_var = nn.Linear(
            self.hidden_size + self.label_size, self.latent_size
        )
        self.G_mean = nn.Linear(
            self.hidden_size + self.label_size, self.latent_size
        )
        self.G_var = nn.Linear(
            self.hidden_size + self.label_size, self.latent_size
        )
        self.Y_mean = nn.Linear(self.hidden_size * 2, self.label_size)
        self.Y_var = nn.Linear(self.hidden_size * 2, self.label_size)

        self.pred_loss = nn.MSELoss()

    def setup_scaler(self, label_mean, label_var):
        scaler = StandardScaler()
        scaler.mean_ = np.array(label_mean)
        scaler.var_ = np.array(label_var)
        scaler.scale_ = np.sqrt(scaler.var_)
        return scaler

    def sample_prior(self, label=None, prob_decode=False):
        if label is None:
            x_tree_vecs = torch.randn(
                1, self.hidden_size + self.label_size
            ).cuda()
            x_mol_vecs = torch.randn(
                1, self.hidden_size + self.label_size
            ).cuda()
        else:
            x_tree_vecs = torch.cat(
                (torch.randn(1, self.hidden_size), label), 1
            ).cuda()
            x_mol_vecs = torch.cat(
                (torch.randn(1, self.hidden_size), label), 1
            ).cuda()

        z_tree_vecs = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs = self.rsample(x_mol_vecs, self.G_mean, self.G_var)
        return self.decode(z_tree_vecs, z_mol_vecs, prob_decode)

    def forward(self, x_batch, label, alpha, beta):
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(
            x_jtenc_holder, x_mpn_holder
        )
        y_vecs, y_kl = self.rsample(
            torch.cat((x_tree_vecs, x_mol_vecs), 1), self.Y_mean, self.Y_var
        )

        if label is None:
            normalized_label = y_vecs
            pred_loss, mae = 0, 0
        else:
            normalized_label = torch.from_numpy(
                self.scaler.transform(label.reshape(-1, 1).numpy())
            ).cuda(0)
            pred_loss = torch.mean(
                self.pred_loss(y_vecs, normalized_label)
            )
            mae = torch.mean(
                torch.abs(
                    label.reshape(-1,1)
                    - self.scaler.inverse_transform(
                        y_vecs.cpu().detach().numpy()
                    )
                )
            )
            
        z_tree_vecs, tree_kl = self.rsample(
            torch.cat((x_tree_vecs, normalized_label), 1),
            self.T_mean,
            self.T_var,
        )
        z_mol_vecs, mol_kl = self.rsample(
            torch.cat((x_mol_vecs, normalized_label), 1),
            self.G_mean,
            self.G_var,
        )

        kl_div = tree_kl + mol_kl + y_kl
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(
            x_batch, z_tree_vecs
        )
        assm_loss, assm_acc = self.assm(
            x_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess
        )

        return (
            word_loss
            + topo_loss
            + assm_loss
            + beta * kl_div
            + alpha * pred_loss,
            kl_div.item(),
            mae,
            word_loss,
            topo_loss,
            assm_loss,
            pred_loss,
            word_acc,
            topo_acc,
            assm_acc,
        )
