import sys
import math

import numpy as np
from rdkit import RDLogger, Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from toolz import partition_all
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler

import molecule_optimizer.externals.fast_jtnn as fast_jtnn
from molecule_optimizer.models.semi_jtvae import SemiJTVAE
from molecule_optimizer.runner.generator_predictor import GeneratorPredictor


class SemiJTVAEGeneratorPredictor(GeneratorPredictor):
    """
    The method class for the Semi-JTVAE algorithm

    Args:
        list_smiles (list): List of smiles in training data.
        labels (list): List of labels for smiles.
        training (boolean): If we are training (as opposed to testing).
        build_vocab (boolean): If we need to build the vocabulary (first time training with this dataset).
        device (torch.device, optional): The device where the model is deployed.
    """

    def __init__(self, list_smiles, build_vocab=True, device=None):
        super().__init__()
        if build_vocab:
            self.vocab = self.build_vocabulary(list_smiles)
        self.model = None
        self.processed_smiles, self.processed_idxs = self.preprocess(list_smiles)

    def get_model(self, task, config_dict):
        if task == "rand_gen":
            # hidden_size, latent_size, depthT, depthG
            self.vae = SemiJTVAE(
                fast_jtnn.Vocab(self.vocab), **config_dict
            ).cuda()
        else:
            raise ValueError("Task {} is not supported in JTVAE!".format(task))

            
    def _get_vocab(self, smiles):
        try:
            mol = fast_jtnn.MolTree(smiles)
            return [c.smiles for c in mol.nodes]
        except:
            return []
            
    
    def build_vocabulary(self, list_smiles):
        """
        Building the vocabulary for training.

        Args:
            list_smiles (list): the list of smiles strings in the dataset.

        :rtype:
            cset (list): A list of smiles that contains the vocabulary for the training data.

        """
        vocab_lists = []
        chunks = list(partition_all(5000, list_smiles))
            
        with Pool() as P:
            for chunk in tqdm(chunks):
                vocab_lists.extend(list(P.map(self._get_vocab, chunk)))
        
        vocabs = [vocab for vocab_list in vocab_lists for vocab in vocab_list]
        cset = set(vocabs)
    
        return list(cset)

    def _tensorize(self, smiles, assm=True):
        try:
            mol_tree = fast_jtnn.MolTree(smiles)
            mol_tree.recover()
            if assm:
                mol_tree.assemble()
                for node in mol_tree.nodes:
                    if node.label not in node.cands:
                        node.cands.append(node.label)
            del mol_tree.mol
            for node in mol_tree.nodes:
                del node.mol
            return mol_tree
        except:
            return None
        

    def preprocess(self, list_smiles):
        """
        Preprocess the molecules.

        Args:
            list_smiles (list): The list of smiles strings in the dataset.

        :rtype:
            preprocessed (list): A list of preprocessed MolTree objects.

        """
        preprocessed = []
        chunks = list(partition_all(5000, list_smiles))
        
        with Pool() as P:
            for chunk in tqdm(chunks):
                preprocessed.extend(list(P.map(self._tensorize, chunk)))
        preprocessed = np.array(preprocessed)
        
        processed_idxs = (preprocessed != None).nonzero()[0]
        processed_smiles = preprocessed[processed_idxs]
        return processed_smiles, processed_idxs
    
    def get_processed_labels(self, labels):
        return labels[self.processed_idxs]
        
    def train_gen_pred(
        self,
        loader,
        load_epoch,
        lr,
        anneal_rate,
        clip_norm,
        num_epochs,
        alpha,
        beta,
        max_beta,
        step_beta,
        anneal_iter,
        kl_anneal_iter,
        print_iter,
        save_iter,
    ):
        """
        Train the Junction Tree Variational Autoencoder for the random generation task.

        Args:
            loader (MolTreeFolder): The MolTreeFolder loader.
            load_epoch (int): The epoch to load from state dictionary.
            lr (float): The learning rate for training.
            anneal_rate (float): The learning rate annealing.
            clip_norm (float): Clips gradient norm of an iterable of parameters.
            num_epochs (int): The number of training epochs.
            beta (float): The KL regularization weight.
            max_beta (float): The maximum KL regularization weight.
            step_beta (float): The KL regularization weight step size.
            anneal_iter (int): How often to step in annealing the learning rate.
            kl_anneal_iter (int): How often to step in annealing the KL regularization weight.
            print_iter (int): How often to print the iteration statistics.
            save_iter (int): How often to save the iteration statistics.

        """
        vocab = fast_jtnn.Vocab(self.vocab)

        for param in self.vae.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

        if 0 > 0:
            self.vae.load_state_dict(
                torch.load("checkpoints" + "/model.iter-" + str(load_epoch))
            )

        print(
            "Model #Params: %dK"
            % (sum([x.nelement() for x in self.vae.parameters()]) / 1000,)
        )

        optimizer = optim.Adam(self.vae.parameters(), lr=lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, anneal_rate)
        scheduler.step()

        def param_norm(m):
            return math.sqrt(
                sum([p.norm().item() ** 2 for p in m.parameters()])
            )

        def grad_norm(m):
            return math.sqrt(
                sum(
                    [
                        p.grad.norm().item() ** 2
                        for p in m.parameters()
                        if p.grad is not None
                    ]
                )
            )

        total_step = load_epoch
        meters = np.zeros(10)

        for epoch in range(num_epochs):
            for batch in loader:
                total_step += 1

                self.vae.zero_grad()
                labelled_data = batch["labelled_data"]
                unlabelled_data = batch["unlabelled_data"]
                labels = batch["labels"]

                (
                    labelled_loss,
                    labelled_kl_div,
                    labelled_mae,
                    labelled_word_loss,
                    labelled_topo_loss,
                    labelled_assm_loss,
                    labelled_pred_loss,
                    labelled_wacc,
                    labelled_tacc,
                    labelled_sacc,
                ) = self.vae(labelled_data, labels, alpha, beta)
                (
                    unlabelled_loss,
                    unlabelled_kl_div,
                    unlabelled_mae,
                    unlabelled_word_loss,
                    unlabelled_topo_loss,
                    unlabelled_assm_loss,
                    unlabelled_pred_loss,
                    unlabelled_wacc,
                    unlabelled_tacc,
                    unlabelled_sacc,
                ) = self.vae(unlabelled_data, None, alpha, beta)

                loss = labelled_loss + unlabelled_loss
                kl_div = labelled_kl_div + unlabelled_kl_div
                mae = labelled_mae + unlabelled_mae
                word_loss = labelled_word_loss + unlabelled_word_loss
                topo_loss = labelled_topo_loss + unlabelled_topo_loss
                assm_loss = labelled_assm_loss + unlabelled_assm_loss
                pred_loss = labelled_pred_loss + unlabelled_pred_loss
                wacc = (labelled_wacc + unlabelled_wacc) / 2
                tacc = (labelled_tacc + unlabelled_tacc) / 2
                sacc = (labelled_sacc + unlabelled_sacc) / 2

                loss.backward()
                nn.utils.clip_grad_norm_(self.vae.parameters(), clip_norm)
                optimizer.step()

                meters = meters + np.array(
                    [loss.detach().cpu(), kl_div, mae/print_iter, word_loss.detach().cpu(), topo_loss.detach().cpu(), assm_loss.detach().cpu(), pred_loss.detach().cpu(), wacc*100/print_iter, tacc*100/print_iter, sacc*100/print_iter]
                )

                if total_step % print_iter == 0:
                    meters /= 50
                    print(
                        "[%d] Alpha: %.3f, Beta: %.3f, Loss: %.2f, KL: %.2f, MAE: %.5f, Word Loss: %.2f, Topo Loss: %.2f, Assm Loss: %.2f, Pred Loss: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f"
                        % (
                            total_step,
                            alpha,
                            beta,
                            meters[0],
                            meters[1],
                            meters[2],
                            meters[3],
                            meters[4],
                            meters[5],
                            meters[6],
                            meters[7],
                            meters[8],
                            meters[9],
                            param_norm(self.vae),
                            grad_norm(self.vae),
                        )
                    )
                    sys.stdout.flush()
                    meters *= 0

                if total_step % save_iter == 0:
                    torch.save(
                        self.vae.state_dict(),
                        "saved" + "/model.iter-" + str(total_step),
                    )

                if total_step % anneal_iter == 0:
                    scheduler.step()
                    print("learning rate: %.6f" % scheduler.get_lr()[0])

                if (
                    total_step % kl_anneal_iter == 0
                    and total_step >= anneal_iter
                ):
                    beta = min(max_beta, beta + step_beta)

    def run_rand_gen(self, num_samples):
        """
        Sample new molecules from the trained model.

        Args:
            num_samples (int): Number of samples to generate from the trained model.

        :rtype:
            samples (list): samples is a list of generated molecules.

        """
        torch.manual_seed(0)
        samples = [self.vae.sample_prior() for _ in range(num_samples)]
        return samples
