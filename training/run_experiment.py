%load_ext autoreload
%autoreload 2
import os
import json
import rdkit
import torch
import pickle 
import argparse
import numpy as np

from dig.ggraph.dataset import ZINC250k, ZINC800
from molecule_optimizer.externals.fast_jtnn.datautils import SemiMolTreeFolder
from molecule_optimizer.runner.semi_jtvae import SemiJTVAEGeneratorPredictor
from torch_geometric.data import DenseDataLoader

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)


def _setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    """
    Run an experiment.
    Sample command:
    ```
    python training/run_experiment.py --config_path=configs/rand_gen_zinc250k_config_dict.json
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    print(args.config_path)
    
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    
    conf = json.load(open(args.config_path))

    if False:
        print("Processing Dataset...")
        _ = ZINC250k(
            root=conf["data"]["root"],
            one_shot=False,
            use_aug=False,
        )

    zinc_250_jt = torch.load(
        os.path.join(conf["data"]["root"], conf["data"]["processed_path"])
    )
    smiles = zinc_250_jt[-1]
    labels = zinc_250_jt[0].y

    if False:
        runner = SemiJTVAEGeneratorPredictor(smiles, labels)
        runner.get_model(
            "rand_gen",
            {
                "hidden_size": conf["model"]["hidden_size"],
                "latent_size": conf["model"]["latent_size"],
                "depthT": conf["model"]["depthT"],
                "depthG": conf["model"]["depthG"],
                "label_size": 1,
                "label_mean": float(torch.mean(labels)),
                "label_var": float(torch.var(labels)),
            },
        )
        with open('runner.pickle', 'wb') as f: 
            pickle.dump(runner, f)

    with open('runner.pickle', 'rb') as f: 
    runner = pickle.load(f)

    loader = SemiMolTreeFolder(
        runner.smiles,
        runner.labels,
        runner.vocab,
        conf["batch_size"],
        num_workers=conf["num_workers"],
    )

    print("Training model...")
    runner.train_gen_pred(
        loader=loader,
        load_epoch=0,
        lr=conf["lr"],
        anneal_rate=conf["anneal_rate"],
        clip_norm=conf["clip_norm"],
        num_epochs=conf["num_epochs"],
        alpha=conf["alpha"],
        beta=conf["beta"],
        max_beta=conf["max_beta"],
        step_beta=conf["step_beta"],
        anneal_iter=conf["anneal_iter"],
        kl_anneal_iter=conf["kl_anneal_iter"],
        print_iter=conf["print_iter"],
        save_iter=conf["save_iter"],
    )

if __name__ == '__main__':
    main()
