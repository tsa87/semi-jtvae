import os
import json
import rdkit
import torch
import pickle 
import argparse
import numpy as np
import pandas as pd
import math 

from molecule_optimizer.externals.fast_jtnn.datautils import SemiMolTreeFolder, SemiMolTreeFolderTest
from molecule_optimizer.runner.semi_jtvae import SemiJTVAEGeneratorPredictor
from torch_geometric.data import DenseDataLoader

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1)
torch.manual_seed(1)

#N_TEST = 10000
N_TEST = 200
VAL_FRAC = 0.05


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
    
    cont = False
    chem_prop = "LogP"

    parser = _setup_parser()
    args = parser.parse_args()
    print(args.config_path)
    
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    
    conf = json.load(open(args.config_path))

    csv = pd.read_csv("~/scratch/ZINC_310k.csv")

    smiles = csv['SMILES']
    smiles = smiles[:60000]

    # if 'runner.xml' not in os.listdir("."):
    #     runner = SemiJTVAEGeneratorPredictor(smiles)
    #     with open('runner.xml', 'wb') as f:
    #         pickle.dump(runner, f)



    if cont == True:
        with open("saved/runner_20_" + chem_prop + "_50_1.xml", 'rb') as f: 
            runner = pickle.load(f)
            
    else:
        
        if "saved/ runner_20_" + chem_prop + "_50_1.xml" not in os.listdir("."):
            runner = SemiJTVAEGeneratorPredictor(smiles)
            with open("runner_20_" + chem_prop + "_50_1.xml", 'wb') as f:
                pickle.dump(runner, f)
    
    labels = torch.tensor(csv[chemprop][:60000]).float()
    
    #labels = torch.tensor(csv[chemprop]).float()

    runner.get_model( "rand_gen",{
        "hidden_size": conf["model"]["hidden_size"],
        "latent_size": conf["model"]["latent_size"],
        "depthT": conf["model"]["depthT"],
        "depthG": conf["model"]["depthG"],
        "label_size": 1,
        "label_mean": float(torch.mean(labels)),
        "label_var": float(torch.var(labels)),
    },)

    labels = runner.get_processed_labels(labels)
    preprocessed = runner.processed_smiles
    
    if cont == True:
        
        L_train = torch.load("L_train_" + chem_prop + "_50_1.pt")
        L_test = torch.load("L_test_" + chem_prop + "_50_1.pt")
        L_Val = torch.load("L_Val_" + chem_prop + "_50_1.pt")
        
        with open("train_" + chem_prop + "_50_1.npy", 'rb') as f:
            X_train = np.load(f, allow_pickle=True)

        with open("test_" + chem_prop + "_50_1.npy", 'rb') as f:
            X_test = np.load(f, allow_pickle=True)

        with open("validation_" + chem_prop + "_50_1.npy", 'rb') as f:
            X_Val = np.load(f, allow_pickle=True)
            
    else:
        
        perm_id=np.random.permutation(len(labels))
        X_train = preprocessed[perm_id[N_TEST:]]
        L_train = torch.tensor(labels.numpy()[perm_id[N_TEST:]])


        X_test = preprocessed[perm_id[:N_TEST]]
        L_test = torch.tensor(labels.numpy()[perm_id[:N_TEST]])

        val_cut = math.floor(len(X_train) * VAL_FRAC)

        X_Val = X_train[:val_cut]
        L_Val = L_train[:val_cut]

        X_train = X_train[val_cut :]
        L_train = L_train[val_cut :]

        with open("train_" + chem_prop + "_50_1.npy", 'wb') as f:
            np.save(f, X_train)

        with open("test_" + chem_prop + "_50_1.npy", 'wb') as f:
            np.save(f, X_test)

        with open("validation_" + chem_prop + "_50_1.npy", 'wb') as f:
            np.save(f, X_Val)
            
        torch.save(L_train, "L_train_" + chem_prop + "_50_1.pt")

        torch.save(L_test, "L_test_" + chem_prop + "_50_1.pt")

        torch.save(L_Val, "L_Val_" + chem_prop + "_50_1.pt")
        
    print("Training model...")
    runner.train_gen_pred(
    X_train,
    L_train,
    X_test,
    L_test,
    X_Val,
    L_Val,
    load_epoch = 0,
    lr=conf["lr"],
    anneal_rate=conf["anneal_rate"],
    clip_norm=conf["clip_norm"],
    num_epochs=conf["num_epochs"],
    alpha=conf["alpha"],
    max_alpha=conf["max_alpha"],
    step_alpha=conf["step_alpha"],
    beta=conf["beta"],
    max_beta=conf["max_beta"],
    step_beta=conf["step_beta"],
    anneal_iter=conf["anneal_iter"],
    alpha_anneal_iter=conf["alpha_anneal_iter"],
    kl_anneal_iter=conf["kl_anneal_iter"],
    print_iter=conf["print_iter"],
    save_iter=conf["save_iter"],
    batch_size=conf["batch_size"],
    num_workers=conf["num_workers"],
    label_pct=0.5,
    chem_prop = chem_prop
    )

if __name__ == '__main__':
    main()
