#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`Autoencoder Time Series Clustering<src.ml.cluster.autoencoder>` module.

Module Description
==================

Module for clustering time series samples using an AutoEncoder.

.. moduleauthor:: David Grethlein

Module Contents
===============

"""


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import traceback

from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn

from ...utils.misc import dbg, err


#==============================================================================
#               TIME SERIES ENCODER CLASS
#==============================================================================
class Encoder(nn.Module):
    def __init__(self,
                 ts_len        : int,
                 num_features  : int,
                 embedding_dim : int):
        """_summary_

        Args:
            ts_len (int): _description_
            num_features (int): _description_
            embedding_dim (int): _description_
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=embedding_dim,
                            num_layers=1,
                            batch_first=True)


    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # x shape: (batch_size, ts_len, num_features)
        _, (hidden, _) = self.lstm(x)

        # hidden shape: (1, batch_size, embedding_dim)
        # squeezed shape: (batch_size, embedding_dim)
        return hidden.squeeze(0)

#==============================================================================
#               TIME SERIES DECODER CLASS
#==============================================================================
class Decoder(nn.Module):
    def __init__(self,
                 ts_len        : int,
                 num_features  : int,
                 embedding_dim : int):
        """_summary_

        Args:
            ts_len (int): _description_
            num_features (int): _description_
            embedding_dim (int): _description_
        """
        super().__init__()
        self.ts_len = ts_len
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=embedding_dim,
                            num_layers=1,
                            batch_first=True)
        self.output_layer = nn.Linear(embedding_dim, num_features)


    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # x shape: (batch_size, embedding_dim)
        # Replicate the latent vector for each timestep (RepeatVector logic)
        x = x.unsqueeze(1).repeat(1, self.ts_len, 1)
        x, _ = self.lstm(x)
        # output shape: (batch_size, ts_len, num_features)
        return self.output_layer(x)


#==============================================================================
#               TIME SERIES AUTOENCODER CLASS
#==============================================================================
class LSTMAutoencoder(nn.Module):
    def __init__(self,
                 ts_len        : int,
                 num_features  : int,
                 embedding_dim : int):
        """_summary_

        Args:
            ts_len (int): _description_
            num_features (int): _description_
            embedding_dim (int): _description_
        """
        super().__init__()
        self.encoder = Encoder(ts_len,
                               num_features,
                               embedding_dim)

        self.decoder = Decoder(ts_len,
                               num_features,
                               embedding_dim)


    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#==============================================================================
#       RUNNING AUTOENCODER TIME SERIES CLUSTERING FUNCTION(s)
#==============================================================================
def run_autoencoder_ts_clustering(ts_dfs        : List[pd.DataFrame],
                                  batch_size    : int,
                                  embedding_dim : int,
                                  random_seed   : int = 0,
                                  verbose       : bool = False):
    """_summary_

    Args:
        ts_dfs (List[pd.DataFrame]): _description_
        batch_size (int): _description_
        embedding_dim (int): _description_
        random_seed (int, optional): _description_. Defaults to 0.
        verbose (bool, optional): _description_. Defaults to False.
    """
    try:
        # Seeding the random number generators at work for weights initialization.
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Converts list of time series DataFrame(s) to numpy array.
        X = np.array([df.to_numpy() for df in ts_dfs]).reshape(len(ts_dfs),
                                                               len(ts_dfs[0]),
                                                               1)

        if verbose:
            print(f"\n// {dbg()}  Time series data converted to numpy array of shape: "
                  + f"{X.shape}\n")

        # Converts numpy array to pytorch tensor.
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Hyperparameters to be logged in mlflow.
        config = {"num_samples"   : len(ts_dfs),
                  "ts_len"        : len(ts_dfs[0]),
                  "num_features"  : 1,
                  "embedding_dim" : embedding_dim,
                  "lr"            : 0.005,
                  "epochs"        : 80,
                  "batch_size"    : batch_size,
                  "num_clusters"  : 9,
                  "random_seed"   : random_seed}

        # Start running a named experiment in the MLFlow experiment tracking system.
        mlflow.set_experiment("TS_LSTM_Clustering")
        with mlflow.start_run(log_system_metrics=False):

            # Log the hyperparameters used
            mlflow.log_params(config)

            # Initialize the autoencoder model, loss function, and optimizer
            model = LSTMAutoencoder(ts_len=len(ts_dfs[0]),
                                    num_features=1,
                                    embedding_dim=embedding_dim)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

            # Model training loop
            model.train()
            for epoch in range(config["epochs"]):
                permutation = torch.randperm(X_tensor.size(0))
                epoch_loss = 0

                # Running randomly sampled batches of time series samples within each epoch.
                for i in range(0, X_tensor.size(0), config["batch_size"]):
                    indices = permutation[i:i+config["batch_size"]]
                    batch_x = X_tensor[indices]

                    optimizer.zero_grad()
                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_x)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * batch_x.size(0)

                avg_epoch_loss = epoch_loss / config["num_samples"]

                mlflow.log_metric("recon_loss", avg_epoch_loss, step=epoch)

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"// {dbg()}  Epoch[{epoch+1}] loss : {avg_epoch_loss:.4f}")

            model.eval()
            with torch.no_grad():
                latent_embeddings = model.encoder(X_tensor).numpy()

            # Execute K-Means clustering
            kmeans = KMeans(n_clusters=config["num_clusters"],
                            random_state=random_seed)
            cluster_labels = kmeans.fit_predict(latent_embeddings)

            # Compute and log clustering quality metrics
            sil_score = silhouette_score(latent_embeddings, cluster_labels)
            mlflow.log_metric("silhouette_score", sil_score)

            # Log the clustering distribution as metadata
            for idx in range(config["num_clusters"]):
                count = int(np.sum(cluster_labels == idx))
                mlflow.log_metric(f"cluster_{idx}_count", count)

            fig, ax = plt.subplots(figsize=(10,4))
            for cluster_id in range(config["num_clusters"]):
                cluster_samples = X[cluster_labels == cluster_id]
                mean_pattern = np.mean(cluster_samples, axis=0).squeeze()
                ax.plot(mean_pattern, label=f"Cluster[{cluster_id}]")

            ax.set_title("Discovered Clusters vis LSTM Autoencoder")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("kwh_electricity_consumed")
            ax.legend()

            plot_path = "./results/LSTM_autoencoder_cluster_patterns.png"
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)
            os.remove(plot_path)

            mlflow.pytorch.log_model(pytorch_model=model, 
                                     name="LSTM_Autoencoder",
                                     serialization_format="pickle")


    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't run autoencoder time series clustering!\n")
        traceback.print_exc()


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
