import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import scanpy as sc
import anndata as ad
from .TOSICA_model import scTrans_model as create_model


def todense(adata):
    """Convert adata.X to dense numpy array (handles sparse CSR/CSC matrices)."""
    import scipy
    if isinstance(adata.X, scipy.sparse.csr_matrix) or isinstance(adata.X, scipy.sparse.csc_matrix):
        return adata.X.todense()
    else:
        return adata.X


def get_weight(att_mat, pathway):
    """Attention rollout (Abnar & Zuidema, 2020): trace attention flow from CLS token
    through all transformer layers to input pathway tokens.

    Returns a single-row DataFrame with columns = pathway names and values =
    cumulative attention each pathway received from the CLS token.
    """
    # Stack per-layer attention matrices → (n_layers, heads, seq, seq)
    att_mat = torch.stack(att_mat).squeeze(1)
    # Average across attention heads → (n_layers, seq, seq)
    att_mat = torch.mean(att_mat, dim=1)
    # Add identity to model skip connections (each token always attends to itself)
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    # Re-normalize rows so each sums to 1
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    # Recursively multiply attention matrices across layers to get total attention flow
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # Extract final layer's CLS→pathway attention (skip CLS→CLS at index 0)
    v = joint_attentions[-1]
    v = pd.DataFrame(v[0,1:].detach().numpy()).T
    v.columns = pathway
    return v


def prediect(adata, model_weight_path, project, mask_path, laten=False,
             save_att='X_att', save_lantent='X_lat', n_step=10000,
             cutoff=0.1, n_unannotated=1, batch_size=50, embed_dim=48,
             depth=2, num_heads=4):
    """TOSICA inference pipeline: load trained model, run prediction on all cells
    in adata, and return a new AnnData with attention weights (or latent embeddings)
    and predicted cell type labels.

    Parameters
    ----------
    adata : AnnData
        Input gene expression data (cells x genes).
    model_weight_path : str
        Path to saved model checkpoint (.pth).
    project : str
        Project directory name (contains mask.npy, pathway.csv, label_dictionary.csv).
    mask_path : str
        Path to the binary gene-pathway mask (n_genes x n_pathways).
    laten : bool
        If True, output CLS embeddings; if False (default), output attention weights.
    save_att, save_lantent : str
        Unused parameters (kept for API compatibility).
    n_step : int
        Number of cells to process per chunk (for memory management).
    cutoff : float
        Minimum softmax probability to assign a class; below this → "Unknown".
    n_unannotated : int
        Number of trailing "catch-all" pathway tokens to exclude from output.
    batch_size : int
        Mini-batch size for DataLoader.
    embed_dim, depth, num_heads : int
        Model architecture hyperparameters (must match the saved checkpoint).

    Returns
    -------
    AnnData with X = attention weights or latent embeddings, obs = original
    metadata + 'Prediction' (cell type name) + 'Probability' (confidence).
    """

    # ── Phase 1: Setup ──────────────────────────────────────────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    num_genes = adata.shape[1]

    # Load the binary mask (n_genes x n_pathways): which genes belong to which pathway
    mask = np.load(mask_path)
    project_path = os.getcwd()+'/%s'%project

    # Load pathway names and cell type label dictionary
    pathway = pd.read_csv(project_path+'/pathway.csv', index_col=0)
    dictionary = pd.read_table(project_path+'/label_dictionary.csv', sep=',',header=0,index_col=0)
    n_c = len(dictionary)                        # number of known cell type classes
    label_name = dictionary.columns[0]           # e.g. 'Celltype'
    dictionary.loc[(dictionary.shape[0])] = 'Unknown'  # append Unknown as class index n_c
    # Build integer → cell type name lookup: {0: 'T cell', 1: 'B cell', ..., n_c: 'Unknown'}
    dic = {}
    for i in range(len(dictionary)):
        dic[i] = dictionary[label_name][i]

    # ── Phase 2: Model creation and weight loading ──────────────────────
    model = create_model(num_classes=n_c, num_genes=num_genes,mask = mask, has_logits=False,depth=depth,num_heads=num_heads).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()  # disable dropout for inference

    # ── Phase 3: Gene-to-token importance extraction ────────────────────
    # Extract the masked linear layer weights to see which genes drive each pathway token
    parm={}
    for name,parameters in model.named_parameters():
        parm[name]=parameters.detach().cpu().numpy()
    gene2token = parm['feature_embed.fe.weight']
    # Reshape flat weight (n_pathways*embed_dim, n_genes) → (n_pathways, embed_dim, n_genes)
    gene2token = gene2token.reshape((int(gene2token.shape[0]/embed_dim),embed_dim,adata.shape[1]))
    gene2token = abs(gene2token)                 # absolute value = magnitude of contribution
    gene2token = np.max(gene2token,axis=1)       # max across embed dims → (n_pathways, n_genes)
    gene2token = pd.DataFrame(gene2token)
    gene2token.columns=adata.var_names           # columns = gene names
    gene2token.index = pathway['0']              # rows = pathway names
    gene2token.to_csv(project_path+'/gene2token_weights.csv')  # save for downstream analysis

    # ── Phase 4: Initialize accumulators ────────────────────────────────
    # NOTE: these are overwritten every batch; lines 75-81 are effectively dead code
    # from an earlier version that tried to accumulate across batches
    latent = torch.empty([0,embed_dim]).cpu()
    att = torch.empty([0,(len(pathway))]).cpu()
    predict_class = np.empty(shape=0)
    pre_class = np.empty(shape=0)
    latent = torch.squeeze(latent).cpu().numpy()
    l_p = np.c_[latent, predict_class,pre_class]
    att = np.c_[att, predict_class,pre_class]

    all_line = adata.shape[0]                    # total number of cells
    n_line = 0                                   # current cell index pointer
    adata_list = []                              # collect per-batch AnnData results

    # ── Phase 5: Chunked inference loop ─────────────────────────────────
    # Process cells in chunks of n_step to avoid OOM
    while (n_line) <= all_line:
        # Edge case: if remaining cells would produce a final batch of exactly 1 sample,
        # skip 2 cells to avoid BatchNorm/LayerNorm issues with batch_size=1
        if (all_line-n_line)%batch_size != 1:
            # Normal case: take up to n_step cells
            expdata = pd.DataFrame(todense(adata[n_line:n_line+min(n_step,(all_line-n_line))]),index=np.array(adata[n_line:n_line+min(n_step,(all_line-n_line))].obs_names).tolist(), columns=np.array(adata.var_names).tolist())
            print(n_line)
            n_line = n_line+n_step
        else:
            # Batch-size-1 avoidance: take 2 fewer cells
            # BUG: n_line is set to a relative value, not an absolute position
            expdata = pd.DataFrame(todense(adata[n_line:n_line+min(n_step,(all_line-n_line-2))]),index=np.array(adata[n_line:n_line+min(n_step,(all_line-n_line-2))].obs_names).tolist(), columns=np.array(adata.var_names).tolist())
            n_line = (all_line-n_line-2)
            print(n_line)

        # Convert chunk to tensor and wrap in DataLoader for mini-batching
        expdata = np.array(expdata)
        expdata = torch.from_numpy(expdata.astype(np.float32))
        data_loader = torch.utils.data.DataLoader(expdata,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True)

        # ── Phase 6: Mini-batch inference ───────────────────────────────
        with torch.no_grad():
            # predict class
            for step, data in enumerate(data_loader):
                #print(step)
                exp = data
                # Forward pass returns: latent (CLS embedding), logits, attention weights
                lat, pre, weights = model(exp.to(device))

                # ── Phase 7: Softmax + unknown thresholding ─────────────
                pre = torch.squeeze(pre).cpu()
                pre = F.softmax(pre,1)           # logits → class probabilities
                predict_class = np.empty(shape=0)
                pre_class = np.empty(shape=0)
                for i in range(len(pre)):
                    if torch.max(pre, dim=1)[0][i] >= cutoff:
                        # Confident prediction: assign the argmax class
                        predict_class = np.r_[predict_class,torch.max(pre, dim=1)[1][i].numpy()]
                    else:
                        # Low confidence: assign "Unknown" (index n_c)
                        predict_class = np.r_[predict_class,n_c]
                    # Store the max probability as a confidence score
                    pre_class = np.r_[pre_class,torch.max(pre, dim=1)[0][i]]

                # ── Phase 8: Build per-batch AnnData ────────────────────
                l_p = torch.squeeze(lat).cpu().numpy()   # CLS embeddings
                att = torch.squeeze(weights).cpu().numpy()  # attention weights
                # Metadata: predicted class index + confidence for each cell
                meta = np.c_[predict_class,pre_class]
                meta = pd.DataFrame(meta)
                meta.columns = ['Prediction','Probability']
                meta.index = meta.index.astype('str')
                if laten:
                    # Output mode: CLS token embeddings (for UMAP, clustering, etc.)
                    l_p = l_p.astype('float32')
                    new = sc.AnnData(l_p, obs=meta)
                else:
                    # Output mode: attention weights per pathway (default)
                    # Trim off n_unannotated trailing catch-all tokens (not real pathways)
                    att = att[:,0:(len(pathway)-n_unannotated)]
                    att = att.astype('float32')
                    varinfo = pd.DataFrame(pathway.iloc[0:len(pathway)-n_unannotated,0].values,index=pathway.iloc[0:len(pathway)-n_unannotated,0],columns=['pathway_index'])
                    new = sc.AnnData(att, obs=meta, var = varinfo)
                adata_list.append(new)

    # ── Phase 9: Concatenate all batches and map labels ─────────────────
    print(all_line)
    new = ad.concat(adata_list)                  # stack all batch results
    new.obs.index = adata.obs.index              # restore original cell barcodes
    # Convert integer predictions → human-readable cell type names
    new.obs['Prediction'] = new.obs['Prediction'].map(dic)
    # Copy over all original metadata columns (donor, batch, etc.)
    new.obs[adata.obs.columns] = adata.obs[adata.obs.columns].values
    return(new)
