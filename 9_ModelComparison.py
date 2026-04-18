import math
import sys

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from torch.fx import Transformer
from allen_brain.TOSICA.train import balance_populations, set_seed, read_gmt, create_pathway_mask, get_gmt, MyDataSet, create_model
from allen_brain.TOSICA import TOSICA_model

# My implemented models
from allen_brain.models.CellTypeAttention import TOSICA as my_implementation_TOSICA
from allen_brain.models.CellTypeCNN import CellTypeCNN
from allen_brain.models.CellTypeGNN import CellTypeGNN, build_knn_edges
from allen_brain.models.CellTypeMLP import MLP_Model
from allen_brain.TOSICA.train import todense
# Torch imports
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
# Other imports
from tqdm import tqdm
import time
import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc


CONFIG = {
    'seed': 1,
    'n_genes': 2000,
    'n_pathways': 200,
    'n_classes': 75,
    'dropout': 0.1,
    'hvg': 10_000,
    'max_g': 300,
    'max_gs': 300,
    'mask_ratio': 0.015,
    'n_unannotated': 1,
    'batch_size': 64,
    'embed_dim': 48,
    'depth': 2,
    'num_heads': 4,
    'lr': 0.001,
    'epochs': 10,
    'lrf': 0.01,
    'n_step': 10000
    
}

PROJECT ={
    'name': 'TOSICA_comparison',
    'model_types': ['TOSICA', 'my_TOSICA', 'MLP', 'GNN', 'CNN'],
    "mask_path": 'TOSICA_comparison/mask.npy',
    "pathway_path": 'TOSICA_comparison/pathway.csv',
    "label_dictionary_path": 'TOSICA_comparison/label_dictionary.csv',
    "model_weight_path": 'TOSICA_comparison/{}-{}.pth',
}


PRE_CONFIG = {
   'laten': False, 'save_att': 'X_att',
   'save_lantent': 'X_lat', 'n_step': 10000, 
   'cutoff': 0.1, 'n_unannotated': 1,
   'batch_size': 50, 'embed_dim': 48, 
   'depth': 2, 'num_heads': 4
}
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'




MODELS = {
    'TOSICA': TOSICA_model.Transformer,
    'my_TOSICA': my_implementation_TOSICA,
    'MLP': MLP_Model,
    'GNN': CellTypeGNN, 
    'CNN': CellTypeCNN
}

def split_dataset(adata, label_name='subclass_label', train_ratio=0.7, val_ratio=0.15):
    """Split adata into train / val / test sets (default 70/15/15)."""
    label_encoder = LabelEncoder()
    genes = np.array(adata.var_names)

    labels = label_encoder.fit_transform(adata.obs[label_name].astype('str').values)
    inverse = label_encoder.inverse_transform(range(labels.max() + 1))

    X = np.asarray(todense(adata), dtype=np.float32)

    # Append labels as last column for balance_populations
    el_data = np.column_stack([X, labels.astype(np.float32)])
    del X
    el_data = balance_populations(data=el_data)

    n_genes = el_data.shape[1] - 1
    n = len(el_data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    print(f"Total samples: {n}, genes: {n_genes}, classes: {len(inverse)}")

    # Shuffle indices instead of copying data through random_split
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_idx)}")

    exp_train = torch.from_numpy(el_data[train_idx, :n_genes])
    label_train = torch.from_numpy(el_data[train_idx, -1].astype(np.int64))
    exp_val = torch.from_numpy(el_data[val_idx, :n_genes])
    label_val = torch.from_numpy(el_data[val_idx, -1].astype(np.int64))
    exp_test = torch.from_numpy(el_data[test_idx, :n_genes])
    label_test = torch.from_numpy(el_data[test_idx, -1].astype(np.int64))

    return exp_train, label_train, exp_val, label_val, exp_test, label_test, inverse, genes


def _forward_model(model, exp, model_type, edge_index=None):
    """Call model.forward(), returning only the logits regardless of model type."""
    if model_type == 'TOSICA':
        _, pred, _ = model(exp)
        return pred
    elif model_type == 'CNN':
        return model(exp.unsqueeze(1))
    elif model_type == 'GNN':
        return model(exp, edge_index)
    else:
        # my_TOSICA, MLP — all return logits directly
        return model(exp)


def _train_epoch(model, optimizer, data_loader, device, epoch, model_type):
    """Train one epoch, handling different model forward signatures."""
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, label = data
        sample_num += exp.shape[0]
        pred = _forward_model(model, exp.to(device), model_type)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, label.to(device)).sum()
        loss = loss_fn(pred, label.to(device))
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def _eval_epoch(model, data_loader, device, epoch, model_type):
    """Evaluate one epoch, handling different model forward signatures."""
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, labels = data
        sample_num += exp.shape[0]
        pred = _forward_model(model, exp.to(device), model_type)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_fn(pred, labels.to(device))
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def _train_graph_epoch(model, graph_data, optimizer, device):
    """Train one epoch for GNN using full-graph forward."""
    model.train()
    optimizer.zero_grad()
    logits = model(graph_data.x, graph_data.edge_index)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits[graph_data.train_mask], graph_data.y[graph_data.train_mask])
    loss.backward()
    optimizer.step()
    acc = (logits[graph_data.train_mask].argmax(1) == graph_data.y[graph_data.train_mask]).float().mean().item()
    return loss.item(), acc


@torch.no_grad()
def _eval_graph_epoch(model, graph_data, device):
    """Evaluate one epoch for GNN using full-graph forward."""
    model.eval()
    logits = model(graph_data.x, graph_data.edge_index)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits[graph_data.val_mask], graph_data.y[graph_data.val_mask])
    acc = (logits[graph_data.val_mask].argmax(1) == graph_data.y[graph_data.val_mask]).float().mean().item()
    return loss.item(), acc


def fit_model(adata, gmt_path, project=None, pre_weights='', label_name='subclass_label',
              max_g=300, max_gs=300, mask_ratio=0.015, n_unannotated=1, batch_size=8,
              embed_dim=48, depth=2, num_heads=4, lr=0.001, epochs=10, lrf=0.01,
              model_type='TOSICA'):
    set_seed(CONFIG['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    today = time.strftime('%Y%m%d', time.localtime(time.time()))
    project = project or gmt_path.replace('.gmt', '') + '_%s' % today
    project_path = os.getcwd() + '/%s' % project
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    tb_writer = SummaryWriter()
    # check if split data exist
    splits_exist = all(os.path.exists(project_path + f'/{f}.npy')
                       for f in ('exp_train', 'label_train', 'exp_val', 'label_val',
                                 'exp_test', 'label_test'))
    if splits_exist:
        exp_train = np.load(project_path + '/exp_train.npy')
        label_train = np.load(project_path + '/label_train.npy')
        exp_val = np.load(project_path + '/exp_val.npy')
        label_val = np.load(project_path + '/label_val.npy')
        exp_test = np.load(project_path + '/exp_test.npy')
        label_test = np.load(project_path + '/label_test.npy')
        inverse = pd.read_csv(project_path + '/label_dictionary.csv', index_col=0).values.flatten()
        print('Split data loaded!')
    else:
        print('Split data not found, creating new split...')
        exp_train, label_train, exp_val, label_val, exp_test, label_test, inverse, genes = \
            split_dataset(adata, label_name)
        np.save(project_path + '/exp_train.npy', exp_train)
        np.save(project_path + '/label_train.npy', label_train)
        np.save(project_path + '/exp_val.npy', exp_val)
        np.save(project_path + '/label_val.npy', label_val)
        np.save(project_path + '/exp_test.npy', exp_test)
        np.save(project_path + '/label_test.npy', label_test)
        pd.DataFrame(inverse, columns=[label_name]).to_csv(project_path + '/label_dictionary.csv', quoting=None)
        print('Split data created and saved!')
    if gmt_path is None:
        mask = np.random.binomial(1, mask_ratio, size=(len(genes), max_gs))
        pathway = list()
        for i in range(max_gs):
            x = 'node %d' % i
            pathway.append(x)
        print('Full connection!')
    else:
        if '.gmt' in gmt_path:
            gmt_path = gmt_path
        else:
            gmt_path = get_gmt(gmt_path)

    if not os.path.exists(project_path + '/mask.npy') or not os.path.exists(project_path + '/pathway.csv'):
        reactome_dict = read_gmt(gmt_path, min_g=0, max_g=max_g)
        mask, pathway = create_pathway_mask(feature_list=genes,
                                            dict_pathway=reactome_dict,
                                            add_missing=n_unannotated,
                                            fully_connected=True)
        pathway = pathway[np.sum(mask, axis=0) > 4]
        mask = mask[:, np.sum(mask, axis=0) > 4]
        pathway = pathway[sorted(np.argsort(np.sum(mask, axis=0))[-min(max_gs, mask.shape[1]):])]
        mask = mask[:, sorted(np.argsort(np.sum(mask, axis=0))[-min(max_gs, mask.shape[1]):])]
        np.save(PROJECT.get('mask_path'), mask)
        pd.DataFrame(pathway).to_csv(PROJECT.get('pathway_path'))
        pd.DataFrame(inverse, columns=[label_name]).to_csv(PROJECT.get('label_dictionary_path'), quoting=None)
        print('Mask created and saved!')
    else:
        mask = np.load(PROJECT.get('mask_path'))
        pathway = pd.read_csv(PROJECT.get('pathway_path'), index_col=0)['0'].tolist()
        print('Mask loaded!')

    num_genes = len(exp_train[0])
    num_classes = np.int64(torch.max(torch.tensor(label_train)) + 1)

    #  Build model 
    if model_type == 'TOSICA':
        model = create_model(num_classes=num_classes, num_genes=num_genes,
                             mask=mask, embed_dim=embed_dim, depth=depth,
                             num_heads=num_heads, has_logits=False).to(device)
    elif model_type == 'my_TOSICA':
        mask_tensor = torch.from_numpy(mask).float() if isinstance(mask, np.ndarray) else mask.float()
        model = my_implementation_TOSICA(
            n_genes=num_genes, n_pathways=mask.shape[1], n_classes=num_classes,
            mask=mask_tensor, embed_dim=embed_dim, n_heads=num_heads,
            n_layers=depth).to(device)
    elif model_type == 'MLP':
        model = MLP_Model(input_dim=num_genes, n_classes=num_classes).to(device)
    elif model_type == 'CNN':
        model = CellTypeCNN(seq_len=num_genes, n_classes=num_classes).to(device)
    elif model_type == 'GNN':
        model = CellTypeGNN(in_dim=num_genes, hidden_dim=embed_dim,
                            n_classes=num_classes).to(device)
    else:
        raise ValueError('Invalid model type: {}'.format(model_type))

    if pre_weights != "":
        assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
        preweights_dict = torch.load(pre_weights, map_location=device)
        print(model.load_state_dict(preweights_dict, strict=False))
    print('Model built!')

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    if model_type == 'GNN':
        # GNN uses full-graph training, not mini-batch DataLoaders
        exp_train_np = exp_train.numpy() if isinstance(exp_train, torch.Tensor) else exp_train
        exp_val_np = exp_val.numpy() if isinstance(exp_val, torch.Tensor) else exp_val
        label_train_t = torch.tensor(label_train).long() if not isinstance(label_train, torch.Tensor) else label_train.long()
        label_val_t = torch.tensor(label_val).long() if not isinstance(label_val, torch.Tensor) else label_val.long()
        X_all = np.vstack([exp_train_np, exp_val_np]).astype(np.float32)
        y_all = torch.cat([label_train_t, label_val_t])
        n_train = len(exp_train_np)
        n_total = len(X_all)
        train_mask = torch.zeros(n_total, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask = ~train_mask
        edge_index = build_knn_edges(X_all, k=15)
        graph_data = Data(
            x=torch.from_numpy(X_all),
            edge_index=edge_index,
            y=y_all,
            train_mask=train_mask,
            val_mask=val_mask
        ).to(device)
        for epoch in range(epochs):
            train_loss, train_acc = _train_graph_epoch(model, graph_data, optimizer, device)
            scheduler.step()
            val_loss, val_acc = _eval_graph_epoch(model, graph_data, device)
            print("[epoch {}] train_loss: {:.3f}, train_acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}".format(
                epoch, train_loss, train_acc, val_loss, val_acc))
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
            torch.save(model.state_dict(), project_path + "/{}-{}.pth".format(model_type, epoch))
    else:
        train_dataset = MyDataSet(exp_train, label_train)
        val_dataset = MyDataSet(exp_val, label_val)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, pin_memory=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                   shuffle=False, pin_memory=True, drop_last=True)
        for epoch in range(epochs):
            train_loss, train_acc = _train_epoch(model=model, optimizer=optimizer,
                                                 data_loader=train_loader, device=device,
                                                 epoch=epoch, model_type=model_type)
            scheduler.step()
            val_loss, val_acc = _eval_epoch(model=model, data_loader=valid_loader,
                                            device=device, epoch=epoch, model_type=model_type)
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
            torch.save(model.state_dict(), project_path + "/{}-{}.pth".format(model_type, epoch))    

def _instantiate_model(model_type, num_genes, num_classes, mask, embed_dim, depth, num_heads):
    """Create a model instance with the correct constructor args for each type."""
    if model_type == 'TOSICA':
        return create_model(num_classes=num_classes, num_genes=num_genes,
                            mask=mask, embed_dim=embed_dim, depth=depth,
                            num_heads=num_heads, has_logits=False)
    elif model_type == 'my_TOSICA':
        mask_tensor = torch.from_numpy(mask).float() if isinstance(mask, np.ndarray) else mask.float()
        return my_implementation_TOSICA(
            n_genes=num_genes, n_pathways=mask.shape[1], n_classes=num_classes,
            mask=mask_tensor, embed_dim=embed_dim, n_heads=num_heads, n_layers=depth)
    elif model_type == 'MLP':
        return MLP_Model(input_dim=num_genes, n_classes=num_classes)
    elif model_type == 'CNN':
        return CellTypeCNN(seq_len=num_genes, n_classes=num_classes)
    elif model_type == 'GNN':
        return CellTypeGNN(in_dim=num_genes, hidden_dim=embed_dim, n_classes=num_classes)
    else:
        raise ValueError('Invalid model type: {}'.format(model_type))


def main():

    if not os.path.exists('data/10x/data.h5ad'):
        adata = sc.read_csv('data/10x/matrix.csv')
        meta = pd.read_csv('data/10x/metadata.csv', index_col=0)
        adata.obs = meta.loc[adata.obs_names]
        adata.write_h5ad('data/10x/data.h5ad')
    else:
        adata = sc.read_h5ad('data/10x/data.h5ad')
    models = ['TOSICA', 'my_TOSICA', 'MLP', 'GNN', 'CNN']

    #  Train all models 
    for model_name in models:
        print(f'\n=== Training {model_name} ===')
        fit_model(adata, gmt_path='data/reactome.gmt',
                  project='TOSICA_comparison', model_type=model_name)

    #  Evaluate all models 
    device = torch.device(DEVICE)
    mask = np.load(PROJECT.get('mask_path'))
    pathway = pd.read_csv(PROJECT.get('pathway_path'), index_col=0)
    dictionary = pd.read_table(PROJECT.get('label_dictionary_path'), sep=',', header=0, index_col=0)
    n_c = len(dictionary)
    label_name = dictionary.columns[0]
    dictionary.loc[dictionary.shape[0]] = 'Unknown'
    dic = {i: dictionary[label_name][i] for i in range(len(dictionary))}
    num_genes = adata.shape[1]

    results = {}
    for model_type in models:
        print(f'\n=== Evaluating {model_type} ===')
        model_path = PROJECT.get('model_weight_path').format(model_type, CONFIG['epochs'] - 1)
        model = _instantiate_model(model_type, num_genes, n_c,
                                   mask, CONFIG['embed_dim'],
                                   CONFIG['depth'], CONFIG['num_heads'])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        #  Gene2token analysis (only for original TOSICA) 
        if model_type == 'TOSICA':
            parm = {}
            for name, parameters in model.named_parameters():
                parm[name] = parameters.detach().cpu().numpy()
            gene2token = parm['feature_embed.fe.weight']
            gene2token = gene2token.reshape(
                (int(gene2token.shape[0] / CONFIG['embed_dim']),
                 CONFIG['embed_dim'], adata.shape[1]))
            gene2token = abs(gene2token)
            gene2token = np.max(gene2token, axis=1)
            gene2token = pd.DataFrame(gene2token)
            gene2token.columns = adata.var_names
            gene2token.index = pathway['0']
            gene2token.to_csv('TOSICA_comparison/gene2token_weights.csv')

        #  Load test set 
        project_path = os.getcwd() + '/TOSICA_comparison'
        exp_test_np = np.load(project_path + '/exp_test.npy').astype(np.float32)
        label_test = np.load(project_path + '/label_test.npy').astype(np.int64)
        exp_test = torch.from_numpy(exp_test_np)

        #  Build GNN graph if needed 
        edge_index = None
        if model_type == 'GNN':
            edge_index = build_knn_edges(exp_test_np, k=15).to(device)

        #  Run inference on test set 
        all_line = len(exp_test)
        n_line = 0
        adata_list = []
        all_preds = []
        while n_line < all_line:
            chunk_size = min(CONFIG['n_step'], all_line - n_line)
            if chunk_size % CONFIG['batch_size'] == 1 and chunk_size > 2:
                chunk_size -= 2
            chunk_exp = exp_test[n_line:n_line + chunk_size]
            print(f'{model_type}: processing cells {n_line} to {n_line + chunk_size}')

            data_loader = torch.utils.data.DataLoader(
                chunk_exp, batch_size=CONFIG['batch_size'],
                shuffle=False, pin_memory=True)

            with torch.no_grad():
                for step, data in enumerate(data_loader):
                    exp = data.to(device)

                    if model_type == 'TOSICA':
                        _, pre_logits, weights = model(exp)
                    elif model_type == 'my_TOSICA':
                        pre_logits, weights = model(exp, return_attention=True)
                    elif model_type == 'GNN':
                        full_logits = model(exp_test.to(device), edge_index)
                        batch_start = n_line + step * CONFIG['batch_size']
                        batch_end = min(batch_start + CONFIG['batch_size'], all_line)
                        pre_logits = full_logits[batch_start:batch_end]
                        weights = None
                    elif model_type == 'CNN':
                        pre_logits = model(exp.unsqueeze(1))
                        weights = None
                    else:  # MLP
                        pre_logits = model(exp)
                        weights = None

                    pre = torch.squeeze(pre_logits).cpu()
                    pre = F.softmax(pre, dim=1) if pre.dim() > 1 else F.softmax(pre.unsqueeze(0), dim=1)
                    predict_class = np.empty(shape=0)
                    pre_class = np.empty(shape=0)
                    for i in range(len(pre)):
                        if torch.max(pre, dim=1)[0][i] >= .1:
                            predict_class = np.r_[predict_class, torch.max(pre, dim=1)[1][i].numpy()]
                        else:
                            predict_class = np.r_[predict_class, n_c]
                        pre_class = np.r_[pre_class, torch.max(pre, dim=1)[0][i]]
                    meta = pd.DataFrame(
                        np.c_[predict_class, pre_class],
                        columns=['Prediction', 'Probability'])
                    meta.index = meta.index.astype('str')

                    all_preds.append(predict_class)

                    if model_type in ('TOSICA', 'my_TOSICA') and weights is not None:
                        n_pw = len(pathway) if isinstance(pathway, list) else len(pathway)
                        att = torch.squeeze(weights).cpu().numpy()
                        att = att[:, 0:(n_pw - PRE_CONFIG['n_unannotated'])]
                        att = att.astype('float32')
                        pw_vals = pathway['0'] if isinstance(pathway, pd.DataFrame) else pathway
                        pw_trimmed = pw_vals[:n_pw - PRE_CONFIG['n_unannotated']]
                        if isinstance(pw_trimmed, pd.Series):
                            pw_trimmed = pw_trimmed.values
                        varinfo = pd.DataFrame(
                            pw_trimmed, index=pw_trimmed,
                            columns=['pathway_index'])
                        new = sc.AnnData(att, obs=meta, var=varinfo)
                    else:
                        new = sc.AnnData(pre.numpy().astype('float32'), obs=meta)
                    adata_list.append(new)

            n_line += chunk_size

        print(f'{model_type}: {all_line} test cells processed')

        #  Compute metrics 
        y_pred = np.concatenate(all_preds).astype(np.int64)
        y_true = label_test[:len(y_pred)]
        # Cells predicted as "Unknown" (n_c) don't match any true label;
        # mask them so they count as wrong but don't break per-class metrics
        known_mask = y_pred < n_c
        acc = accuracy_score(y_true, np.where(known_mask, y_pred, -1))
        f1_macro = f1_score(y_true, np.where(known_mask, y_pred, -1), average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, np.where(known_mask, y_pred, -1), average='weighted', zero_division=0)
        precision = precision_score(y_true, np.where(known_mask, y_pred, -1), average='weighted', zero_division=0)
        recall = recall_score(y_true, np.where(known_mask, y_pred, -1), average='weighted', zero_division=0)
        print(f'\n {model_type} Test Metrics ')
        print(f'  Accuracy:           {acc:.4f}')
        print(f'  F1 (macro):         {f1_macro:.4f}')
        print(f'  F1 (weighted):      {f1_weighted:.4f}')
        print(f'  Precision (weighted):{precision:.4f}')
        print(f'  Recall (weighted):  {recall:.4f}')
        print(f'  Unknown rate:       {1 - known_mask.mean():.4f}')

        result = ad.concat(adata_list)
        result.obs['Prediction'] = result.obs['Prediction'].map(dic)
        results[model_type] = result

    return results


if __name__ == '__main__':
    results = main()
    results['TOSICA'].write_h5ad('TOSICA_comparison/TOSICA_results.h5ad')
    results['my_TOSICA'].write_h5ad('TOSICA_comparison/my_TOSICA_results.h5ad')
    results['MLP'].write_h5ad('TOSICA_comparison/MLP_results.h5ad')
    results['GNN'].write_h5ad('TOSICA_comparison/GNN_results.h5ad')
    results['CNN'].write_h5ad('TOSICA_comparison/CNN_results.h5ad')
    # Now you can load these .h5ad files in Scanpy and compare the UMAPs, attention scores, and predictions across models.
    
    
    
    