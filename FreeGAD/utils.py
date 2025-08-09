import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import torch.nn.functional as F
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def Affinity_Gated_Residual_Encoder(feat_):
    device = feat_[0].device
    base_fea = feat_[0].to(device)
    norm_base_fea = torch.norm(base_fea, 2, 1).add(1e-10)

    attention_score_list = []

    for i in range(1, len(feat_)):
        fea = feat_[i].to(device)
        norm_fea = torch.norm(fea, 2, 1).add(1e-10)
        temp = (fea * base_fea).sum(1) / (norm_fea * norm_base_fea)
        attention_score_list.append(temp.unsqueeze(1))

    attention_scores = torch.cat(attention_score_list, dim=1)
    attention_weights = F.softmax(attention_scores, dim=1)

    for i in range(1, len(feat_)):
        fea = feat_[i].to(device)

        temp = attention_weights[:, i - 1].unsqueeze(1)
        feat_[i] = (1-temp) * fea + temp * base_fea

    embedding_fea = sum(feat_[1:]) / (len(feat_) - 1)

    return embedding_fea


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def propagated(adj_norm, x, k, device):
    x = torch.FloatTensor(x).to(device)
    h_list = [x]
    for _ in range(k):
        h_list.append(torch.spmm(adj_norm.to(device), h_list[-1]))
    return h_list


def load_anomaly_detection_dataset(dataset, datadir='./dataset'):
    data_mat = sio.loadmat(f'{datadir}/{dataset}.mat')
    adj = data_mat['Network']
    feat = data_mat['Attributes']
    if dataset in ['tolokers', "elliptic", "tfinance", "Amazon", "cora"]:
        feat = sp.lil_matrix(feat)
        feat = preprocess_features(feat)
    else:
        feat = sp.lil_matrix(feat)
    truth = data_mat['Label']
    truth = truth.flatten()

    if dataset in ['YelpChi', "BlogCatalog", "Flickr"]:
        adj_norm = process_adj(adj)
    else:
        adj_norm = process_adj(adj + sp.eye(adj.shape[0]))
    feat = feat.toarray()
    return adj_norm, feat, truth, adj


def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def process_adj(adj):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)
