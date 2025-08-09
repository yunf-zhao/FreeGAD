from utils import *
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def Anchor_Guided_Anomaly_Scoring(X, positive_mask, negative_mask, args):
    alpha, beta = args.alpha, args.beta
    query_embed = X

    positive_index = torch.nonzero(positive_mask == True).squeeze(1).tolist()
    positive_anchor_nodes = X[positive_index]

    negative_index = torch.nonzero(negative_mask == True).squeeze(1).tolist()
    negative_anchor_nodes = X[negative_index]

    positive_distances = torch.cdist(query_embed, positive_anchor_nodes)
    negative_distances = torch.cdist(query_embed, negative_anchor_nodes)

    query_score_avg_pos = torch.mean(positive_distances, dim=-1)
    query_score_min_pos, _ = torch.min(positive_distances, dim=-1)
    query_score_max_pos, _ = torch.max(positive_distances, dim=-1)

    query_score_avg_neg = torch.mean(negative_distances, dim=-1)
    query_score_min_neg, _ = torch.min(negative_distances, dim=-1)
    query_score_max_neg, _ = torch.max(negative_distances, dim=-1)

    score_pos = alpha * (query_score_avg_pos + query_score_min_pos + query_score_max_pos)
    score_neg = beta * (query_score_avg_neg + query_score_max_neg + query_score_min_neg)

    query_score = score_pos - score_neg
    return query_score


def FreeGAD(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    adj, attrs, label, adj_label = load_anomaly_detection_dataset(args.dataset)
    attrs = torch.FloatTensor(attrs)
    node_num = attrs.shape[0]

    feat_ = propagated(adj, attrs, args.num_hops, device)
    embedding_fea = Affinity_Gated_Residual_Encoder(feat_)
    orign_fea = feat_[0]

    similiar = torch.nn.functional.cosine_similarity(embedding_fea, orign_fea)

    max_value, max_index = torch.topk(similiar, args.num_shot, largest=True)
    min_value, min_index = torch.topk(similiar, args.num_shot, largest=False)

    max_marsk_matrix = torch.zeros(node_num, dtype=torch.bool)
    min_marsk_matrix = torch.zeros(node_num, dtype=torch.bool)

    max_marsk_matrix[max_index] = 1
    min_marsk_matrix[min_index] = 1

    y = torch.LongTensor(label)
    y = y.to(device)
    max_marsk_matrix = max_marsk_matrix.to(device)
    min_marsk_matrix = min_marsk_matrix.to(device)

    query_score = Anchor_Guided_Anomaly_Scoring(embedding_fea, max_marsk_matrix, min_marsk_matrix, args)
    score = query_score.detach().cpu().numpy()

    y = y.detach().cpu().numpy()

    auprc = average_precision_score(y, score)
    auc = roc_auc_score(y, score)

    return auc, auprc
