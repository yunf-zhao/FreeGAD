import csv
import argparse
from model import *
from utils import set_seed
import warnings


def run_csv_param(path, args, dataset):
    writer = csv.reader(open(path, "r"))
    record = list(writer)
    row_list = []
    for row in record:
        row_list.append(row[0])

    index_dataset = row_list.index(dataset)
    index_list = ["dataSet", "alpha", "beta", "num_hops", "num_shot"]
    index_dict = {}

    for param in index_list:
        index_dict[param] = record[0].index(param)
    index_alpha = index_dict["alpha"]
    index_beta = index_dict["beta"]
    index_num_hops = index_dict["num_hops"]
    index_num_shot = index_dict["num_shot"]

    info = record[index_dataset]
    args.dataset = info[0]
    args.alpha = float(info[index_alpha])
    args.beta = float(info[index_beta])
    args.num_hops = int(info[index_num_hops])
    args.num_shot = int(info[index_num_shot])

    auc, auprc = FreeGAD(args)

    return auc, auprc


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="Reddit", help='dataset name')
    parser.add_argument('--num_shot', type=int, default=30, help='the num of anchor node')
    parser.add_argument('--num-hops', type=int, default=4, help='the num of propagation iteration')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--beta', type=float, default=0.9, help='beta')
    parser.add_argument('--model', type=str, default='FreeGAD')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')

    args = parser.parse_known_args()[0]
    path = "./params/{}.csv".format(args.model)

    dataset_list = ["Amazon", "Reddit", "tolokers", "YelpChi", "tfinance", "questions", "elliptic", "cora",
                    "BlogCatalog", "Flickr"]
    for dataset in dataset_list:
        auc_list, ap_list = [], []
        args.dataset = dataset
        for seed in range(5):
            set_seed(seed)
            auc, ap = run_csv_param(path, args, dataset)
            auc_list.append(auc)
            ap_list.append(ap)
        auc_mean = np.array(auc_list).mean()
        auc_std = np.array(auc_list).std()
        aupr_mean = np.array(ap_list).mean()
        aupr_std = np.array(ap_list).std()
        print('DataSet:{},AUROC:{:.4f}+-{:.3f}, AUPRC:{:.4f}+-{:.3f}'.format(args.dataset, auc_mean, auc_std, aupr_mean,
                                                                             aupr_std))
