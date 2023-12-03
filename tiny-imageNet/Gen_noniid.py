import numpy as np
import torch
from torchvision import datasets
from torchvision import datasets, transforms
from non_iid_dataset import MyDataset
from test_loader import TinyImageNet_load
import os


class Arguments:
    def __init__(self) -> None:
        self.N_CLIENTS = 15
        self.batch_size = 128
        self.test_batch_size = 1000
        self.DIRICHLET_ALPHA = 0.5
        self.data_path = r"/home/zjgsu006/z/dataset/tiny-imagenet-200" # /home/zsh/flowerpot/Dataset  /home/zjgsu006/z/dataset
        self.user_data_distr = []
        self.user_data_cata = []


def dirichlet_split_noniid(train_labels, n_classes, alpha, n_clients):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    '''
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标

    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


def class_count(n_classes, client, train_labels):
    ls = []
    ls_count = []
    ls_cls = []
    for icls in range(n_classes):
        count = 0
        ls_class = []
        for idx in client:
            if icls == train_labels[idx]:
                count += 1
                ls_class.append(idx)
        if count > 0:
            ls.append(ls_class)         # 样本索引
            ls_count.append(count)      # 样本数量
        ls_cls.append(icls)         # 数据类别
        
    return ls, ls_count, ls_cls


def divide_idx(args, n_classes, train_labels):

    # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
    client_idcs = dirichlet_split_noniid(train_labels, n_classes, alpha=args.DIRICHLET_ALPHA, n_clients=args.N_CLIENTS)

    return client_idcs


def divide_dataset(args, client_idcs, n_classes, train_labels, train_data):
    clients_train_loaders = []

    for user_idx in range(args.N_CLIENTS):

        ls, ls_count, ls_cls = class_count(n_classes, client_idcs[user_idx], train_labels)

        print(ls_count)

        # args.user_data_distr.append(ls_count)
        # args.user_data_cata.append(ls_cls)

        client_train_data = []
        client_train_label = []

        for i in range(len(ls)):
            for idx in ls[i]:
                client_train_data.append(train_data[idx][0])
                client_train_label.append(train_data[idx][1])

        client_train = [client_train_data, client_train_label]

        client_train_dataset = MyDataset(client_train, train=True,
            transform=transforms.Compose(
                [
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]
            )
        )

        train_loader = torch.utils.data.DataLoader(client_train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)

        clients_train_loaders.append(train_loader)

    return clients_train_loaders


def get_user_dataset():
    np.random.seed(42)

    args = Arguments()

    train_data = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'),)

    test_data = TinyImageNet_load(root=args.data_path, train=False, 
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                    ]
                ))

    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=args.test_batch_size)

    train_labels = np.array(train_data.targets)

    n_classes = train_labels.max()+1

    client_idcs = divide_idx(args, n_classes, train_labels)

    # print(len(train_data))

    clients_train_loaders =  divide_dataset(args, client_idcs, n_classes, train_labels, train_data)
    
    return clients_train_loaders, test_loader

# train, _ = get_user_dataset()

# print(len(trains))
