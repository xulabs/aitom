import torch
import torchvision
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import math


class CECT_dataset(Dataset):
    def __init__(self, path=None):
        # TODO
        # 1. Initialize file path or list of file names.
        self.path = path
        self.classes = os.listdir(self.path)
        self.class2id = {}
        self.imgs = []

        for each_class in self.classes:
            if not each_class in self.class2id:
                self.class2id[each_class] = len(self.class2id)
            for items in os.listdir(os.path.join(self.path, each_class)):
                self.imgs.append((os.path.join(self.path, each_class, items), each_class))

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        path, CECT_class = self.imgs[index]
        return torch.from_numpy(np.load(path)).unsqueeze(0).type(torch.FloatTensor), torch.from_numpy(
            np.array(self.class2id[CECT_class])).long()

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.imgs)


def sample_data(opt):
    dataset = CECT_dataset(path=opt['src_data'])
    n = len(dataset)
    X = torch.Tensor(n, 1, 28, 28, 28)
    Y = torch.LongTensor(n)

    inds = torch.randperm(len(dataset))
    for i, index in enumerate(inds):
        x, y = dataset[index]
        X[i] = x
        Y[i] = y
    return X, Y


def create_target_samples(opt, n=1, classes_num=23):
    dataset = CECT_dataset(path=opt['tar_data'])
    X, Y = [], []
    classes = classes_num * [n]

    i = 0
    while True:
        if len(X) == n * classes_num:
            break
        x, y = dataset[i]
        if classes[y] > 0:
            X.append(x)
            Y.append(y)
            classes[y] -= 1
        i += 1

    assert (len(X) == n * classes_num)
    return torch.stack(X, dim=0), torch.from_numpy(np.array(Y))


def create_groups(X_s, Y_s, X_t, Y_t, seed=1):
    """
    G1: a pair of pic comes from same domain ,same class
    G3: a pair of pic comes from same domain, different classes

    G2: a pair of pic comes from different domain,same class
    G4: a pair of pic comes from different domain, different classes
    """
    # change seed so every time wo get group data will different in source domain,
    # but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)

    # classes_num*shot
    n = X_t.shape[0]

    # shuffle order
    classes = torch.unique(Y_t)
    classes = classes[torch.randperm(len(classes))]

    class_num = classes.shape[0]
    shot = n // class_num

    def s_idxs(c):
        idx = torch.nonzero(Y_s.eq(int(c)))

        return idx[torch.randperm(len(idx))][:shot * 2].squeeze()

    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))

    source_matrix = torch.stack(source_idxs)

    target_matrix = torch.stack(target_idxs)

    G1, G2, G3, G4 = [], [], [], []
    Y1, Y2, Y3, Y4 = [], [], [], []

    for i in range(class_num):
        for j in range(shot):
            G1.append((X_s[source_matrix[i][j * 2]], X_s[source_matrix[i][j * 2 + 1]]))
            Y1.append((Y_s[source_matrix[i][j * 2]], Y_s[source_matrix[i][j * 2 + 1]]))
            G2.append((X_s[source_matrix[i][j]], X_t[target_matrix[i][j]]))
            Y2.append((Y_s[source_matrix[i][j]], Y_t[target_matrix[i][j]]))
            G3.append((X_s[source_matrix[i % class_num][j]], X_s[source_matrix[(i + 1) % class_num][j]]))
            Y3.append((Y_s[source_matrix[i % class_num][j]], Y_s[source_matrix[(i + 1) % class_num][j]]))
            G4.append((X_s[source_matrix[i % class_num][j]], X_t[target_matrix[(i + 1) % class_num][j]]))
            Y4.append((Y_s[source_matrix[i % class_num][j]], Y_t[target_matrix[(i + 1) % class_num][j]]))

    groups = [G1, G2, G3, G4]
    groups_y = [Y1, Y2, Y3, Y4]

    # make sure we sampled enough samples
    for g in groups:
        assert (len(g) == n)
    return groups, groups_y


def sample_groups(X_s, Y_s, X_t, Y_t, seed=1):
    print("Sampling groups")
    return create_groups(X_s, Y_s, X_t, Y_t, seed=seed)


def build_datapair(batch_size, path):
    dataset = CECT_dataset(path=path)
    CECT_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    X = []
    Y = []
    for x, y in CECT_dataloader:
        X.append(x)
        Y.append(y)
    return X, Y
