import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils import generate_seq


class DalDataset(Dataset):
    def __init__(self, data, sample_rate=1):
        self.data = data
        self.seq = self.data if sample_rate == 1 else generate_seq(data)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        if idx >= len(self.seq):
            raise ValueError("idx out of range")

        return self.seq[idx][0], self.seq[idx][1]


def main():
    data = [[torch.randn(1, 32, 32, 32), np.random.choice(2, 1)] for _ in range(16)]
    dal = DalDataset(data)
    dataloader = DataLoader(dal, batch_size=4, shuffle=False)
    for seq_lst in dataloader:
        print(seq_lst[0].shape)
        print(seq_lst[1].shape)
    print("---------------")
    print("len(dal):", len(dal))
    print("---------------")
    print("len(dataloader.dataset): ", len(dataloader.dataset))
    print("len(dataloader): ", len(dataloader))


if __name__ == "__main__":
    main()
