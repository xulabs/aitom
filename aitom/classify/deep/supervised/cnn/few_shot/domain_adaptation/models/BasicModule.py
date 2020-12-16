import torch


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            name = 'result/best_model.pth'
            torch.save(self.state_dict(), name)
            return name
        else:
            torch.save(self.state_dict(), path)
            return path
