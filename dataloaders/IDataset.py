import torch

class IDataset(torch.utils.data.Dataset):
    """
        Interface Class for data loading. Other than __init__ method should be
        overrided to make suitable interface.
        __getitem__: return a list of images not dictionary
        """

    def __init__(self):
        super(IDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item: int) -> dict:
        raise NotImplementedError

    def transform(self, image):
        raise NotImplementedError

    @staticmethod
    def fillOutputDataDict(inputs: torch.tensor, gts: torch.tensor, original=None) -> dict:
        data = dict()
        data["inputs"] = inputs
        data["gts"] = gts
        if not original is None:
            data['original'] = original

        return data