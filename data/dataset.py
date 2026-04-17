import torch
from torch_geometric.data import Data, Dataset


class SpatioTemporalDataset(Dataset):
    def __init__(self, inputs, split):
        super().__init__()
        required = [
            f"{split}_x",
            f"{split}_y",
            f"{split}_y_mask",
            f"{split}_static_data",
            f"{split}_freq_id",
        ]
        missing = [key for key in required if key not in inputs]
        if missing:
            raise KeyError(f"Missing dataset inputs for split='{split}': {missing}")

        self.x = inputs[f"{split}_x"]
        self.y = inputs[f"{split}_y"]
        self.y_mask = inputs[f"{split}_y_mask"]
        self.static_data = inputs[f"{split}_static_data"]
        self.freq_id = inputs[f"{split}_freq_id"]

        num_samples = self.x.shape[0]
        shapes = {
            "y": self.y.shape[0],
            "y_mask": self.y_mask.shape[0],
            "static_data": self.static_data.shape[0],
            "freq_id": self.freq_id.shape[0],
        }
        inconsistent = {name: size for name, size in shapes.items() if size != num_samples}
        if inconsistent:
            raise ValueError(f"Split='{split}' has inconsistent sample counts: x={num_samples}, others={inconsistent}")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return Data(
            x=torch.from_numpy(self.x[index].T).float(),
            y=torch.from_numpy(self.y[index].T).float(),
            y_mask=torch.from_numpy(self.y_mask[index].T).float(),
            static_data=torch.from_numpy(self.static_data[index]).float(),
            freq_id=torch.tensor([int(self.freq_id[index])], dtype=torch.long),
        )


class SingleTurbineDataset(Dataset):
    def __init__(self, inputs, split):
        super().__init__()
        required = [
            f"{split}_x",
            f"{split}_y",
            f"{split}_y_mask",
        ]
        missing = [key for key in required if key not in inputs]
        if missing:
            raise KeyError(f"Missing single-turbine inputs for split='{split}': {missing}")

        self.x = inputs[f"{split}_x"]
        self.y = inputs[f"{split}_y"]
        self.y_mask = inputs[f"{split}_y_mask"]

        num_samples = self.x.shape[0]
        shapes = {
            "y": self.y.shape[0],
            "y_mask": self.y_mask.shape[0],
        }
        inconsistent = {name: size for name, size in shapes.items() if size != num_samples}
        if inconsistent:
            raise ValueError(
                f"Single-turbine split='{split}' has inconsistent sample counts: x={num_samples}, others={inconsistent}"
            )

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return Data(
            x=torch.from_numpy(self.x[index][None, :]).float(),
            y=torch.from_numpy(self.y[index][None, :]).float(),
            y_mask=torch.from_numpy(self.y_mask[index][None, :]).float(),
        )
