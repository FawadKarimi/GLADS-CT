
import pandas as pd
from dataset import LIDCSliceDataset
from torch.utils.data import DataLoader

df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv")
train_df = df[df["split_slice"] == "train"]

ds = LIDCSliceDataset(train_df, is_train=True)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)

batch = next(iter(loader))
print("DataLoader test passed!")
print("Batch image shape:", batch["image"].shape)
print("Batch label shape:", batch["label"].shape)
print("Batch reader target shape:", batch["reader_target"].shape)
print("Batch disagreement shape:", batch["disagreement"].shape)
