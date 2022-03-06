from SPDSiamese.model import ST_TS_SPDC
from SPDSiamese.Dataloader import Dataset_prep

transformed_dataset = Dataset_prep(train=True)
dataloader = DataLoader(transformed_dataset, batch_size=30, shuffle=True, num_workers=0)
transformed_dataset = Dataset_prep(train=False)
dataloader = DataLoader(transformed_dataset, batch_size=30, shuffle=True, num_workers=0)

