import os, torch, pandas as pd
from regnet.models.regnet import RegNet
from regnet.training.pretrain import build_edge_index, save_outputs
from regnet.preprocessing.data_loader import create_batches_NC

DATA_DIR = 'data/Benchmark Dataset/Lofgof Dataset/mESC/TFs+500'
EXPR_FILE = os.path.join(DATA_DIR,'BL--ExpressionData.csv')
LABEL_FILE = os.path.join(DATA_DIR,'Full_set.csv')
TF_FILE = os.path.join(DATA_DIR,'TF.csv')
TARGET_FILE = os.path.join(DATA_DIR,'Target.csv')
PRETRAIN_DIR = 'pretrain_outputs/Lofgof_mESC_TF500+/pretrain'

expr_df = pd.read_csv(EXPR_FILE,index_col=0)
gene_to_idx = {g:i for i,g in enumerate(expr_df.index)}
edge_index, edge_labels = build_edge_index(LABEL_FILE, TF_FILE, TARGET_FILE, gene_to_idx)

# device
DEVICE = torch.device('cpu')

model = RegNet(expr_df.shape[1],128,64)
model_path = os.path.join(PRETRAIN_DIR,'regnet_pretrained.pth')
model.load_state_dict(torch.load(model_path,map_location=DEVICE),strict=False)
model = model.to(DEVICE)

# create batches
loader = create_batches_NC(expr_df, edge_index, edge_labels, batch_size=1024)

save_outputs(model, loader, DEVICE, PRETRAIN_DIR, expr_df.index.tolist(), edge_index, edge_labels)
print('Outputs regenerated') 