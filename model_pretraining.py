import torch
import os
from easydict import EasyDict
from pocket_strmod.utils.training_trajectory import *
from pocket_strmod.autoregressive_process import pocket_strmod_WithEdgeNew
from pocket_strmod.utils import LoadDataset
from pocket_strmod.utils.model_training import Experiment

torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


protein_featurizer = FeaturizeProteinAtom()
ligand_featurizer = FeaturizeLigandAtom(atomic_numbers=[6,7,8,9,15,16,17,35,53])
traj_fn = LigandTrajectory(perm_type='mix', num_atom_type=9)
pivotal_masker = pivotalMaker(r=6.0, num_work=16, atomic_numbers=[6,7,8,9,15,16,17,35,53])
atom_composer = AtomComposer(
    knn=16, num_workers=16, graph_type='knn', radius=10.0, use_protein_bond=False
    )
combine = Combine(traj_fn, pivotal_masker, atom_composer, lig_only=True)
transform = TrajCompose([
    RefineData(),
    LigandCountNeighbors(),
    protein_featurizer,
    ligand_featurizer,
    combine,
    collate_fn
])

dataset = LoadDataset('ZINC_dataset.lmdb', transform=transform)

print('Num data:', len(dataset))
train_set, valid_set = LoadDataset.split(dataset, val_num=10000, shuffle=True, random_seed=0)


###################################  You can define model parameters here, but be careful that some parameters are related with each other  ###################################################
encoder_cfg = EasyDict(
        {'edge_channels':48, 'num_interactions':6, 'num_heads':6,
     'knn':16, 'cutoff':10.0}
     )
pivotal_net_cfg = EasyDict(
    {'hidden_dim_sca':128, 'hidden_dim_vec':32}
    )
atom_flow_cfg = EasyDict(
    {'hidden_dim_sca':128, 'hidden_dim_vec':32, 'num_flow_layers':6}
    )
pos_predictor_cfg = EasyDict(
    {'num_filters':[128,64], 'n_component':8}
    )
edge_flow_cfg = EasyDict(
    {'edge_channels':64, 'num_filters':[128,32], 'num_bond_types':3,
     'num_heads':8, 'cutoff':10.0, 'num_flow_layers':6}
    )
config = EasyDict(
    {'deq_coeff':0.9, 'hidden_channels':192, 'hidden_channels_vec':48, 'use_conv1d':False, 'num_bond_types':4,
     'bottleneck':(8,2), 'protein_atom_feature_dim':27, 'ligand_atom_feature_dim':15, 'num_atom_type':9,
     'msg_annealing':True, 'encoder':encoder_cfg, 'atom_flow':atom_flow_cfg, 'pos_predictor':pos_predictor_cfg,
     'edge_flow':edge_flow_cfg, 'pivotal_net':pivotal_net_cfg, }
    )
######################################################################################################################################################


model = pocket_strmod_WithEdgeNew(config).to('cuda:0')
print(model.get_parameter_number())
optimizer = torch.optim.Adam(model.parameters(), lr=2.e-4, weight_decay=0, betas=(0.99, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=10, min_lr=1.e-5)

exp = Experiment(
    model, train_set, optimizer, valid_set=valid_set, scheduler=scheduler,
    device='cuda:0', data_parallel=False, use_amp=False
    )
exp.fit_step(
    500000, valid_per_step=5000, train_batch_size=64, valid_batch_size=128, print_log=False,
    with_tb=True, logdir='./pretraining_log', schedule_key='loss', num_workers=16, 
    pin_memory=False, follow_batch=[], exclude_keys=[], collate_fn=None, 
    max_edge_num_in_batch=400000, note='zinc'
    )
