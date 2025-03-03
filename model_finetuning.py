import os
import torch
from pocket_strmod.utils.training_trajectory import *
from pocket_strmod.utils import Experiment, LoadDataset
from pocket_strmod.autoregressive_process import pocket_strmod_WithEdgeNew, reset_parameters

torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


protein_featurizer = FeaturizeProteinAtom()
ligand_featurizer = FeaturizeLigandAtom(atomic_numbers=[6,7,8,9,15,16,17,35,53])
traj_fn = LigandTrajectory(perm_type='mix', num_atom_type=9)
pivotal_masker = pivotalMaker(r=4.0, num_work=16, atomic_numbers=[6,7,8,9,15,16,17,35,53])
atom_composer = AtomComposer(
    knn=16, num_workers=16, graph_type='knn', radius=10.0, use_protein_bond=True
    )
combine = Combine(traj_fn, pivotal_masker, atom_composer, lig_only=False)


transform = TrajCompose([
    RefineData(),
    LigandCountNeighbors(),
    protein_featurizer,
    ligand_featurizer,
    combine,
    collate_fn
])

dataset = LoadDataset('finetuning_data.lmdb', transform=transform)
print('Num data:', len(dataset))
train_set, valid_set = LoadDataset.split(dataset, val_num=100, shuffle=True, random_seed=0)

device = 'cuda:0'
ckpt = torch.load('./pretraining.pt', map_location=device)
config = ckpt['config']
model = pocket_strmod_WithEdgeNew(config).to(device)
model.load_state_dict(ckpt['model'])
print(model.get_parameter_number())
keys = [ 
        'pos_predictor.mu_net', 'pos_predictor.logsigma_net', 'pos_predictor.pi_net',
        'pivotal_net.net.1',
       ]
model = reset_parameters(model, keys)                                                                                                                                      


optimizer = torch.optim.Adam(model.parameters(), lr=2.e-4, weight_decay=0, betas=(0.99, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=10, min_lr=1.e-5)

exp = Experiment(
    model, train_set, optimizer, valid_set=valid_set, scheduler=scheduler,
    device='cuda:0', data_parallel=False, use_amp=False, grad_accu_step=4
    )
exp.fit_step(
    500000, valid_per_step=5000, train_batch_size=1, valid_batch_size=4, print_log=False,
    with_tb=True, logdir='./finetuning_log', schedule_key='loss', num_workers=16, 
    pin_memory=False, follow_batch=[], exclude_keys=[], 
    max_edge_num_in_batch=350000
    )
