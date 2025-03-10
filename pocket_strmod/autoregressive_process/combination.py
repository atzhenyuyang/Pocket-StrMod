import torch
from torch import nn
import torch.nn.functional as F
from .net_utils import AtomEmbedding, embed_compose
from .context_encoder import ContextEncoder
from .new_atom_generator import AtomFlow
from .chemical_bond_generator import BondPredictor
from .position_predictor import PositionPredictor
from .pivotal_atom_selector import pivotalLayerVN
from easydict import EasyDict


encoder_cfg = EasyDict(
    {'edge_channels':8, 'num_interactions':6,
     'knn':32, 'cutoff':10.0}
     )
pivotal_net_cfg = EasyDict(
    {'hidden_dim_sca':32, 'hidden_dim_vec':8}
    )
atom_flow_cfg = EasyDict(
    {'hidden_dim_sca':32, 'hidden_dim_vec':8, 'num_lig_atom_type':10,
     'num_flow_layers':6}
    )
pos_predictor_cfg = EasyDict(
    {'num_filters':[64,64], 'n_component':3}
    )
edge_predictor_cfg = EasyDict(
    {'edge_channels':8, 'num_filters':[32,8], 'num_bond_types':3,
     'num_heads':4, 'cutoff':10.0}
    )
config = EasyDict(
    {'deq_coeff':0.9, 'hidden_channels':32, 'hidden_channels_vec':8, 'num_atom_type':10,
     'protein_atom_feature_dim':27, 'ligand_atom_feature_dim':16, 'num_bond_types':3,
     'encoder':encoder_cfg, 'atom_flow':atom_flow_cfg, 'pos_predictor':pos_predictor_cfg,
     'edge_predictor':edge_predictor_cfg, 'pivotal_net':pivotal_net_cfg}
    )


class pocket_strmod_(nn.Module):
    def __init__(self, config) -> None:
        super(pocket_strmod_, self).__init__()
        self.config = config
        self.num_bond_types = config.num_bond_types
        
        self.emb_dim = [config.hidden_channels, config.hidden_channels_vec]
        self.protein_atom_emb = AtomEmbedding(config.protein_atom_feature_dim, 1, *self.emb_dim)
        self.ligand_atom_emb = AtomEmbedding(config.ligand_atom_feature_dim, 1, *self.emb_dim)
        self.atom_type_embedding = nn.Embedding(config.num_atom_type, config.hidden_channels)

        self.encoder = ContextEncoder(
            hidden_channels=self.emb_dim, edge_channels=config.encoder.edge_channels, 
            num_edge_types=config.num_bond_types, num_interactions=config.encoder.num_interactions, 
            k=config.encoder.knn, cutoff=config.encoder.cutoff, bottleneck=config.bottleneck
            )
        self.pivotal_net = pivotalLayerVN(
            self.emb_dim[0], self.emb_dim[1], config.pivotal_net.hidden_dim_sca, 
            config.pivotal_net.hidden_dim_vec, bottleneck=config.bottleneck
            )
        self.atom_flow = AtomFlow(
            self.emb_dim[0], self.emb_dim[1], config.atom_flow.hidden_dim_sca,
            config.atom_flow.hidden_dim_vec, num_lig_atom_type=config.atom_flow.num_lig_atom_type,
            num_flow_layers=config.atom_flow.num_flow_layers, bottleneck=config.bottleneck
            )
        self.pos_predictor = PositionPredictor(
            self.emb_dim[0], self.emb_dim[1], config.pos_predictor.num_filters,
            config.pos_predictor.n_component, bottleneck=config.bottleneck
            )
        self.edge_predictor = BondPredictor(
            self.emb_dim[0], self.emb_dim[1], config.edge_predictor.edge_channels, 
            config.edge_predictor.num_filters, config.edge_predictor.num_bond_types, 
            num_heads=config.edge_predictor.num_heads, cutoff=config.edge_predictor.cutoff,
            bottleneck=config.bottleneck
            )
    
    def get_parameter_number(self):                                                                                                                                                                                     
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def get_loss(self, data):
        h_cpx = embed_compose(data.cpx_feature.float(), data.cpx_pos, data.idx_ligand_ctx_in_cpx, 
                              data.idx_protein_in_cpx, self.ligand_atom_emb, 
                              self.protein_atom_emb, self.emb_dim)
        
        h_cpx = self.encoder(
            node_attr = h_cpx,
            pos = data.cpx_pos,
            edge_index = data.cpx_edge_index,
            edge_feature = data.cpx_edge_feature
        )
       
        pivotal_pred = self.pivotal_net(h_cpx, data.idx_ligand_ctx_in_cpx)
        pivotal_loss = F.binary_cross_entropy_with_logits(
            input=pivotal_pred, target=data.ligand_pivotal.view(-1, 1).float()
        )
        
        
        x_z = F.one_hot(data.atom_label, num_classes=self.config.atom_flow.num_lig_atom_type).float()  
        x_z += self.config.deq_coeff * torch.rand(x_z.size(), device=x_z.device) 
        z_atom, atom_log_jacob = self.atom_flow(x_z, h_cpx, data.pivotal_idx_in_context)
        ll_atom = (1/2 * (z_atom ** 2) - atom_log_jacob).mean() 
        

        # for position loss
        atom_type_emb = self.atom_type_embedding(data.atom_label)
        relative_mu, abs_mu, sigma, pi = self.pos_predictor(
            h_cpx,  # +atom_type_emb[data.step_batch]
            data.pivotal_idx_in_context,
            data.cpx_pos, 
            atom_type_emb=atom_type_emb
            )
        #y_pos = torch.rand_like(data.y_pos) * 0.05 + data.y_pos
        loss_pos = -torch.log(
            self.pos_predictor.get_mdn_probability(abs_mu, sigma, pi, data.y_pos) + 1e-16
            ).mean()#.clamp_max(10.)  # 最大似然log
        

        
        edge_index_query = torch.stack([data.edge_query_index_0, data.edge_query_index_1])
        pos_query_knn_edge_idx = torch.stack(
            [data.pos_query_knn_edge_idx_0, data.pos_query_knn_edge_idx_1]
            )
        edge_pred = self.edge_predictor(
            pos_query=data.y_pos, 
            edge_index_query=edge_index_query, 
            cpx_pos=data.cpx_pos, 
            node_attr_compose=h_cpx, 
            edge_index_q_cps_knn=pos_query_knn_edge_idx, 
            index_real_cps_edge_for_atten=data.index_real_cps_edge_for_atten, 
            tri_edge_index=data.tri_edge_index, 
            tri_edge_feat=data.tri_edge_feat,
            atom_type_emb=atom_type_emb
            )
        loss_edge = F.cross_entropy(edge_pred, data.edge_label, reduction ='mean')
        
        
       
        loss = torch.nan_to_num(ll_atom)\
             + torch.nan_to_num(loss_pos)\
             + torch.nan_to_num(loss_edge)\
             + torch.nan_to_num(pivotal_loss)\
             #+ torch.nan_to_num(surf_loss)
        out_dict = {
            'loss':loss, 'loss_atom':ll_atom, 'loss_edge':loss_edge,
            'loss_pos':loss_pos, 'pivotal_loss':pivotal_loss, #'surf_loss':torch.nan_to_num(surf_loss)
            }
        return out_dict

    def choose_pivotal(self, h_ctx, ctx_idx, pivotal_threshold, choose_max):
        pivotal_pred = self.pivotal_net(h_ctx, ctx_idx)
        pivotal_prob = torch.sigmoid(pivotal_pred).view(-1)
        if choose_max:
            max_idx = pivotal_pred.argmax()
            pivotal_idx_candidate = ctx_idx[max_idx].view(-1)
            pivotal_prob = pivotal_prob[max_idx].view(-1)
        else:
            pivotal_mask = (pivotal_prob >= pivotal_threshold).view(-1)
            pivotal_idx_candidate = ctx_idx[pivotal_mask]
            pivotal_prob = pivotal_prob[pivotal_mask]
            if pivotal_mask.sum() == 0:
                return False, pivotal_prob
        return pivotal_idx_candidate, pivotal_prob 