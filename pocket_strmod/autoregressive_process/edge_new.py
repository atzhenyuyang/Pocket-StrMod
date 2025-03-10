import torch
from torch import nn
import torch.nn.functional as F
from .net_utils import AtomEmbedding, embed_compose
from .context_encoder import ContextEncoder
from .new_atom_generator import AtomFlow
from .chemical_bond_generator import BondFlowNew
from .position_predictor import PositionPredictor
#from .pos_filter import PositionFilter
from .pivotal_atom_selector import pivotalLayerVN
from .layers import GlobalAttention
from easydict import EasyDict
from torch_scatter import scatter_add, scatter_mean


encoder_cfg = EasyDict(
    {'edge_channels':8, 'num_interactions':6,
     'knn':32, 'cutoff':10.0}
     )
pivotal_net_cfg = EasyDict(
    {'hidden_dim_sca':32, 'hidden_dim_vec':8}
    )
atom_flow_cfg = EasyDict(
    {'hidden_dim_sca':32, 'hidden_dim_vec':8, 'num_flow_layers':6}
    )
pos_predictor_cfg = EasyDict(
    {'num_filters':[64,64], 'n_component':3}
    )
pos_filter_cfg = EasyDict(
    {'edge_channels':8, 'num_filters':[32,16]}
    )
edge_flow_cfg = EasyDict(
    {'edge_channels':8, 'num_filters':[32,8], 'num_bond_types':3,
     'num_heads':2, 'cutoff':10.0, 'num_flow_layers':3}
    )
config = EasyDict(
    {'deq_coeff':0.9, 'hidden_channels':32, 'hidden_channels_vec':8, 'bottleneck':8, 'use_conv1d':False,
     'encoder':encoder_cfg, 'atom_flow':atom_flow_cfg, 'pos_predictor':pos_predictor_cfg,
     'pos_filter':pos_filter_cfg, 'edge_flow':edge_flow_cfg, 'pivotal_net':pivotal_net_cfg}
    )


class pocket_strmod_WithEdgeNew(nn.Module):
    def __init__(self, config) -> None:
        super(pocket_strmod_WithEdgeNew, self).__init__()
        self.config = config
        self.num_bond_types = config.num_bond_types
        self.msg_annealing = config.msg_annealing        
        self.emb_dim = [config.hidden_channels, config.hidden_channels_vec]
        self.protein_atom_emb = AtomEmbedding(config.protein_atom_feature_dim, 1, *self.emb_dim)
        self.ligand_atom_emb = AtomEmbedding(config.ligand_atom_feature_dim, 1, *self.emb_dim)
        self.atom_type_embedding = nn.Embedding(config.num_atom_type, config.hidden_channels)

        self.encoder = ContextEncoder(
            hidden_channels=self.emb_dim, edge_channels=config.encoder.edge_channels, 
            num_edge_types=config.num_bond_types, num_interactions=config.encoder.num_interactions, 
            k=config.encoder.knn, cutoff=config.encoder.cutoff, bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d, num_heads=config.encoder.num_heads
            )
        self.pivotal_net = pivotalLayerVN(
            self.emb_dim[0], self.emb_dim[1], config.pivotal_net.hidden_dim_sca, 
            config.pivotal_net.hidden_dim_vec, bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d
            )
        self.atom_flow = AtomFlow(
            self.emb_dim[0], self.emb_dim[1], config.atom_flow.hidden_dim_sca,
            config.atom_flow.hidden_dim_vec, num_lig_atom_type=config.num_atom_type,
            num_flow_layers=config.atom_flow.num_flow_layers, bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d
            )
        self.pos_predictor = PositionPredictor(
            self.emb_dim[0], self.emb_dim[1], config.pos_predictor.num_filters,
            config.pos_predictor.n_component, bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d
            )
        self.edge_flow = BondFlowNew(
            self.emb_dim[0], self.emb_dim[1], config.edge_flow.edge_channels, 
            config.edge_flow.num_filters, config.edge_flow.num_bond_types, 
            num_heads=config.edge_flow.num_heads, cutoff=config.edge_flow.cutoff,
            num_st_layers=config.edge_flow.num_flow_layers, bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d
            )
    
    def get_parameter_number(self):                                                                                                                                                                                     
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def get_loss(self, data):
        h_cpx = embed_compose(data.cpx_feature.float(), data.cpx_pos, data.idx_ligand_ctx_in_cpx, 
                              data.idx_protein_in_cpx, self.ligand_atom_emb, 
                              self.protein_atom_emb, self.emb_dim)
        # encoding context
        h_cpx = self.encoder(
            node_attr = h_cpx,
            pos = data.cpx_pos,
            edge_index = data.cpx_edge_index,
            edge_feature = data.cpx_edge_feature,
            annealing=self.msg_annealing
        )
        # for pivotal loss
        pivotal_pred = self.pivotal_net(h_cpx, data.idx_ligand_ctx_in_cpx)
        pivotal_loss = F.binary_cross_entropy_with_logits(
            input=pivotal_pred, target=data.ligand_pivotal.view(-1, 1).float(),
            #reduction='none'
        )
        # for pivotal loss in protein
        pivotal_pred_apo = self.pivotal_net(h_cpx, data.apo_protein_idx)
        surf_loss = F.binary_cross_entropy_with_logits(
            input=pivotal_pred_apo, target=data.candidate_pivotal_label_in_protein.view(-1, 1).float(), #reduction='none'
        )
        # for atom loss
        x_z = F.one_hot(data.atom_label, num_classes=self.config.num_atom_type).float()  #[50,27]
        x_z += self.config.deq_coeff * torch.rand(x_z.size(), device=x_z.device) #[50,27]
        z_atom, atom_log_jacob = self.atom_flow(x_z, h_cpx, data.pivotal_idx_in_context)
        ll_atom = (1/2 * (z_atom ** 2) - atom_log_jacob).mean()
        #ll_atom = scatter_mean(ll_atom, data.atom_label_batch, dim=0).mean()

        # for position loss
        atom_type_emb = self.atom_type_embedding(data.atom_label)
        relative_mu, abs_mu, sigma, pi = self.pos_predictor(
            h_cpx,  # +atom_type_emb[data.step_batch]
            data.pivotal_idx_in_context,
            data.cpx_pos, 
            atom_type_emb=atom_type_emb,
            )
        #y_pos = torch.rand_like(data.y_pos) * 0.05 + data.y_pos
        loss_pos = -torch.log(
            self.pos_predictor.get_mdn_probability(abs_mu, sigma, pi, data.y_pos) + 1e-16
            ).mean()#.clamp_max(10.)  # 最大似然log
        #loss_pos = scatter_mean(loss_pos, data.atom_label_batch, dim=0).mean()
        
        # for edge loss
        z_edge = F.one_hot(data.edge_label, num_classes=4).float()
        z_edge += self.config.deq_coeff * torch.rand(z_edge.size(), device=z_edge.device)
        edge_index_query = torch.stack([data.edge_query_index_0, data.edge_query_index_1])
        pos_query_knn_edge_idx = torch.stack(
            [data.pos_query_knn_edge_idx_0, data.pos_query_knn_edge_idx_1]
            )
        z_edge, edge_log_jacob = self.edge_flow(
            z_edge=z_edge,
            pos_query=data.y_pos, 
            edge_index_query=edge_index_query, 
            cpx_pos=data.cpx_pos, 
            node_attr_compose=h_cpx, 
            edge_index_q_cps_knn=pos_query_knn_edge_idx, 
            index_real_cps_edge_for_atten=data.index_real_cps_edge_for_atten, 
            tri_edge_index=data.tri_edge_index, 
            tri_edge_feat=data.tri_edge_feat,
            atom_type_emb=atom_type_emb,
            annealing=self.msg_annealing
            )
        ll_edge = (1/2 * (z_edge ** 2) - edge_log_jacob).mean()
        loss = torch.nan_to_num(ll_atom)\
             + torch.nan_to_num(loss_pos)\
             + torch.nan_to_num(ll_edge)\
             + torch.nan_to_num(pivotal_loss)\
             + torch.nan_to_num(surf_loss)\
             #+ torch.nan_to_num(loss_fake)\
             #+ torch.nan_to_num(loss_real)\
        
        out_dict = {
            'loss':loss, 
            'loss_atom':ll_atom, 'loss_edge':ll_edge, #'loss_fake':loss_fake, 'loss_real':loss_real,
            'loss_pos':loss_pos, 'pivotal_loss':pivotal_loss, 'surf_loss':torch.nan_to_num(surf_loss)
            }
        return out_dict


class pocket_strmod_WithEdgeImprove(nn.Module):
    def __init__(self, config) -> None:
        super(pocket_strmod_WithEdgeNew, self).__init__()
        self.config = config
        self.num_bond_types = config.num_bond_types
        self.msg_annealing = config.msg_annealing        
        self.emb_dim = [config.hidden_channels, config.hidden_channels_vec]
        self.protein_atom_emb = AtomEmbedding(config.protein_atom_feature_dim, 1, *self.emb_dim)
        self.ligand_atom_emb = AtomEmbedding(config.ligand_atom_feature_dim, 1, *self.emb_dim)
        self.atom_type_embedding = nn.Embedding(config.num_atom_type, config.hidden_channels)

        self.encoder = ContextEncoder(
            hidden_channels=self.emb_dim, edge_channels=config.encoder.edge_channels, 
            num_edge_types=config.num_bond_types, num_interactions=config.encoder.num_interactions, 
            k=config.encoder.knn, cutoff=config.encoder.cutoff, bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d, num_heads=config.encoder.num_heads
            )
        self.lig_ctx_emb = GlobalAttention(
            self.emb_dim[0], self.emb_dim[1], self.emb_dim[0], self.emb_dim[1],
            bottleneck=config.bottleneck, use_conv1d=config.use_conv1d
            )
        self.pivotal_net = pivotalLayerVN(
            self.emb_dim[0], self.emb_dim[1], config.pivotal_net.hidden_dim_sca, 
            config.pivotal_net.hidden_dim_vec, bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d
            )
        self.atom_flow = AtomFlow(
            self.emb_dim[0], self.emb_dim[1], config.atom_flow.hidden_dim_sca,
            config.atom_flow.hidden_dim_vec, num_lig_atom_type=config.num_atom_type,
            num_flow_layers=config.atom_flow.num_flow_layers, bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d
            )
        self.pos_predictor = PositionPredictor(
            self.emb_dim[0], self.emb_dim[1], config.pos_predictor.num_filters,
            config.pos_predictor.n_component, bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d
            )
        self.edge_flow = BondFlowNew(
            self.emb_dim[0], self.emb_dim[1], config.edge_flow.edge_channels, 
            config.edge_flow.num_filters, config.edge_flow.num_bond_types, 
            num_heads=config.edge_flow.num_heads, cutoff=config.edge_flow.cutoff,
            num_st_layers=config.edge_flow.num_flow_layers, bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d
            )
    
    def get_parameter_number(self):                                                                                                                                                                                     
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def get_loss(self, data):
        h_cpx = embed_compose(data.cpx_feature.float(), data.cpx_pos, data.idx_ligand_ctx_in_cpx, 
                              data.idx_protein_in_cpx, self.ligand_atom_emb, 
                              self.protein_atom_emb, self.emb_dim)
        # encoding context
        h_cpx = self.encoder(
            node_attr = h_cpx,
            pos = data.cpx_pos,
            edge_index = data.cpx_edge_index,
            edge_feature = data.cpx_edge_feature,
            annealing=self.msg_annealing
        )
        # embedding ligand context
        lig_ctx_embed = self.lig_ctx_emb(
            h_cpx,
            data.idx_ligand_ctx_in_cpx,
            data.ligand_ctx_step
        )
        h_pivotal = [
            torch.cat([lig_ctx_embed[0],h_cpx[0][data.pivotal_idx_in_context]], dim=0),
            torch.cat([lig_ctx_embed[1],h_cpx[1][data.pivotal_idx_in_context]], dim=1)
            ]
        h_pivotal = self.gb_net(h_pivotal)
        
        # for pivotal loss
        pivotal_pred = self.pivotal_net(h_cpx, data.idx_ligand_ctx_in_cpx)
        pivotal_loss = F.binary_cross_entropy_with_logits(
            input=pivotal_pred, target=data.ligand_pivotal.view(-1, 1).float(),
            #reduction='none'
        )
        # for pivotal loss in protein
        pivotal_pred_apo = self.pivotal_net(h_cpx, data.apo_protein_idx)
        surf_loss = F.binary_cross_entropy_with_logits(
            input=pivotal_pred_apo, target=data.candidate_pivotal_label_in_protein.view(-1, 1).float(), #reduction='none'
        )
        # for atom loss
        x_z = F.one_hot(data.atom_label, num_classes=self.config.num_atom_type).float()  #[50,27]
        x_z += self.config.deq_coeff * torch.rand(x_z.size(), device=x_z.device) #[50,27]
        z_atom, atom_log_jacob = self.atom_flow(x_z, h_cpx, data.pivotal_idx_in_context)
        ll_atom = (1/2 * (z_atom ** 2) - atom_log_jacob).mean()
        #ll_atom = scatter_mean(ll_atom, data.atom_label_batch, dim=0).mean()

        # for position loss
        atom_type_emb = self.atom_type_embedding(data.atom_label)
        relative_mu, abs_mu, sigma, pi = self.pos_predictor(
            h_cpx,  # +atom_type_emb[data.step_batch]
            data.pivotal_idx_in_context,
            data.cpx_pos, 
            atom_type_emb=atom_type_emb,
            )
        #y_pos = torch.rand_like(data.y_pos) * 0.05 + data.y_pos
        loss_pos = -torch.log(
            self.pos_predictor.get_mdn_probability(abs_mu, sigma, pi, data.y_pos) + 1e-16
            ).mean()#.clamp_max(10.)  # 最大似然log
        #loss_pos = scatter_mean(loss_pos, data.atom_label_batch, dim=0).mean()
        
        # for edge loss
        z_edge = F.one_hot(data.edge_label, num_classes=4).float()
        z_edge += self.config.deq_coeff * torch.rand(z_edge.size(), device=z_edge.device)
        edge_index_query = torch.stack([data.edge_query_index_0, data.edge_query_index_1])
        pos_query_knn_edge_idx = torch.stack(
            [data.pos_query_knn_edge_idx_0, data.pos_query_knn_edge_idx_1]
            )
        #pos_query = data.y_pos + torch.randn_like(data.y_pos) * 0.1
        z_edge, edge_log_jacob = self.edge_flow(
            z_edge=z_edge,
            pos_query=data.y_pos, 
            edge_index_query=edge_index_query, 
            cpx_pos=data.cpx_pos, 
            node_attr_compose=h_cpx, 
            edge_index_q_cps_knn=pos_query_knn_edge_idx, 
            index_real_cps_edge_for_atten=data.index_real_cps_edge_for_atten, 
            tri_edge_index=data.tri_edge_index, 
            tri_edge_feat=data.tri_edge_feat,
            atom_type_emb=atom_type_emb,
            annealing=self.msg_annealing
            )
        ll_edge = (1/2 * (z_edge ** 2) - edge_log_jacob).mean()

        loss = torch.nan_to_num(ll_atom)\
             + torch.nan_to_num(loss_pos)\
             + torch.nan_to_num(ll_edge)\
             + torch.nan_to_num(pivotal_loss)\
             + torch.nan_to_num(surf_loss)\
             #+ torch.nan_to_num(loss_fake)\
             #+ torch.nan_to_num(loss_real)\
        
        out_dict = {
            'loss':loss, 
            'loss_atom':ll_atom, 'loss_edge':ll_edge, #'loss_fake':loss_fake, 'loss_real':loss_real,
            'loss_pos':loss_pos, 'pivotal_loss':pivotal_loss, 'surf_loss':torch.nan_to_num(surf_loss)
            }
        return out_dict