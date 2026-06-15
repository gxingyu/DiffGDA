import torch
import torch.nn.functional as F
import time
import tqdm
import numpy as np
from tqdm import tqdm, trange
from torch_geometric.loader import NeighborLoader
import torch.nn as nn
from pygda.models import BaseGDA
from pygda.utils import logger, MMD
from pygda.metrics import eval_macro_f1, eval_micro_f1
from model.dugda_base import DUGDABase
from models.guidanceNetwork import Guidance
from utilss.loader import (
    load_model_params,
    load_model_optimizer,
    load_loss_fn,
    load_sampling_fn
)
from torch_sparse import SparseTensor
from utilss.sample import get_subgraph, update
from utilss.guidance import train_domin_cls, get_density
class DUGDA(BaseGDA):
    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        config,
        num_layers=3,
        gnn='gcn',
        mode="node",
        dropout=0.2,
        act=F.relu,
        adv=None,
        alignment="mmd",
        weight=0.05,
        weight_decay=0.0005,
        lr=0.001,
        epoch=150,
        device="cuda:0",
        batch_size=16,
        num_neigh=-1,
        verbose=2,
        sample_epoch=1,
        s_pnums=0,
        t_pnums=10,
        alpha=0.1,
        x_limit=3,
        adj_limit=3,
        **kwargs,
    ):
        super(DUGDA, self).__init__(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            weight_decay=weight_decay,
            lr=lr,
            epoch=epoch,
            device=device,
            batch_size=batch_size,
            num_neigh=num_neigh,
            verbose=verbose,
            **kwargs,
        )
        if adv is not None:
            alignment = "adv" if adv else "mmd"
        if alignment not in {"mmd", "adv"}:
            raise ValueError("alignment must be either 'mmd' or 'adv'.")
        self.alignment = alignment
        self.adv = alignment == "adv"
        self.weight = weight
        self.mode = mode
        self.config = config
        self.kwargs = kwargs
        self.cls = None
        self.gnn = gnn
        self.sample_epoch = sample_epoch
        self.s_pnums = s_pnums
        self.t_pnums = t_pnums
        self.alpha = alpha
        self.x_limit = x_limit
        self.adj_limit = adj_limit

        self.params_x, self.params_adj = load_model_params(self.config)
        self.x_forward = nn.Linear(self.in_dim, 100).to(self.device)
        self.x_back = nn.Linear(100, self.in_dim).to(self.device)
        self.seed = 521070
        self.best_metrics = {"micro_f1": -1.0, "macro_f1": -1.0, "diff_epoch": -1}
        self.last_metrics = None
    def init_model(self, **kwargs):

        return DUGDABase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            adv=self.adv,
            dropout=self.dropout,
            act=self.act,
            mode=self.mode,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data, alpha):

        source_logits = self.cls(source_data, self.s_pnums)
        train_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = train_loss

        if self.mode == 'node':
            source_batch = None
            target_batch = None
        else:
            source_batch = source_data.batch
            target_batch = target_data.batch

        source_features = self.cls.feat_bottleneck(source_data.x, source_data.edge_index, source_batch, self.s_pnums)
        target_features = self.cls.feat_bottleneck(target_data.x, target_data.edge_index, target_batch, self.t_pnums)

        if self.alignment == "adv":
            source_dlogits = self.cls.domain_classifier(source_features, alpha)
            target_dlogits = self.cls.domain_classifier(target_features, alpha)
            
            domain_label = torch.tensor(
                [0] * source_data.x.shape[0] + [1] * target_data.x.shape[0]
                ).to(self.device)
            
            domain_loss = F.cross_entropy(torch.cat([source_dlogits, target_dlogits], 0), domain_label)
            loss = loss + self.weight * domain_loss
        else:
            mmd_loss = MMD(source_features, target_features)
            loss = loss + mmd_loss * self.weight

        target_logits = self.cls(target_data, self.t_pnums)

        return loss, source_logits, target_logits
    def predict(self, data, source=False):
        self.cls.eval()
        for idx, sampled_data in enumerate(self.target_loader):
            sampled_data = sampled_data.to(self.device)
            with torch.no_grad():
                logits = self.cls(sampled_data, self.t_pnums)

                if idx == 0:
                    logits, labels = logits, sampled_data.y
                else:
                    sampled_logits, sampled_labels = logits, sampled_data.y
                    logits = torch.cat((logits, sampled_logits))
                    labels = torch.cat((labels, sampled_labels))

        return logits, labels
    def fit(self, source_data, target_data):

        self.model_x, self.optimizer_x, self.scheduler_x = load_model_optimizer(
            self.params_x, self.config.train, self.device
        )

        self.model_adj, self.optimizer_adj, self.scheduler_adj = load_model_optimizer(
            self.params_adj, self.config.train, self.device
        )
        self.model_x = self.model_x.to(self.device)
        self.model_adj = self.model_adj.to(self.device)


        self.loss_fn = load_loss_fn(self.config)


        self.source_loader = NeighborLoader(
            source_data, self.num_neigh, batch_size=source_data.x.shape[0]
        )
        self.target_loader = NeighborLoader(
            target_data, self.num_neigh, batch_size=target_data.x.shape[0]
        )

        torch.manual_seed(self.seed)
        subgraph_node_nums=int(source_data.x.shape[0]*self.alpha)
        if subgraph_node_nums <= 0:
            subgraph_node_nums = 1
        subgraph_data, re_source_data = get_subgraph(subgraph_node_nums,source_data)
        target_subgraph_data, re_target_data = get_subgraph(subgraph_node_nums,target_data)
        subgraph_loader = NeighborLoader(
            subgraph_data, self.num_neigh, batch_size=self.batch_size
        )

        domin_cls = train_domin_cls(re_source_data, re_target_data, self.device, subgraph_data, target_subgraph_data)

        guidance_x = Guidance(self.config.guidance.x.in_dim, 
                              self.config.guidance.hid_dim, 
                              self.config.guidance.hid_dim,
                              self.config.guidance.out_dim
                              )
        guidance_x = guidance_x.to(self.device)
        guidance_adj = Guidance(self.config.guidance.adj.in_dim, 
                              self.config.guidance.hid_dim, 
                              self.config.guidance.hid_dim,
                              self.config.guidance.out_dim
                              )
        guidance_adj = guidance_adj.to(self.device)
        guidance_x_opt = torch.optim.Adam(guidance_x.parameters(), lr=0.01)
        guidance_adj_opt = torch.optim.Adam(guidance_adj.parameters(), lr=0.01)

        total_epoch = self.config.train.num_epochs

        for diff_epoch in trange(
            0, total_epoch, desc="[Epoch]", position=0, leave=True
        ):

            self.train_x = []
            self.train_adj = []


            self.model_x.train()
            self.model_adj.train()
            # -----------------------------------------------------------------------------------------------
            for idx, (sampled_source_data, sampled_target_data) in enumerate(
                zip(subgraph_loader, self.target_loader)
            ):


                self.optimizer_x.zero_grad()
                self.optimizer_adj.zero_grad()

                y_one_hot = F.one_hot(
                    sampled_source_data.y, num_classes=self.num_classes
                ).float()
                x = sampled_source_data.x
                x = self.x_forward(x)
                x = torch.cat([x, y_one_hot], dim=1).to(self.device)

                num_nodes = sampled_source_data.num_nodes
                adj = SparseTensor(
                    row=sampled_source_data.edge_index[0],
                    col=sampled_source_data.edge_index[1],
                    sparse_sizes=(num_nodes, num_nodes),
                ).to_dense()
                

                x = x.unsqueeze(0).repeat(100,1,1).to(self.device)
                adj = adj.unsqueeze(0).repeat(100,1,1).to(self.device)

                loss_subject = (x, adj)

                density = get_density(domin_cls, sampled_source_data)

                loss_x, loss_adj = self.loss_fn(
                    density, guidance_x, guidance_adj, guidance_x_opt, guidance_adj_opt, self.model_x, self.model_adj, *loss_subject
                )

                loss_x.backward(retain_graph=True)
                loss_adj.backward(retain_graph=True)





                self.optimizer_x.step()
                self.optimizer_adj.step()


                self.train_x.append(loss_x.item())
                self.train_adj.append(loss_adj.item())

            if self.config.train.lr_schedule:
                self.scheduler_x.step()
                self.scheduler_adj.step()

            mean_train_x = np.mean(self.train_x)
            mean_train_adj = np.mean(self.train_adj)

            tqdm.write(
                f"[EPOCH {diff_epoch + 1:04d}]  diffusion_train adj: {mean_train_adj} | diffusion_train x: {mean_train_x}"
            )
            flag = 0
            if diff_epoch >= flag:

                full_y_one_hot = F.one_hot(
                    source_data.y, num_classes=self.num_classes
                ).float()
                full_x = torch.cat([source_data.x, full_y_one_hot], dim=1).to(self.device)

                num_nodes = source_data.num_nodes
                full_adj = SparseTensor(
                    row=source_data.edge_index[0],
                    col=source_data.edge_index[1],
                    sparse_sizes=(num_nodes, num_nodes),
                ).to_dense()


                self.sampling_fn = load_sampling_fn(
                    self.config, self.config.sampler, self.config.sample, self.device, subgraph_node_nums
                )
                updated_source_data = source_data
                for i in range(self.sample_epoch):
                    x, adj, _ = self.sampling_fn(guidance_x, guidance_adj, self.model_x, self.model_adj, None)

                    x = x.squeeze(0)
                    adj = adj.squeeze(0)

                    x_last_5 = x[:, -self.num_classes:]

                    x_softmax = F.softmax(x_last_5, dim=1)

                    _, indices = torch.max(x_softmax, dim=1)

                    y = indices
                    x = x[:, :(-self.num_classes)]
                    x = self.x_back(x)

                    edge_index = torch.nonzero(adj > self.adj_limit, as_tuple=False).t()
                    x = torch.where(x > self.x_limit, torch.ones_like(x), torch.zeros_like(x))

                    updated_source_data = update(updated_source_data, x, edge_index, y, subgraph_data).to(self.device)

                self.train_loader = NeighborLoader(
                    updated_source_data, self.num_neigh, batch_size=updated_source_data.num_nodes
                )

                self.cls = self.init_model(**self.kwargs)
                optimizer = torch.optim.Adam(
                    self.cls.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay
                )
                # cls training
                start_time = time.time()
                for epoch in range(self.epoch):
                    epoch_loss = 0
                    epoch_train_logits = None
                    epoch_train_labels = None

                    p = float(epoch) / self.epoch
                    alpha = 2. / (1. + np.exp(-10. * p)) - 1

                    for idx, (sampled_train_data, sampled_target_data) in enumerate(zip(self.train_loader, self.target_loader)):
                        self.cls.train()
                        loss, train_logits, target_logits = self.forward_model(sampled_train_data, sampled_target_data, alpha)
                        epoch_loss += loss.item()

                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()

                        if idx == 0:
                            epoch_train_logits, epoch_train_labels = train_logits, sampled_train_data.y
                        else:   
                            train_logits, train_labels = train_logits, sampled_train_data.y
                            epoch_train_logits = torch.cat((epoch_train_logits, train_logits))
                            epoch_train_labels = torch.cat((epoch_train_labels, train_labels))
                    
                    epoch_train_preds = epoch_train_logits.argmax(dim=1)
                    micro_f1_score = eval_micro_f1(epoch_train_labels, epoch_train_preds)
                    
                    logger(epoch=epoch,
                        loss=epoch_loss,
                        source_train_acc=micro_f1_score,
                        time=time.time() - start_time,
                        verbose=self.verbose,
                        train=True)
                # evaluate the performance
                logits, labels = self.predict(target_data)
                preds = logits.argmax(dim=1)
                mi_f1 = eval_micro_f1(labels, preds)
                ma_f1 = eval_macro_f1(labels, preds)
                self.last_metrics = {
                    "micro_f1": float(mi_f1),
                    "macro_f1": float(ma_f1),
                    "diff_epoch": diff_epoch + 1,
                }
                if mi_f1 > self.best_metrics["micro_f1"]:
                    self.best_metrics = dict(self.last_metrics)
                print("target:  ")
                print('[epoch] '+ str(diff_epoch + 1) + ' micro-f1: ' + str(mi_f1) + ' macro-f1: ' + str(ma_f1))

    def process_graph(self, data):
        pass

    
