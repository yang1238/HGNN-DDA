import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
import csv


class MultiDeep(nn.Module):
    def __init__(self,nprotein, ndrug, nproteinHIN, nproteinESM, ndrugHIN, ndrugPC, nhid_GAT, nheads_GAT, nhid_MHSA, nheads_MHSA, alpha):
        """Dense version of GAT."""
        super(MultiDeep, self).__init__()

        self.protein_attentions1 = [GraphAttentionBiLSTMConvolution(nproteinESM, nhid_GAT) for _ in range(nheads_GAT)]
        for i, attention in enumerate(self.protein_attentions1):
            self.add_module('Attention_Protein1_{}'.format(i), attention)
        self.protein_MultiHead1 = [selfattention(nhid_GAT * nheads_GAT + nproteinHIN + nproteinESM, nhid_MHSA, nhid_GAT * nheads_GAT + nproteinHIN + nproteinESM) for _ in range(nheads_MHSA)]
        for i, attention in enumerate(self.protein_MultiHead1):
            self.add_module('Self_Attention_Protein1_{}'.format(i), attention)
        self.protein_prolayer1 = nn.Linear(nhid_GAT * nheads_GAT, nhid_GAT * nheads_GAT, bias=False)
        self.protein_LNlayer1 = nn.LayerNorm(nhid_GAT * nheads_GAT + nproteinHIN + nproteinESM)

        self.drug_attentions1 = [GraphAttentionBiLSTMConvolution(ndrugPC, nhid_GAT) for _ in range(nheads_GAT)]
        for i, attention in enumerate(self.drug_attentions1):
            self.add_module('Attention_Drug1_{}'.format(i), attention)
        self.drug_MultiHead1 = [selfattention(nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC, nhid_MHSA, nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC) for _ in range(nheads_MHSA)]
        for i, attention in enumerate(self.drug_MultiHead1):
            self.add_module('Self_Attention_Drug1_{}'.format(i), attention)
        self.drug_prolayer1 = nn.Linear(nhid_GAT * nheads_GAT, nhid_GAT * nheads_GAT, bias=False)
        self.drug_LNlayer1 = nn.LayerNorm(nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC)

        self.FClayer1 = nn.Linear(nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC + nhid_GAT * nheads_GAT + nproteinHIN + nproteinESM, (nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC) * 2)
        self.FClayer2 = nn.Linear((nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC) * 2, (nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC) * 2)
        self.FClayer3 = nn.Linear((nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC) * 2, 1)
        self.output = nn.Sigmoid()

    def forward(self, proteinHIN, proteinESM, protein_adj, drugHIN, drug_PC, drug_adj, idx_protein_drug, device):
        proteinx = torch.cat([att(proteinESM, protein_adj) for att in self.protein_attentions1], dim=1)
        proteinx = self.protein_prolayer1(proteinx)
        proteinmax = torch.cat([proteinx, proteinESM, proteinHIN],dim=1)
        temp = torch.zeros_like(proteinmax)
        for selfatt in self.protein_MultiHead1:
            temp = temp + selfatt(proteinmax)
        proteinx = temp +proteinmax
        proteinx = self.protein_LNlayer1(proteinx)

        drugx = torch.cat([att(drug_PC, drug_adj) for att in self.drug_attentions1], dim=1)
        drugx = self.drug_prolayer1(drugx)
        drugmax = torch.cat([drugx, drug_PC, drugHIN],dim=1)
        temp = torch.zeros_like(drugmax)
        for selfatt in self.drug_MultiHead1:
            temp = temp + selfatt(drugmax)
        drugx = temp + drugmax
        drugx = self.drug_LNlayer1(drugx)


        protein_drug_x = torch.cat((proteinx[idx_protein_drug[:, 0]], drugx[idx_protein_drug[:, 1]]), dim=1)
        protein_drug_x = protein_drug_x.to(device)
        protein_drug_x = self.FClayer1(protein_drug_x)
        protein_drug_x = F.relu(protein_drug_x)
        protein_drug_x = self.FClayer2(protein_drug_x)
        protein_drug_x = F.relu(protein_drug_x)
        protein_drug_x = self.FClayer3(protein_drug_x)
        protein_drug_x = protein_drug_x.squeeze(-1)
        return protein_drug_x

