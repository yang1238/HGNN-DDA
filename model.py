import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
import csv


class MultiDeep(nn.Module):
    def __init__(self,ndisease, ndrug, ndiseaseHIN, ndiseaseESM, ndrugHIN, ndrugPC, nhid_GAT, nheads_GAT, nhid_MHSA, nheads_MHSA, alpha):
        """Dense version of GAT."""
        super(MultiDeep, self).__init__()

        self.disease_attentions1 = [GraphAttentionBiLSTMConvolution(ndiseaseESM, nhid_GAT) for _ in range(nheads_GAT)]
        for i, attention in enumerate(self.disease_attentions1):
            self.add_module('Attention_Disease1_{}'.format(i), attention)
        self.disease_MultiHead1 = [selfattention(nhid_GAT * nheads_GAT + ndiseaseHIN + ndiseaseESM, nhid_MHSA, nhid_GAT * nheads_GAT + ndiseaseHIN + ndiseaseESM) for _ in range(nheads_MHSA)]
        for i, attention in enumerate(self.disease_MultiHead1):
            self.add_module('Self_Attention_Disease1_{}'.format(i), attention)
        self.disease_prolayer1 = nn.Linear(nhid_GAT * nheads_GAT, nhid_GAT * nheads_GAT, bias=False)
        self.disease_LNlayer1 = nn.LayerNorm(nhid_GAT * nheads_GAT + ndiseaseHIN + ndiseaseESM)

        self.drug_attentions1 = [GraphAttentionBiLSTMConvolution(ndrugPC, nhid_GAT) for _ in range(nheads_GAT)]
        for i, attention in enumerate(self.drug_attentions1):
            self.add_module('Attention_Drug1_{}'.format(i), attention)
        self.drug_MultiHead1 = [selfattention(nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC, nhid_MHSA, nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC) for _ in range(nheads_MHSA)]
        for i, attention in enumerate(self.drug_MultiHead1):
            self.add_module('Self_Attention_Drug1_{}'.format(i), attention)
        self.drug_prolayer1 = nn.Linear(nhid_GAT * nheads_GAT, nhid_GAT * nheads_GAT, bias=False)
        self.drug_LNlayer1 = nn.LayerNorm(nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC)

        self.FClayer1 = nn.Linear(nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC + nhid_GAT * nheads_GAT + ndiseaseHIN + ndiseaseESM, (nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC) * 2)
        self.FClayer2 = nn.Linear((nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC) * 2, (nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC) * 2)
        self.FClayer3 = nn.Linear((nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC) * 2, 1)
        self.output = nn.Sigmoid()

    def forward(self, diseaseHIN, diseaseESM, disease_adj, drugHIN, drug_PC, drug_adj, idx_disease_drug, device):
        diseasex = torch.cat([att(diseaseESM, disease_adj) for att in self.disease_attentions1], dim=1)
        diseasex = self.disease_prolayer1(diseasex)
        diseasemax = torch.cat([diseasex, diseaseESM, diseaseHIN],dim=1)
        temp = torch.zeros_like(diseasemax)
        for selfatt in self.disease_MultiHead1:
            temp = temp + selfatt(diseasemax)
        diseasex = temp +diseasemax
        diseasex = self.disease_LNlayer1(diseasex)

        drugx = torch.cat([att(drug_PC, drug_adj) for att in self.drug_attentions1], dim=1)
        drugx = self.drug_prolayer1(drugx)
        drugmax = torch.cat([drugx, drug_PC, drugHIN],dim=1)
        temp = torch.zeros_like(drugmax)
        for selfatt in self.drug_MultiHead1:
            temp = temp + selfatt(drugmax)
        drugx = temp + drugmax
        drugx = self.drug_LNlayer1(drugx)


        disease_drug_x = torch.cat((diseasex[idx_disease_drug[:, 0]], drugx[idx_disease_drug[:, 1]]), dim=1)
        disease_drug_x = disease_drug_x.to(device)
        disease_drug_x = self.FClayer1(disease_drug_x)
        disease_drug_x = F.relu(disease_drug_x)
        disease_drug_x = self.FClayer2(disease_drug_x)
        disease_drug_x = F.relu(disease_drug_x)
        disease_drug_x = self.FClayer3(disease_drug_x)
        disease_drug_x = disease_drug_x.squeeze(-1)
        return disease_drug_x


