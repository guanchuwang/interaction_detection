import os
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import autograd
import torch.nn.functional as F

import datasets

# from sklearn.metrics import f1_score
# from mlp import MLP, load_dataset_from_arrow
import time


class MLP(nn.Module):

    def __init__(self, num_genes=27985
                 , hidden1=2048
                 , hidden2=512
                 , n_class=177):
        super().__init__()
        # self.layers = nn.Sequential(
        #     nn.Linear(num_genes, hidden1),
        #     nn.ReLU(),
        #     nn.Linear(hidden1, hidden2),
        #     nn.ReLU(),
        #     nn.Linear(hidden2, n_class)
        # )
        self.layers = nn.Sequential(
            nn.Linear(num_genes, hidden1),
            nn.Softplus(),
            nn.Linear(hidden1, hidden2),
            nn.Softplus(),
            nn.Linear(hidden2, n_class)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

    def forward_plus(self, x):
        '''Forward pass'''

        x1 = self.layers[0](x)
        y = self.layers[1:](x1)

        return y, x1

    def layer1_weight(self):

        return self.layers[0].weight


def get_second_order_grad(model, x, class_idx):
    # x = torch.FloatTensor(x).to(device)
    # print(x.dtype)

    with torch.set_grad_enabled(True):

        if x.nelement() < 2:
            return np.array([])

        x.requires_grad = True
        output, x1 = model.forward_plus(x)
        y = torch.gather(output, 1, class_idx).sum(dim=0)

        grads = autograd.grad(y, x1, create_graph=True, allow_unused=True)[0].sum(dim=0)  # [valid_index]

        grad_list = []
        for j, grad in enumerate(grads):
            grad2 = autograd.grad(grad, x1, retain_graph=True)[0] # .sum(dim=0)  # [valid_index]
            grad_list.append(grad2.detach())

    with torch.no_grad():

        grad_matrix = torch.stack(grad_list).permute((1, 0, 2))
        del grad_list
        grad_matrix0 = torch.zeros((27985, 27985), device=x.device)
        weight1 = mlp.layer1_weight()
        for idx in range(grad_matrix.shape[0]):
            grad_matrix0 += weight1.T.mm(grad_matrix[idx]).mm(weight1).abs()

    # print(grad_matrix0[0:2])

    return grad_matrix0


# def get_second_order_grad2(model, x, class_idx):
#
#     # x = torch.FloatTensor(x).to(device)
#     # print(x.dtype)
#
#     with torch.set_grad_enabled(True):
#
#         if x.nelement() < 2:
#             return np.array([])
#
#         x.requires_grad = True
#         output = model(x)
#         y = output[:, class_idx]
#
#         grads = autograd.grad(y, x, create_graph=True, allow_unused=True)[0].squeeze() # [valid_index]
#
#
#         grad_list = []
#         for j, grad in enumerate(grads):
#             grad2 = autograd.grad(grad, x, retain_graph=True)[0].squeeze() # [valid_index]
#             print(grad2.abs())
#             # hegsns
#
#             if j == 1:
#
#                 return
#
#             grad_list.append(grad2.detach())
#
#     with torch.no_grad():
#
#         grad_matrix = torch.stack(grad_list)
#         # weight1 = model.layer1_weight(valid_index)
#         # grad_matrix0 = weight1.T.mm(grad_matrix).mm(weight1)
#
#     return torch.abs(grad_matrix)



@torch.no_grad()
def distance_estimation(x, reference):

    distance_vector = torch.abs(x - reference).squeeze(dim=0)
    distance_i, distance_j = torch.meshgrid(distance_vector, distance_vector)
    distance_matrix = distance_i * distance_j

    return distance_matrix


# @torch.no_grad()
# def error_term_estimation(mlp, x, class_idx, reference=0, device=torch.device("cpu")):
#     # interaction_scores = {}
#
#     valid_index = torch.where(x > reference)[1]
#     grad_matrix_1 = get_second_order_grad(mlp, x, class_idx, valid_index)
#     # distance_matrix = distance_estimation(x[:, valid_index], reference)
#     # error_matrix = grad_matrix_1 * distance_matrix * (1-torch.eye(distance_matrix.shape[0], device=device))
#     # error_matrix = grad_matrix_1 * (1-torch.eye(distance_matrix.shape[0]))
#
#     # print(grad_matrix_1.mean())
#     # print(distance_matrix.mean())
#     # print(error_matrix.mean())
#
#     # return error_matrix, valid_index
#     return grad_matrix_1, valid_index


def save_checkpoint(fname, **kwargs):

    checkpoint = {}

    for key, value in kwargs.items():
        checkpoint[key] = value
        # setattr(self, key, value)

    torch.save(checkpoint, fname)


if __name__ == '__main__':
    batch_size = 512 # 1 #
    use_gpu = True
    device = torch.device("cuda:1")
    # device2 = torch.device("cuda:2")
    # device3 = torch.device("cuda:3")
    print_every = 10
    data_dir = './data'
    # output_p = '/home/zli17/work/projects/scRNA_cell_type_prediction/output'
    # set fixed random seed
    torch.manual_seed(7747)
    # ds = load_dataset_from_arrow(data_dir=data_dir)
    ds = datasets.load_from_disk("./data")
    ds_list = ds.to_pandas()["gex"].values
    ds_torch = torch.from_numpy(np.concatenate(ds_list, axis=0)).type(torch.float)
    ds_torch = ds_torch.reshape(len(ds_list), -1)
    trainloader = DataLoader(ds_torch, batch_size=batch_size, shuffle=False)
    # # evalloader = DataLoader(ds['validation'].with_format("torch"), batch_size=batch_size)
    # label2id = {l:i for i, l in enumerate(ds['labels'])}
    # id2label = {i:l for i, l in enumerate(ds['labels'])}
    
    # Initializing Net
    mlp = MLP() # num_genes=len(ds['gene_list_order']), n_class=ds['labels'].shape[0])

    state_dict = torch.load("model_epoch_9.pth")
    mlp.load_state_dict(state_dict)

    if use_gpu:
        # torch.cuda.set_device(device)
        mlp = mlp.to(device)

    print(mlp)
    # print(next(mlp.parameters()).device)
    print(((ds_torch > 0).sum(dim=1)/len(ds_torch[0]))[:20])
    print(ds_torch.shape)
    # print(ds_torch.mean(dim=0))
    # hegsns
    # print(pos_ratio)
    #
    # hegsns
    # training loop

    # print(ds.to_pandas()["gex"].values)
    # print(type(ds.to_pandas()["gex"]))
    # print(ds_torch.shape)
    # print(trainloader)

    feature_dim = ds_torch.shape[1]
    instance_num = len(trainloader)
    interaction_buf = torch.zeros((feature_dim, feature_dim), device=device)
    # interaction_cnt = torch.zeros((2048, 2048), device=device)
    # interaction_buf_tmp = torch.zeros((feature_dim, feature_dim), device=device2)
    # interaction_cnt_tmp = torch.zeros((feature_dim, feature_dim), device=device2)
    test_num = 0

    torch.cuda.synchronize()
    t0 = time.time()

    mlp.eval()
    # for i, inputs in enumerate(trainloader):

    with torch.no_grad():
        for index, inputs in enumerate(trainloader):
            # get data and ground truth
            # inputs, targets = data['gex'], data['label']
            # targets = torch.tensor([label2id[t] for t in targets])
            # if use_gpu:
            #     inputs, targets = inputs.cuda(), targets.cuda()
            # forward pass of data through net
            torch.cuda.synchronize()
            t2 = time.time()

            if use_gpu:
                inputs = inputs.to(device)

            outputs = mlp(inputs)
            class_idx = outputs.argmax(dim=1).unsqueeze(dim=1)
            test_num = index+1

            interaction = get_second_order_grad(mlp, inputs, class_idx)

            interaction_buf += interaction
            max_interaction = interaction_buf.max()

            torch.cuda.synchronize()
            t1 = time.time()
            print("Index {}, Max {}, Batch time {}, remain time {}".format(index, max_interaction, (t1 - t2), (t1 - t0)/(index+1)*(instance_num-index-1)))

        # if index == 100: # instance_num-1:
        #     break


    # print(interaction_buf.mean())
    # print(interaction_buf[interaction_buf > 0])

    save_checkpoint("./output/feature_interaction.pth.tar",
                    interaction=interaction_buf,
                    test_num=test_num,
                    )

