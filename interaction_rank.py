import torch
import pandas as pd

fname = "./output/feature_interaction_20221217.pth.tar"
checkpoint = torch.load(fname)
interaction = checkpoint["interaction"]

use_gpu = True
device = torch.device("cuda:1")



if use_gpu:
    # torch.cuda.set_device(device)
    # mlp = mlp.to(device)
    interaction = interaction.to(device)

print(interaction[0])
print(interaction[:, 0])

with torch.no_grad():

    feature_dim = interaction.shape[0]
    interaction = torch.triu(interaction, diagonal=1)
    interaction = interaction.reshape(-1)
    total_num = feature_dim * (feature_dim - 1) // 2
    interaction_sorted, interaction_rank = interaction.sort(descending=True)
    interaction_rank = interaction_rank[: total_num]
    interaction_sorted = interaction_sorted[: total_num]

    print(interaction_rank, interaction_sorted)
    interaction = interaction.reshape((feature_dim, feature_dim))
    ridx_buf, cidx_buf = interaction_rank // feature_dim, interaction_rank % feature_dim
    interaction_output = torch.cat([ridx_buf.unsqueeze(dim=1), cidx_buf.unsqueeze(dim=1), interaction_sorted.unsqueeze(dim=1)], dim=1)

    for idx in range(10):
        ridx, cidx = ridx_buf[idx], cidx_buf[idx]
        print(ridx, cidx, interaction[ridx, cidx])
        # print(cidx, ridx, interaction[cidx, ridx])

top_K = 10000

interaction_topK = interaction_output[:top_K].to(torch.device("cpu")).numpy()
pd.DataFrame(interaction_topK, columns=["Index1", "Index2", "Interaction Strength"]).to_csv(fname[:-8]+"_top_"+str(top_K)+".csv")

interaction_buttomK = interaction_output[-top_K:].to(torch.device("cpu")).numpy()
pd.DataFrame(interaction_buttomK, columns=["Index1", "Index2", "Interaction Strength"]).to_csv(fname[:-8]+"_buttom_"+str(top_K)+".csv")





# # Initializing Net
# mlp = MLP()  # num_genes=len(ds['gene_list_order']), n_class=ds['labels'].shape[0])
#
# state_dict = torch.load("model_epoch_9.pth")
# mlp.load_state_dict(state_dict)
# with torch.no_grad():
#
#     weight1 = mlp.layer1_weight()
#     grad_matrix0 = weight1.T.mm(grad_matrix).mm(weight1)




# print(interaction_buf)
# print(interaction_buf.shape)
# print(interaction_buf.mean())
# print(interaction_buf[interaction_buf > 0])