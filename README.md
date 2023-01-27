## Interaction Detection

### Dependency

```
torch
numpy
datasets
```

### Guidance

Change the target model and deployed cuda device, and run the command:
```
python interaction_detection.py
```
After the program ends, it will generate _feature_interaction.pth.tar_ in the _output_ folder. 
Then, run the following command:
```
python interaction_rank.py
```
It will generate _feature_interaction_top_10000.csv_ and _feature_interaction_20221216_buttom_10000.csv_ in _output_ folder.
The format is [feature_index, feature_index, SCORE], for example, [3581 16170 5838.744] means the features 3581 and 16170 are top candidates with score 5838.744, where 3581 and 16170 corresponding to the column number of input (counts from zero). 

### Attention
Replace the torch.nn.Relu() activation function into the torch.nn.Softplus() function before detecting the interaction.

