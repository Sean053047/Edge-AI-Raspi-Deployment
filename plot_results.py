import os
import json
encoder = 'vitb'
result_dir = f'results/{encoder}'
results_dict = dict()
bits_set = set()
gsize_set = set()
options = ['rmse', 'd1', 'd2', 'd3', 'abs_rel']
lower_is_better = [True, False, False, False, True]

for pth in sorted(os.listdir(result_dir)):
    if pth.endswith('.png'): continue
    with open(os.path.join(result_dir, pth), 'r') as f:
        r = json.load(f)
    if 'origin' in pth:
        results_dict['origin'] = r
    else:
        pth = pth.strip('.json')
        bits, gsize = int(pth.split('_')[1]), int(pth.split('_')[-1])
        bits_set.add(bits)
        gsize_set.add(gsize)
        results_dict[(bits, gsize)] = r

bits_labels = sorted(list(bits_set))
gsize_labels = sorted(list(gsize_set))


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

for opt, lsb in zip(options, lower_is_better):
    plt.figure(figsize=(10, 6))
    data = np.zeros((len(bits_labels), len(gsize_labels)))
    for i, bits in enumerate(bits_labels):
        for j, gsize in enumerate(gsize_labels):
            if (bits, gsize) not in results_dict:
                continue
            r = results_dict[(bits, gsize)]
            data[i, j] = r[opt]
    cmap = 'viridis' if not lsb else 'viridis_r'
    ax = sns.heatmap(data, annot=True, fmt='.4f', cmap=cmap, xticklabels=gsize_labels, yticklabels=bits_labels)
    ax.set_title(f'{encoder} {opt}| Origin value: {results_dict["origin"][opt]:.4f}')
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Bits')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{encoder}_{opt}.png'))
    plt.close()
