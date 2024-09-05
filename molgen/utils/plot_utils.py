import os
from typing import Dict, List

import matplotlib.pyplot as plt

def save_plot(data: Dict[str, List[float]], path="/mobileye/VAL_ARCH/org/MolGen/plots"):
    for title, plt_data in data.items():
        plt.plot(range(len(plt_data)), plt_data)

        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Mean CrossEntropyLoss")
        # plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.savefig(os.path.join(path, f"{title}.png"))
        plt.clf()