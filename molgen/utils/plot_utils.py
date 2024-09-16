import os
from typing import Dict, List, Optional

import wandb
import matplotlib.pyplot as plt


def save_plot(
        data: Dict[str, List[float]],
        path: Optional[str] = None
):
    """
    Generate a matplotlib plot of the data and save the graph to path
    :param data: Dictionary of data to plot
    :param path: Path to save the plot
    """
    if path is None:
        path = os.path.join(os.getcwd(), "plots")
    for title, plt_data in data.items():
        plt.plot(range(len(plt_data)), plt_data)

        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Mean CrossEntropyLoss")
        # plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.savefig(os.path.join(path, f"{title}.png"))
        plt.clf()


def log_metrics_to_wandb(
        wandb_key: str,
        project_name: str,
        project_entity: str,
        training_config: Dict[str, float],
        run_name: str = None,
):
    """
    This method will log the provided metrics dictionary to wandb project
    :param wandb_key: API key
    :param project_name: wandb project name where metrics will be logged. default name is set in consts.py
    :param project_entity: entity name under which to find the wandb project
    :param training_config: dictionary of training hyperparameter configuration to log evaluation results
    :param run_name: parameter used to name the run in wandb. default is None in which case wandb will assign a name
    """
    wandb.login(key=wandb_key)
    wandb.init(project=project_name, entity=project_entity, config=training_config, name=run_name)
    return wandb.run
