import torch
import random

# Importing your dataset class
from molgen.datasets.smiles_dataset import PreTrainDecisionGPTSmilesDataset

# Mock tokenizer class to bypass the BOS/EOS property checks
class MockTokenizer:
    def __init__(self):
        self.bos_token_id = 9
        self.eos_token_id = 10
        self.pad_token_id = 8
        self.unk_token_id = 11

    def encode(self, smiles, return_tensors=False):
        # Maps characters to dummy IDs so that the precomputation step can run
        mapping = {"C": 0, "O": 1, "N": 2, "H": 3, "(": 4, ")": 5, "=": 6, "1": 7}
        encoded = [mapping.get(char, 11) for char in smiles]
        return [encoded]

# Mock reward function class
class MockReward:
    def __init__(self, value):
        self.value = value
    def __call__(self, smiles):
        return self.value

def test_my_randomization_directly():
    print("--- Direct sanity check for the __getitem__ randomization logic ---")
    
    tokenizer = MockTokenizer()
    smiles_list = ["CCO", "CNC", "CC(=O)O", "C1CCCCC1", "CC(=O)Nc1ccc(cc1)O"]
    reward_functions = [MockReward(0.8), MockReward(2.5), MockReward(0.5)]
    
    dataset = PreTrainDecisionGPTSmilesDataset(
        smiles=smiles_list,
        tokenizer=tokenizer,
        reward_func=reward_functions,
        string_type="SMILES"
    )
    
    # Direct validation step-by-step from the dataset without DataLoader
    for idx in range(len(dataset)):
        print(f"\n[Direct call to molecule index: {idx + 1}]")
        
        # Pulling the dynamic sample from your __getitem__
        sample = dataset[idx]
        
        filtered_rtgs = sample["rtgs"]
        selected_goals = sample["goal_idx"]
        
        print(f"-> goal_idx (randomly selected): {selected_goals}")
        print(f"-> Number of selected goals: {len(selected_goals)}")
        print(f"-> Number of rtgs returned in the list: {len(filtered_rtgs)}")
        
        # Check if the number of RTG tracks matches the number of sampled goals
        if len(selected_goals) == len(filtered_rtgs):
            print("V SUCCESS: The number of RTGs perfectly matches the number of goals!")
        else:
            print("X ERROR: Mismatch found between the selected goals and the RTG lists structure.")

if __name__ == "__main__":
    test_my_randomization_directly()