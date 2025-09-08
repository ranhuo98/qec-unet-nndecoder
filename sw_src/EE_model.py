import os 
import copy
import json
from typing import Tuple

from matplotlib import pyplot as plt
from utils import report_num_params
import torch
import brevitas
from tqdm import tqdm, trange
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import (
    Int8Bias,
    Int8ActPerTensorFloat, 
    SignedBinaryWeightPerTensorConst, 
    SignedTernaryWeightPerTensorConst,
    SignedBinaryActPerTensorConst
)
from brevitas.quant_tensor import IntQuantTensor
from brevitas.quant_tensor import QuantTensor
from Train import (
    data_packer 
)

from qecsim import paulitools as pt
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.rotatedplanar import (
    RotatedPlanarCode, 
    RotatedPlanarMPSDecoder
    )

set_weight_bit_width = 4
set_activation_bit_width = 4

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            
            qnn.QuantConv2d(in_channels, out_channels, kernel_size = 3, weight_bit_width=set_weight_bit_width,
                            padding = 1, bias=False, return_quant_tensor=True),
            #qnn.QuantConv2d(in_channels, out_channels, kernel_size = 3, weight_quant=SignedBinaryWeightPerTensorConst,
            #                padding = 1, bias=False, return_quant_tensor=True),
            nn.BatchNorm2d(out_channels),
            qnn.QuantReLU(bit_width=set_activation_bit_width, return_quant_tensor=True),
            
            qnn.QuantConv2d(out_channels, out_channels, kernel_size = 3, weight_bit_width=set_weight_bit_width,
                            padding = 1, bias=False, return_quant_tensor=True),
            #qnn.QuantConv2d(out_channels, out_channels, kernel_size = 3, weight_quant=SignedBinaryWeightPerTensorConst,
            #                padding = 1, bias=False, return_quant_tensor=True),
            nn.BatchNorm2d(out_channels),
            qnn.QuantReLU(bit_width=set_activation_bit_width, return_quant_tensor=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class Encoder(nn.Module):
    def __init__(self, scale, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            
            nn.MaxPool2d(scale), 
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Upsampling(nn.Module):
    def __init__(self, scale, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.Sequential(
            
            qnn.QuantConv2d(in_channels, out_channels, kernel_size=1, weight_bit_width=set_weight_bit_width, 
                            bias=False, return_quant_tensor=True),
            #qnn.QuantConv2d(in_channels, in_channels//2, kernel_size=1, weight_quant=SignedBinaryWeightPerTensorConst, 
            #                bias=False, return_quant_tensor=True),
            nn.BatchNorm2d(out_channels),
            qnn.QuantReLU(bit_width=set_activation_bit_width, return_quant_tensor=True),
            qnn.QuantUpsamplingNearest2d(scale_factor=scale, return_quant_tensor=True)
        )
        self.quant_inp = qnn.QuantIdentity(act_quant = Int8ActPerTensorFloat, bit_width=set_activation_bit_width, return_quant_tensor=True)

    def forward(self, x1, x2):
        # x1 is the feature map after upsampling
        # x2 is the feature map from the corresponding layer in the downsampling path for concatenation
        x1 = self.up(x1)
        x1 = self.quant_inp(x1)
        x2 = self.quant_inp(x2)
        x = x1 + x2
        x = self.quant_inp(x)
        
        return x



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = qnn.QuantConv2d(in_channels, out_channels, kernel_size = 1, weight_bit_width=set_weight_bit_width, bias=False)
        #self.conv = qnn.QuantConv2d(in_channels, out_channels, kernel_size = 1, weight_quant=SignedBinaryWeightPerTensorConst, bias=False)

    def forward(self, x):
        return self.conv(x)
    

class EA_LAYER_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.quant_inp = qnn.QuantIdentity(bit_width=4, return_quant_tensor=True)
        self.double_conv1 = DoubleConv(in_channels, out_channels)
        self.double_conv5 = DoubleConv(out_channels, 8)
        self.output_conv6 = OutConv(8, 4)
    
    def forward(self, x):
        x = self.quant_inp(x)
        x1 = self.double_conv1(x)
        x = self.double_conv5(x1)
        x = self.output_conv6(x)

        return x, x1
        




class EA_LAYER_2(nn.Module):
    def __init__(self, previous_model, in_channels, out_channels):
        super().__init__()

        self.quant_inp = previous_model.quant_inp
        self.double_conv1 = previous_model.double_conv1
        self.double_conv5 = previous_model.double_conv5
        self.output_conv6 = previous_model.output_conv6

        self.encode2 = Encoder(2, in_channels, out_channels) # (16, 32)
        self.double_conv4_1 = DoubleConv(out_channels, in_channels)
        self.upsample4_2 = Upsampling(2, 16, 16)
    
    def forward(self, x):
        x = self.quant_inp(x)
        x1 = self.double_conv1(x)
        x = self.encode2(x1)
        x = self.double_conv4_1(x)
        x = self.upsample4_2(x, x1)
        x = self.double_conv5(x)
        x = self.output_conv6(x)

        return x, x1
    


class EA_LAYER_3(nn.Module):
    def __init__(self, previous_model, in_channels, out_channels):
        super().__init__()

        self.quant_inp = previous_model.quant_inp
        self.double_conv1 = previous_model.double_conv1
        self.encode2 = previous_model.encode2
        self.double_conv4_1 = previous_model.double_conv4_1
        self.upsample4_2 = previous_model.upsample4_2
        self.double_conv5 = previous_model.double_conv5
        self.output_conv6 = previous_model.output_conv6

        self.encode3 = Encoder(3, in_channels, out_channels) # (32, 64)
        self.upsample3 = Upsampling(3, 64, 32)
    
    def forward(self, x):
        x = self.quant_inp(x)
        x1 = self.double_conv1(x)
        x2 = self.encode2(x1)
        x = self.encode3(x2)
        x = self.upsample3(x, x2)
        x = self.double_conv4_1(x)
        x = self.upsample4_2(x, x1)
        x = self.double_conv5(x)
        x = self.output_conv6(x)

        return x, x1
    


class EA_LAYER_3_inference(nn.Module):
    """
    Inference‐mode wrapper with per‐sample early exit after the first conv block.
    """
    def __init__(self, previous_model: nn.Module, ee_threshold: float):
        super().__init__()
        # Copy pretrained submodules
        self.quant_inp      = previous_model.quant_inp
        self.double_conv1   = previous_model.double_conv1
        self.encode2        = previous_model.encode2
        self.encode3        = previous_model.encode3
        self.upsample3      = previous_model.upsample3
        self.double_conv4_1 = previous_model.double_conv4_1
        self.upsample4_2    = previous_model.upsample4_2
        self.double_conv5   = previous_model.double_conv5
        self.output_conv6   = previous_model.output_conv6

        self.ee_threshold = ee_threshold

    def _clone_intqt(self, x: IntQuantTensor) -> IntQuantTensor:
        """
        Make a deep copy of an IntQuantTensor by cloning its .value and
        preserving all its quantization metadata.
        """
        return IntQuantTensor(
            value       = x.value.clone(),
            scale       = x.scale,
            zero_point  = x.zero_point,
            bit_width   = x.bit_width,
            signed      = x.signed,
            training    = x.training,
        )
    
    def _slice_intqt(self, x: IntQuantTensor, idx) -> IntQuantTensor:
        """
        Return a sliced IntQuantTensor (by batch indices) preserving metadata.
        """
        sliced_val = x.value[idx]
        return IntQuantTensor(
            value       = sliced_val,
            scale       = x.scale,
            zero_point  = x.zero_point,
            bit_width   = x.bit_width,
            signed      = x.signed,
            training    = x.training,
        )


    def _ee_condition_batch(self, x1: IntQuantTensor) -> torch.BoolTensor:
        """
        Compute a (B,) mask: True if sample should exit early (no squared-sum above threshold).
        """
        # 0) pull out the float Tensor
        x = x1.value                   # (B, C, H, W)

        # 1) sum over C → (B, 1, H, W)
        x_sum = x.sum(dim=1, keepdim=True)

        # 2) flatten H,W → (B, H*W)
        B = x_sum.size(0)
        x_flat = x_sum.view(B, -1)

        # 3) square, then count how many > threshold per sample → (B,)
        # count = (x_flat.pow(2) > self.ee_threshold).sum(dim=1)
        count = (x_flat > self.ee_threshold).sum(dim=1)

        # 4) early‐exit if zero entries exceed threshold
        return count == 0

    def forward(self, x: torch.Tensor) -> Tuple[IntQuantTensor, IntQuantTensor, torch.BoolTensor]:
        """
        Args:
          x: input images (B, …)

        Returns:
          out:        final logits (B, Cout, Hout, Wout)
          xIM:        intermediate IntQuantTensor after first block (B, C1, H1, W1)
          exit_mask:  BoolTensor of shape (B,) where True means ‘exited early’
        """
        # 1) Initial quant + first conv block
        x0 = self.quant_inp(x)
        x1 = self.double_conv1(x0)

        # Clone x1 so we can return it unchanged as xIM
        xIM = self._clone_intqt(x1)

        # 2) Decide early‐exit per sample
        exit_mask = self._ee_condition_batch(x1)  # (B,)

        # 3) Prepare a “combined features” IntQuantTensor we can overwrite
        combined = self._clone_intqt(x1)

        # 4) Process the “deeper” path only for the non‐exit indices
        nonexit_idx = (~exit_mask).nonzero(as_tuple=True)[0]
        if nonexit_idx.numel() > 0:
            xd = self._slice_intqt(x1, nonexit_idx)
            x2 = self.encode2(xd)
            x3 = self.encode3(x2)
            u3 = self.upsample3(x3, x2)
            x4 = self.double_conv4_1(u3)
            u4 = self.upsample4_2(x4, xd)
            # Overwrite combined.value for non-exit samples
            combined.value[nonexit_idx] = u4.value

        # Final conv on all samples
        out = self.double_conv5(combined)
        out = self.output_conv6(out)

        return out, xIM, exit_mask


    

def label_to_binary(d, error_label):
    # error_label should have the shape: (batch_size, d, d)
    batch_size = error_label.shape[0]
    error_label = error_label[:, :5, :5]
    error_label = error_label.reshape(batch_size, -1) # shape: (batch_size, d * d)
    #convert label to binary to make the array compatible with the following bsp operation
    error_binary = np.zeros((batch_size, d * d * 2), dtype=int)
    error_binary[:, :d * d] = np.where((error_label == 1) | (error_label == 3), 1, 0)
    error_binary[:, d * d:] = np.where((error_label == 2) | (error_label == 3), 1, 0)

    return error_binary


def logical_test(d, E_predict, E_target):

    correct_total = 0
    correct_indices = []
    incorrect_indices = []
    
    code = RotatedPlanarCode(d, d)
    code_logicals = np.array([[[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 1],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]],

                            [[0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0]]])

    E_predict_binary = label_to_binary(d, E_predict)
    E_target_binary = label_to_binary(d, E_target)
    Errors = np.stack((E_predict_binary, E_target_binary), axis = 1) # shape: (batch_size, 2, 2*d*d)
    for index, error in enumerate(Errors):
	logical_predict = pt.bsp(error[0] ^ error[1], code_logicals.reshape(2, 2* d * d).T)

        if logical_predict[0] == 0 & logical_predict[1] == 0:
            correct_total += 1
            correct_indices.append(index)
        else:
            incorrect_indices.append(index)  # Store incorrect sample index

    return correct_total, correct_indices, incorrect_indices



def train(device, model, d, train_loader, optimizer, criterion):
    losses = []
    correct = 0
    total = 0
    # ensure model is in training mode
    model.train()    
    
    for i, (input, target, _) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()   
                
        # forward pass
        output, _ = model(input)
        loss = criterion(output, target)
        
        # backward pass + run optimizer to update weights
        loss.backward()
        optimizer.step()

        # keep track of loss value
        losses.append(loss.data.cpu().numpy())

        # get the predicted errors on physical data qubuts
        _, predicted = output.max(dim=1)
        correct_num, _, _ = logical_test(d, predicted, target)
        correct += correct_num
        total += target.size(0)
            
    accuracy = correct / total 

    return losses, accuracy



def validate(device, model, d, test_loader, criterion):
    # For validate, we use the same dataset as the beginning because we want to test our model in a general case.

    losses = []
    # ensure model is in training mode
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for i, (input, target) in enumerate(test_loader):        
            input, target = input.to(device), target.to(device) 
                    
            # forward pass
            output, _ = model(input)
            loss = criterion(output, target)    
            # keep track of loss value
            losses.append(loss.data.cpu().numpy()) 
            # test accuracy
            # get the predicted errors on physical data qubuts
            _, predicted = output.max(dim=1)
            # compare the predicted logical errors with the target logical errors
            correct_num, _, _ = logical_test(d, predicted, target)
            correct += correct_num
            
            total += target.size(0)
            
    accuracy = correct / total

    return losses, accuracy




class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Check if sample is a tuple or list.
        if isinstance(sample, (tuple, list)):
            # Only use the first two elements (data and target)
            # and ignore any additional elements.
            data, target = sample[:2]
        else:
            raise ValueError("Dataset item is not a tuple or list.")
        
        # Return the index along with the data and target.
        return data, target, idx
    


def separate_dataloader(device, model, d, train_loader):
    model.eval()  # Set model to evaluation mode

    correct_indices = []
    incorrect_indices = []

    correct_tensorIMs = []
    incorrect_tensorIMs = []

    indexed_dataset = IndexedDataset(train_loader.dataset)
    train_loader = DataLoader(indexed_dataset, batch_size=train_loader.batch_size, shuffle=True)

    with torch.no_grad():  # No gradient calculation needed for inference
        for batch_idx, (inputs, targets, batch_indices) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Get model predictions
            outputs, tensorIM = model(inputs)
            _, predicted = outputs.max(dim=1)

            _, batch_correct_idx, batch_incorrect_idx = logical_test(d, predicted.cpu().numpy(), targets.cpu().numpy())

            # Convert the batch indices to a list (if not already)
            if isinstance(batch_indices, torch.Tensor):
                batch_indices = batch_indices.cpu().tolist()
            
            # Use the returned batch indices from logical_test to index into the actual dataset indices.
            correct_batch_indices = [batch_indices[i] for i in batch_correct_idx]
            incorrect_batch_indices = [batch_indices[i] for i in batch_incorrect_idx]

            correct_indices.extend(correct_batch_indices)
            incorrect_indices.extend(incorrect_batch_indices)

            # Convert QuantTensor to a standard tensor before indexing
            tensorIM_tensor = tensorIM.tensor if hasattr(tensorIM, "tensor") else tensorIM  # Extract tensor from QuantTensor

            # Use advanced indexing to extract tensorIM values
            if len(batch_correct_idx) > 0:
                correct_tensorIMs.extend(tensorIM_tensor[batch_correct_idx].cpu().numpy())  
            if len(batch_incorrect_idx) > 0:
                incorrect_tensorIMs.extend(tensorIM_tensor[batch_incorrect_idx].cpu().numpy())


    # Get the dataset from DataLoader
    dataset = train_loader.dataset

    # Create subsets for correct and incorrect samples
    correct_subset = Subset(dataset, correct_indices)
    incorrect_subset = Subset(dataset, incorrect_indices)

    # Create new DataLoaders
    correct_dataloader = DataLoader(correct_subset, batch_size=train_loader.batch_size, shuffle=True)
    incorrect_dataloader = DataLoader(incorrect_subset, batch_size=train_loader.batch_size, shuffle=True)

    print(f"Correct Samples: {len(correct_indices)}")
    print(f"Incorrect Samples: {len(incorrect_indices)}")

    return correct_dataloader, incorrect_dataloader, correct_tensorIMs, incorrect_tensorIMs



# def run_train(model, save_dir, save_dir_withZero, d = 5, error_rate = 0.01, samples = 10, num_epochs = 20, train_size = 10000, test_size = 2000, batch_size = 1024, lr = 0.001, step_size = 50, gamma = 1):
def load_dataloader(config, train_dataLoader = None):

    # configure the train using config file
    save_dir = config["save_dir"]
    save_dir_withZero = config["save_dir_withZero"]
    # model_save_path = os.path.join(config["model_save_path"], "model.pth")
    d = config["d"]
    error_rate = config["error_rate"]
    samples = config["samples"]
    num_epochs = config["num_epochs"]
    train_size = config["train_size"]
    test_size = config["test_size"]
    batch_size = config["batch_size"]
    # lr = config["lr"]
    # step_size = config["step_size"]
    # gamma = config["gamma"]

    train_quantized_loader, test_quantized_loader = data_packer(save_dir, save_dir_withZero, d, error_rate, samples, train_size, test_size, batch_size)
    if train_dataLoader is not None:
        print("Using the provided DataLoader for training.")
        train_quantized_loader = train_dataLoader
    if not isinstance(train_quantized_loader.dataset, IndexedDataset):
        train_quantized_loader = DataLoader(IndexedDataset(train_quantized_loader.dataset), batch_size=batch_size, shuffle=True)
        print("Convert to IndexedDataset.")
    print(f"Train size: {len(train_quantized_loader.dataset)}.")
    print(f"Test size: {len(test_quantized_loader.dataset)}.")

    return train_quantized_loader, test_quantized_loader

def run_train(model_class, model_args, config, train_quantized_loader, test_quantized_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Target device: " + str(device))

    d = config["d"]
    model_save_path = os.path.join(config["model_save_path"], "model.pth")
    num_epochs = config["num_epochs"]
    lr = config["lr"]
    model = model_class(*model_args).to(device)
    report_num_params(model)
    # Loss criterion and optimizer
    class_weights = torch.tensor([0.9,  1.2,  1.2, 1.2])
    criterion = nn.CrossEntropyLoss(weight = class_weights).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999)
    )
    # scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # Adjust gamma for learning rate decay
    
    
    # Track losses and accuracy
    running_loss_train = []
    running_loss_validate = []
    running_validation_acc = []
    running_train_acc = []
    best_acc = 0.0  # Track the highest accuracy

    # Training loop
    t = trange(num_epochs, desc="Progress", leave=True)
    for epoch in t:
        loss_epoch_train, train_accuracy= train(device, model, d, train_quantized_loader, optimizer, criterion)
        loss_epoch_validation, validation_accuracy = validate(device, model, d, test_quantized_loader, criterion)
        # scheduler.step()
        
        # Update metrics
        mean_train_loss = np.mean(loss_epoch_train)
        mean_validation_loss = np.mean(loss_epoch_validation)
        running_loss_train.append(mean_train_loss)
        running_loss_validate.append(mean_validation_loss)
        running_train_acc.append(train_accuracy)
        running_validation_acc.append(validation_accuracy)

        print(
                f"--Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss = {mean_train_loss:.6f}, "
                f"Validation Loss = {mean_validation_loss:.6f}, "
                f"Train Accuracy = {train_accuracy:.6f}, "
                f"Validation Accuracy = {validation_accuracy:.6f}"
            )
    model_save_path = os.path.join(config["model_save_path"], f"{model_class.__name__}_Train{train_accuracy}_Validation{validation_accuracy}.pth")
    torch.save(model.state_dict(), model_save_path)
        
    print("Training complete.")

    dataLoader_True, dataLoader_False, tensorIM_True, tensorIM_False = separate_dataloader(device, model, d, train_quantized_loader)

    return model, tensorIM_True, tensorIM_False, dataLoader_True, dataLoader_False



def separateDataset_basedOn_errorCount(model, config):

    # Dictionary to store separated tensors
    separated_data = {}

    _, test_dataLoader = data_packer(config["save_dir"], config["save_dir_withZero"], config["d"], config["error_rate"], config["samples"], 100, 200000, 200000)
    with torch.no_grad():
        for _, (input, target) in enumerate(test_dataLoader):

            outputs, tensorIM = model(input)
            # tensorIM[0] is the data we need, the rest is the metadata of IntQuantTensor class
            tensorIM = tensorIM[0].cpu().numpy()
            
            # Count the number of non-zero values for each sample
            non_zero_counts = (target != 0).sum(dim=(1, 2))  # Sum over both dimensions (6x6)

            # Categorize outputs by non-zero count
            for i, count in enumerate(non_zero_counts):
                count = count.item()  # Convert tensor to int
                # Initialize lists for each category if not already present
                if count not in separated_data:
                    separated_data[count] = {"inputs": [], "targets": [], "tensorIMs": []}

                # Store the corresponding input and target
                separated_data[count]["inputs"].append(input[i].cpu().numpy())  # Convert to NumPy
                separated_data[count]["targets"].append(target[i].cpu().numpy())
                separated_data[count]["tensorIMs"].append(tensorIM[i])
    
    # (Optional) Print summary information for each group
    for count, data in sorted(separated_data.items()):
        print(f"Non-zero count: {count}  ->  Number of samples: {len(data['inputs'])}")
    
    return separated_data


def aggregate_and_flatten(tensorIM_list):
    """
    For a list of arrays with shape (16, 6, 6), sum along axis 0 (the 16 matrices) 
    to get an array of shape (1, 6, 6), then squeeze and flatten into a 1D array.
    Returns a concatenated 1D array of all values.
    """
    # Sum each array over the first dimension to get shape (1,6,6), then flatten and concatenate.
    summed_matrices = [np.sum(matrix, axis=0, keepdims=True) for matrix in tensorIM_list]
    # summed_matrices_flatten = np.power(np.concatenate([arr.flatten() for arr in summed_matrices]), 2)
    summed_matrices_flatten = np.concatenate([arr.flatten() for arr in summed_matrices])
    summed_matrices_flatten = summed_matrices_flatten.reshape(-1, 36)  # Reshape to (1, 36)

    return summed_matrices_flatten


def find_thresh(model, config):

    separated_data = separateDataset_basedOn_errorCount(model, config)
    aggregated_data = {}  # To store aggregated 1D arrays for each group

    # Iterate over each group key in separated_data (e.g., group keys like 0, 4, etc.)
    for group_key in sorted(separated_data.keys()):
        tensorIM_list = separated_data[group_key]["tensorIMs"]
        aggregated_data[group_key] = aggregate_and_flatten(tensorIM_list)
        print(f"Group {group_key} aggregated data shape: {aggregated_data[group_key].shape}")

    baseline_max = np.max(aggregated_data[0])  # Arbitrary value for the baseline max
    print(f"Baseline max value: {baseline_max}")

    return baseline_max


def run(device, config, mode, test_dataLoader, thresh):

    e1_modelPostTraining = EA_LAYER_1(2, 16)
    e2_modelPostTraining = EA_LAYER_2(e1_modelPostTraining, 16, 32)
    e3_modelPostTraining = EA_LAYER_3(e2_modelPostTraining, 32, 64)

    if mode == "train":
        train_quantized_loader, test_quantized_loader = load_dataloader(config)
        e1_modelPostTraining, True_e1_tensorIM, False_e1_tensorIM, True_e1, False_e1 = run_train(EA_LAYER_1, (2, 16), config, train_quantized_loader, test_quantized_loader)
        e2_modelPostTraining, True_e2_tensorIM, False_e2_tensorIM, True_e2, False_e2 = run_train(EA_LAYER_2, (e1_modelPostTraining, 16, 32), config, train_quantized_loader, test_quantized_loader)
        e3_modelPostTraining, True_e3_tensorIM, False_e3_tensorIM, True_e3, False_e3 = run_train(EA_LAYER_3, (e2_modelPostTraining, 32, 64), config, train_quantized_loader, test_quantized_loader)
    elif mode == "inference":
        model_save_path = os.path.join(config["model_save_path"], config["model_name"])
        print("Loading model from: ", model_save_path)
        e3_modelPostTraining.load_state_dict(torch.load(model_save_path, map_location=device))
        e3_model_EE = EA_LAYER_3_inference(e3_modelPostTraining, thresh)

        correct_ee = 0
        correct_deep = 0
        total = 0
        total_ee = 0
        ee_count = 0
        ee_accuracy = 0
        with torch.no_grad():
            e3_model_EE.eval()
            for _, (input, target) in enumerate(test_dataLoader):
                correct_num_ee = 0
                correct_num_deep = 0
                no_ee_correct_num = 0
                output, tensorIM, ee_condition = e3_model_EE(input)
                no_ee_output, _ = e3_modelPostTraining(input)
                _, predicted = output.max(dim=1)
                _, no_ee_predicted = no_ee_output.max(dim=1)
                no_ee_correct_num, _, _ = logical_test(config["d"], no_ee_predicted, target)
                ee_idx = ee_condition.nonzero(as_tuple=True)[0]
                deep_idx = (~ee_condition).nonzero(as_tuple=True)[0]
                ee_count += ee_idx.numel()
                total_ee = ee_count
                if ee_idx.numel() > 0:   
                    correct_num_ee, _, _ = logical_test(config["d"], predicted[ee_idx], target[ee_idx])
                if deep_idx.numel() > 0:  
                    correct_num_deep, _, _ = logical_test(config["d"], predicted[deep_idx], target[deep_idx])
                correct_ee += correct_num_ee
                correct_deep += correct_num_deep
                total += target.size(0)
        if total_ee > 0:
            ee_accuracy = correct_ee / total_ee
        total_accuracy = (correct_ee + correct_deep) / total
        no_ee_accuracy = no_ee_correct_num / total
        ee_probability = total_ee / total
        print("Total samples: ", total)
        print("Total early exit samples: ", total_ee)
        print("ee_probability: ", ee_probability)
        print("ee_accuracy: ", ee_accuracy)
        print("total_accuracy: ", total_accuracy)
        print("no_ee_accuracy: ", no_ee_accuracy)
        
    return ee_probability, ee_accuracy, total_accuracy, no_ee_accuracy

