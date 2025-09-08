from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Set, Tuple
import stim
import numpy as np

import os 
import copy
import glob
import argparse
from utils import report_num_params
import torch
import brevitas
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset, TensorDataset
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import (
    Int8Bias,
    Int8ActPerTensorFloat, 
    SignedBinaryWeightPerTensorConst, 
    SignedTernaryWeightPerTensorConst,
    SignedBinaryActPerTensorConst
)

from stim import PauliString
from circuit_generator import *

import matplotlib.pyplot as plt
import json
from itertools import combinations
import collections

from qecsim import paulitools as pt
from qecsim.models.rotatedplanar import RotatedPlanarCode



def print_2D(values: Dict[complex, Any]):
    assert all(v.real == int(v.real) for v in values), "All real parts must be integers"
    assert all(v.imag == int(v.imag) for v in values), "All imaginary parts must be integers"
    assert all(v.real >= 0 for v in values), "All real parts must be non-negative"
    assert all(v.imag >= 0 for v in values), "All imaginary parts must be non-negative"
    w = int(max((v.real for v in values), default = 0 ) + 1)
    h = int(max((v.imag for v in values), default = 0 ) + 1)
    s = ""
    for y in range(h):
        for x in range(w):
            s += str(values.get(x + y * 1j, '_'))
        s += "\n"
    print(s)

def torus(c: complex, distance: int) -> complex:
    r = c.real % (distance * 2)
    i = c.imag % (distance * 2)
    return r + i * 1j

@dataclass
class EdgeType:
    pauli: str
    top_left_delta: complex
    top_right_delta: complex
    bottom_left_delta: complex
    bottom_right_delta: complex

@dataclass
class NoiseType:
    after_clifford_depolarization: float
    after_reset_flip_probability: float
    before_measure_flip_probability: float
    before_round_data_depolarization: float


def generate_circuit(distance: int, rounds: int, error_rates: float) -> stim.Circuit:

    # find hex_centers (aka ancilla qubits coordinates)
    hex_centers : Dict[complex, int] = {}
    for row in range(distance + 1):
        for col in range(distance + 1):
            center = row * 2j + col * 2
            category = 2 - ((row + col) % 2)
            hex_centers[center] = category
    
    for k in [k for k, v in hex_centers.items()
                if (v == 2 and (k.imag == 0 or k.imag == distance * 2))
                    or ((k.imag // 2) % 2 == 1 and k.real == 0)
                    or ((k.imag // 2) % 2 == 0 and k.real == distance * 2)]:
        del hex_centers[k]


    edge_types = [
        EdgeType(pauli = 'X', top_left_delta = -1j - 1, top_right_delta = -1j + 1, bottom_left_delta = 1j - 1, bottom_right_delta = 1j + 1),
        EdgeType(pauli = 'Z', top_left_delta = -1j - 1, top_right_delta = -1j + 1, bottom_left_delta = 1j - 1, bottom_right_delta = 1j + 1)
    ]

    data_qubits_coordinates: Set[complex] = set()
    for h in hex_centers:
        for edge_type in edge_types:
            q = h + edge_type.top_left_delta
            data_qubits_coordinates.add(torus(q, distance))
            q = h + edge_type.top_right_delta
            data_qubits_coordinates.add(torus(q, distance))
            q = h + edge_type.bottom_left_delta
            data_qubits_coordinates.add(torus(q, distance))
            q = h + edge_type.bottom_right_delta
            data_qubits_coordinates.add(torus(q, distance))

    fused_dict = dict(hex_centers)
    for q in data_qubits_coordinates:
        fused_dict[q] = 'q'

    print_2D(fused_dict)

    q2i : Dict[complex, int] = {q: i+1 for i, q in enumerate(
        sorted(fused_dict, key=lambda v: (v.imag, v.real)))}
    

    delta_names = [
        'bottom_right_delta',
        'bottom_left_delta',
        'top_right_delta',
        'top_left_delta',
    ]

    edge_groups: Dict[str, Dict[str, Set[Tuple[complex, complex]]]] = {
        name: {'X': set(), 'Z': set()}
        for name in delta_names
    }

    for r in (1, 2):  # 1 → X, 2 → Z
        category = edge_types[r - 1]
        pauli = category.pauli  # 'X' or 'Z'
        relevant_hexes = [h for h, cat in hex_centers.items() if cat == r]
        for h in relevant_hexes:
            q0 = h
            for name in delta_names:
                delta = getattr(category, name)
                qn = h + delta
                if qn in fused_dict:
                    # Order the pair for X vs Z
                    pair = (q0, qn) if pauli == 'X' else (qn, q0)
                    # Store under the outer delta key, inner pauli key
                    edge_groups[name][pauli].add(pair)

    round_circuits = []
    circuit = stim.Circuit()
    x_qubits = [q2i[k] for k, v in fused_dict.items() if v == 1]
    z_qubits = [q2i[k] for k, v in fused_dict.items() if v == 2]
    data_qubits = sorted([q2i[k] for k, v in fused_dict.items() if v == 'q'])

    print(f"X qubits: {x_qubits}")
    print(f"Z qubits: {z_qubits}")
    print(f"Data qubits: {data_qubits}")

    noise = NoiseType(
        after_clifford_depolarization=error_rates,
        after_reset_flip_probability=2/3*error_rates,
        before_measure_flip_probability=2/3*error_rates,
        before_round_data_depolarization=0
    )

    circuit.append_operation('TICK')
    # add before round data qubit noise here if any
    circuit.append_operation("DEPOLARIZE1", data_qubits, noise.before_round_data_depolarization)
    circuit.append_operation('H', x_qubits)
    # add single-gate noise here
    circuit.append_operation('DEPOLARIZE1', x_qubits, noise.after_clifford_depolarization)
    circuit.append_operation('TICK')
    for k, v in edge_groups.items():
        for k2, v2 in v.items():
            for pair in v2:
                circuit.append_operation('CNOT', [q2i[pair[0]], q2i[pair[1]]])
        for k2, v2 in v.items():
            for pair in v2:
                # add two-qubit gate noise here
                circuit.append_operation("DEPOLARIZE2", [q2i[pair[0]], q2i[pair[1]]], noise.after_clifford_depolarization)
        circuit.append_operation('TICK')

    circuit.append_operation('H', x_qubits)
    # add single-gate noise here
    circuit.append_operation('DEPOLARIZE1', x_qubits, noise.after_clifford_depolarization)
    circuit.append_operation('TICK')
    # add before measure noise here
    circuit.append_operation('X_ERROR', x_qubits + z_qubits, noise.before_measure_flip_probability)
    circuit.append_operation('MR', x_qubits + z_qubits)
    circuit.append_operation("X_ERROR", x_qubits + z_qubits, noise.after_reset_flip_probability)

    round_circuits.append(circuit)

    measurements = len(x_qubits + z_qubits)
    print(f"Measurements: {measurements}")


    det_circuits = []
    circuit = stim.Circuit()
    x_qubits_coords = [k for k, v in fused_dict.items() if v == 1]
    z_qubits_coords = [k for k, v in fused_dict.items() if v == 2]
    data_qubits_coords = [k for k, v in fused_dict.items() if v == 'q']
    circuit.append_operation("SHIFT_COORDS", [], [0, 0, 1])
    for i, q in enumerate(x_qubits_coords + z_qubits_coords):
        record_targets = []
        record_targets.append(stim.target_rec(i - measurements))
        record_targets.append(stim.target_rec(i - measurements - measurements))
        circuit.append_operation("DETECTOR", record_targets, [q.imag, q.real, 0])
    det_circuits.append(circuit)

    initial_circuits = stim.Circuit()
    circuit = stim.Circuit()
    initial_det_circuits = []
    circuit.append_operation('R', x_qubits + z_qubits)
    # add after reset noise here
    circuit.append_operation("X_ERROR", x_qubits + z_qubits, noise.after_reset_flip_probability)
    circuit.append_operation('R', data_qubits)
    # add after reset noise here
    circuit.append_operation("X_ERROR", data_qubits, noise.after_reset_flip_probability)
    initial_det_circuits.append(circuit)
    circuit = stim.Circuit()
    for i, q in enumerate(x_qubits_coords + z_qubits_coords):
        record_targets = []
        record_targets.append(stim.target_rec(i - measurements))
        circuit.append_operation("DETECTOR", record_targets, [q.imag, q.real, 0])

    initial_circuits = initial_det_circuits[0] + round_circuits[0] + circuit


    stable_circuits = stim.Circuit()
    stable_circuits = round_circuits[0] + det_circuits[0]

    measurements_data_qubits = len(data_qubits)
    print(f"Measurements data qubits: {measurements_data_qubits}")
    end_circuits = []
    circuit = stim.Circuit()
    circuit.append_operation('M', data_qubits)
    record_targets = []
    for i, q in enumerate(data_qubits_coords):
        record_targets.append(stim.target_rec(i - measurements_data_qubits))
        # circuit.append_operation("DETECTOR", record_targets[-1], [q.imag, q.real, 1])
    circuit.append_operation("OBSERVABLE_INCLUDE", record_targets[0:distance])
    end_circuits.append(circuit)


    full_circuits = stim.Circuit()
    for q, i in q2i.items():
        full_circuits.append_operation("QUBIT_COORDS", [i], [q.imag, q.real])
    full_circuits += initial_circuits + stable_circuits * (rounds - 1) + end_circuits[0]

    return full_circuits, x_qubits_coords, z_qubits_coords, data_qubits_coords


def formatting_detector_samples_2_ancilla_matrix(d, rounds, circuit_noise, shots):

    circuit, x_qubits_coords, z_qubits_coords, data_qubits_coords = generate_circuit(distance=d, rounds=rounds, error_rates=circuit_noise)
    print(circuit)

    det_sampler = circuit.compile_detector_sampler()
    det_samples, obserables = det_sampler.sample(shots = shots, separate_observables = True)

    _, total_cols = det_samples.shape

    block_size = total_cols // rounds
    samples = det_samples.reshape(shots, rounds, block_size)

    x_rows = np.array([int(q.imag//2) for q in x_qubits_coords])
    x_cols = np.array([int(q.real//2) for q in x_qubits_coords])
    z_rows = np.array([int(q.imag//2) for q in z_qubits_coords])
    z_cols = np.array([int(q.real//2) for q in z_qubits_coords])

    grid_shape = (shots, rounds, d+1, d+1)
    X_ancilla = np.zeros(grid_shape, dtype=int)
    Z_ancilla = np.zeros(grid_shape, dtype=int)

    X_ancilla[:, 1:, x_rows, x_cols] = samples[:, 1:, :block_size // 2]
    Z_ancilla[:, :, z_rows, z_cols] = samples[:, :, block_size // 2:]

    combined = np.stack([X_ancilla, Z_ancilla], axis=2)
    ancilla_qubits = (
        combined
        .transpose(0, 2, 1, 3, 4)              # (shots, 2, rounds, d+1, d+1)
        .reshape(shots, rounds * 2, d+1, d+1)  # (shots, rounds*2, d+1, d+1)
    )

    return ancilla_qubits, obserables


set_weight_bit_width = 8
set_activation_bit_width = 8

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
    def __init__(self, d, in_channels, out_channels):
        super().__init__()
        self.conv = qnn.QuantConv2d(in_channels, out_channels, kernel_size = 1, weight_bit_width=set_weight_bit_width, bias=False)
        self.flatten = nn.Flatten()
        self.activation = qnn.QuantReLU(bit_width=set_activation_bit_width, return_quant_tensor=True)
        self.linear = qnn.QuantLinear(out_channels*(d+1)*(d+1), 1, weight_bit_width=set_weight_bit_width, bias=False)
        #self.conv = qnn.QuantConv2d(in_channels, out_channels, kernel_size = 1, weight_quant=SignedBinaryWeightPerTensorConst, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
        # return self.conv(x)
    
    
class Unet_circuit_2L(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=set_activation_bit_width, return_quant_tensor=True)
        self.double_conv1 = DoubleConv(6, 16)
        self.encode2 = Encoder(2, 16, 32) # (16, 32)
        self.upsample4_2 = Upsampling(2, 32, 16)
        self.double_conv5 = DoubleConv(16, 8)
        self.output_conv6 = OutConv(d, 8, 4)
    
    def forward(self, x):
        x = self.quant_inp(x)
        x1 = self.double_conv1(x)
        x = self.encode2(x1)
        x = self.upsample4_2(x, x1)
        x = self.double_conv5(x)
        x = self.output_conv6(x)

        return x, x1
    
    
class Unet_circuit_3L(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=set_activation_bit_width, return_quant_tensor=True)
        self.double_conv1 = DoubleConv(10, 16)
        self.encode2 = Encoder(2, 16, 32) # (16, 32)
        self.encode3 = Encoder(3, 32, 64)
        self.upsample3 = Upsampling(3, 64, 32)
        self.double_conv4_1 = DoubleConv(32, 16)
        self.upsample4_2 = Upsampling(2, 16, 16)
        self.double_conv5 = DoubleConv(16, 8)
        self.output_conv6 = OutConv(d, 8, 4)
    
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
    
    
class Unet_circuit_4L(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=set_activation_bit_width, return_quant_tensor=True)
        self.double_conv1 = DoubleConv(14, 16)
        self.encode2 = Encoder(2, 16, 32) # (16, 32)
        self.encode3 = Encoder(2, 32, 64)
        self.encode4 = Encoder(2, 64, 128)
        self.upsample4 = Upsampling(2, 128, 64)
        self.double_conv5_1 = DoubleConv(64, 32)
        self.upsample5_2 = Upsampling(2, 32, 32)
        self.double_conv6_1 = DoubleConv(32, 16)
        self.upsample6_2 = Upsampling(2, 16, 16)
        self.double_conv7 = DoubleConv(16, 8)
        self.output_conv8 = OutConv(d, 8, 4)
    
    def forward(self, x):
        x = self.quant_inp(x)
        x1 = self.double_conv1(x)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x = self.encode4(x3)
        x = self.upsample4(x, x3)
        x = self.double_conv5_1(x)
        x = self.upsample5_2(x, x2)
        x = self.double_conv6_1(x)
        x = self.upsample6_2(x, x1)
        x = self.double_conv7(x)
        x = self.output_conv8(x)

        return x, x1
    
    
def train(device, model, d, train_loader, optimizer, criterion):
    losses = []
    correct = 0
    total = 0
    # ensure model is in training mode
    model.train()    
    
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        target = target.float()  # Convert target to float

        optimizer.zero_grad()   
                
        # forward pass
        output, _ = model(input)
        loss = criterion(output, target)
        
        # backward pass + run optimizer to update weights
        loss.backward()
        optimizer.step()

        # keep track of loss value
        losses.append(loss.data.cpu().numpy())

        probabilities = torch.sigmoid(output)
        predicted_labels = (probabilities > 0.5).float()

        correct += (predicted_labels == target).sum().item()
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
            target = target.float()  # Convert target to float
            # forward pass
            output, _ = model(input)
            loss = criterion(output, target)    
            # keep track of loss value
            losses.append(loss.data.cpu().numpy()) 
            # test accuracy
            probabilities = torch.sigmoid(output)
            predicted_labels = (probabilities > 0.5).float()
            batch_accuracy = (predicted_labels == target).float().mean().item()
            correct += (predicted_labels == target).sum().item()
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


def data_loader(observable_measurement, ancilla_qubits, dataset_size = 100, batch_size = 256, train_size = 1000, test_size = 1000):

    observable_measurement = observable_measurement.reshape(dataset_size, 1)

    ancilla_tensor = torch.tensor(ancilla_qubits, dtype=torch.float32)
    data_tensor = torch.tensor(observable_measurement, dtype=torch.long)


    # generate dataset for training
    random_subset_ancilla_tensor_train = ancilla_tensor[:train_size]
    random_subset_data_tensor_train = data_tensor[:train_size]

    train_quantized_dataset = TensorDataset(random_subset_ancilla_tensor_train, random_subset_data_tensor_train)

    # generate dataset for test
    random_subset_ancilla_tensor_test = ancilla_tensor[-test_size:]
    random_subset_data_tensor_test = data_tensor[-test_size:]

    test_quantized_dataset = TensorDataset(random_subset_ancilla_tensor_test, random_subset_data_tensor_test)

    # dataset loaders
    train_quantized_loader = DataLoader(train_quantized_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_quantized_loader = DataLoader(test_quantized_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_quantized_loader, test_quantized_loader


def load_dataloader(config, data_qubits, ancilla_qubits, train_dataLoader = None):

    # configure the train using config file
    train_size = config["train_size"]
    test_size = config["test_size"]
    batch_size = config["batch_size"]
    dataset_size = config["dataset_size"]

    train_quantized_loader, test_quantized_loader = data_loader(data_qubits, ancilla_qubits, dataset_size, batch_size, train_size, test_size)
    
    print(f"Train size: {len(train_quantized_loader.dataset)}.")
    print(f"Test size: {len(test_quantized_loader.dataset)}.")

    return train_quantized_loader, test_quantized_loader


def run_train(config, d, noise):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Target device: " + str(device))

    print(json.dumps(config, indent=4))
    print("Noise level: ", noise)

    weight_quant = config["weight_quant"]
    act_quant = config["act_quant"]
    d = d
    model_save_path = os.path.join(config["model_save_path"], "model.pth")
    num_epochs = config["num_epochs"]
    lr = config["lr"]
    dataset_size = config["dataset_size"]
    rounds = d
    noise = noise
    step_size = config["step_size"]
    gamma = config["gamma"]
    run_index = 0

    # dataset generation, should be done only once
    ancilla_path = f"./ancilla_qubits_d{d}_r{rounds}_noise{noise}.npy"
    observable_path = f"./observable_measurement_d{d}_r{rounds}_noise{noise}.npy"
    if os.path.exists(ancilla_path) and os.path.exists(observable_path):
        ancilla_qubits = np.load(ancilla_path)
        observable_measurement = np.load(observable_path)
        print("Loaded ancilla qubits and observable measurement from files.")
    else:
        print("Generating ancilla qubits and observable measurement (this may take a while)...")
        ancilla_qubits, observable_measurement = formatting_detector_samples_2_ancilla_matrix(d, rounds, noise, dataset_size)
        np.save(ancilla_path, ancilla_qubits)
        np.save(observable_path, observable_measurement)
        print("Saved dataset to files.")
    print("Ancilla qubits shape: ", ancilla_qubits.shape)
    print("Observable measurement shape: ", observable_measurement.shape)

    train_quantized_loader, test_quantized_loader = load_dataloader(config, observable_measurement, ancilla_qubits)
    print("Finished generating dataloader")

    load_model_path = f"./detector_sample_model/D{d}_Noise{noise}_*_W{set_weight_bit_width}A{set_activation_bit_width}.pth"
    load_model_path = glob.glob(load_model_path)
    load_model_path = load_model_path[0] 
    print("Load model path: ", load_model_path)

    if d == 3:
        model = Unet_circuit_2L(d)
    elif d == 5:
        model = Unet_circuit_3L(d)
    elif d == 7:
        model = Unet_circuit_4L(d)

    if os.path.exists(load_model_path):
        model.load_state_dict(torch.load(load_model_path, map_location=device))
        print("Loading model:", load_model_path)
    else:
        print("No model found, training from scratch.")

    model.to(device)
    report_num_params(model)

    # Loss criterion and optimizer
    class_weights = torch.tensor([1, 1, 1, 1])
    criterion = nn.CrossEntropyLoss(weight = class_weights).to(device)
    # criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999)
    )
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # Adjust gamma for learning rate decay
   
    
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
        scheduler.step()
        
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
        
    
    print("Training complete.")
    model_save_path = f"./detector_sample_model/D{d}_Noise{noise}_Train{train_accuracy}_Validation{validation_accuracy}_W{set_weight_bit_width}A{set_activation_bit_width}_{run_index+1}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
