import argparse
import torch
import torch.nn
import time
import gtn
import gtn.criterion

from time_utils import time_func


parser = argparse.ArgumentParser("CTC Benchmark with GTN")

parser.add_argument("-T", "--num_time_steps", type=int, required=False, default=150)
parser.add_argument("-L", "--target_size", type=int, required=False, default=35)
parser.add_argument("-B", "--batch_size", type=int, required=False, default=32)
parser.add_argument("-C", "--num_alphabets", type=int, required=False, default=10001)

parser.add_argument("-N", "--num_iters", type=int, required=False, default=25)

args = parser.parse_args()

T, L, C, B = args.num_time_steps, args.target_size, args.num_alphabets, args.batch_size
input = torch.randn(T, B, C).log_softmax(2).detach().requires_grad_().cuda()
target = torch.randint(low=1, high=C, size=(B, L), dtype=torch.long)
input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long)
target_lengths = torch.full(size=(B,), fill_value=L, dtype=torch.long)

N = args.num_iters

############  TORCH BENCHMARK  ############

print("Running torch ctc benchmark ...")


def pytorch_ctc_func():
    ctc_loss = torch.nn.CTCLoss()
    loss = ctc_loss(input, target, input_lengths, target_lengths)
    loss.backward()


time_func(pytorch_ctc_func, N, "PyTorch CTC ", iscuda=True)

############  GTN BENCHMARK  ############

# Since forward score is not implemented on cuda, we currently run the
# GTN code on CPU by moving the data between CPU <--> GPU

print("Running gtn ctc benchmark ...")


def process(b):
    # create emission graph; moves data from GPU -> CPU
    g_emissions = gtn.linear_graph(T, C, gtn.Device(gtn.CPU), True)
    cpu_data = input[:, b, :].cpu().contiguous()
    g_emissions.set_weights(cpu_data.data_ptr())

    tgt_length = target_lengths[b]
    g_loss = gtn.criterion.ctc_loss(g_emissions, target[b, :tgt_length].tolist(), 0)

    gtn.backward(g_loss)

    # access the gradient ; moves data from CPU -> GPU
    grad_list = g_emissions.grad().weights_to_list()
    grad_tensor = torch.tensor(grad_list).cuda()


def gtn_ctc_func():
    gtn.parallel_for(process, range(B))


time_func(gtn_ctc_func, N, "GTN CTC ", iscuda=True)
