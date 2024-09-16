
import torch
import torch.nn as nn
from conformer import Conformer

import os
import sys
import time
import argparse
# from torch._inductor import config
# import torch._inductor
# torch._inductor.config.profiler_mark_wrapper_call = True
# torch._inductor.config.cpp.enable_kernel_profile = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=1, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=0, type=int, help='test warmup')
    parser.add_argument('--device', default='cpu', type=str, help='cpu, cuda or xpu')
    parser.add_argument("--compile", action='store_true', default=False,
                    help="enable torch.compile")
    parser.add_argument("--backend", type=str, default='inductor',
                    help="enable torch.compile backend")
    parser.add_argument("--triton_cpu", action='store_true', default=False,
                    help="enable triton_cpu")
    args = parser.parse_args()
    print(args)
    return args

def main(args):
    sequence_length, dim = 12345, 80
    if args.triton_cpu:
        import torch._inductor.config
        torch._inductor.config.cpu_backend="triton"
    import torch
    cuda = True if args.device == 'cuda' and torch.cuda.is_available() else False
    device = torch.device('cuda' if cuda else 'cpu')

    criterion = nn.CTCLoss().to(device)
    # inputs
    inputs = torch.rand(args.batch_size, sequence_length, dim).to(device)
    input_lengths = torch.LongTensor([12345] * args.batch_size)
    # targets
    targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2]] * args.batch_size).to(device)
    target_lengths = torch.LongTensor([9] * args.batch_size)

    model = Conformer(num_classes=10, 
                    input_dim=dim, 
                    encoder_dim=32, 
                    num_encoder_layers=3).to(device)
    if args.profile:
        if args.device == 'cpu':
            prof_act = [torch.profiler.ProfilerActivity.CPU]
        elif args.device == 'cuda':
            prof_act = [torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
        with torch.profiler.profile(
            activities=prof_act,
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int((args.num_iter)/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            args.p = p
            evaluate(args, model, inputs, input_lengths, criterion, targets, target_lengths)
    else:
        evaluate(args, model, inputs, input_lengths, criterion, targets, target_lengths)


def evaluate(args, model, inputs, input_lengths, criterion, targets, target_lengths):
    model.eval()
    print(inputs.shape, input_lengths.shape)
    if args.compile:
        model = torch.compile(model, backend=args.backend, options={"freezing": True})
    if args.channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
            criterion = criterion.to(memory_format=torch.channels_last)
            print("---- Use NHWC model")
            inputs = inputs.contiguous(memory_format=torch.channels_last)
            targets = targets.contiguous(memory_format=torch.channels_last)
            print("---- Use NHWC inputs")
        except Exception as e:
            print(e)
    if args.jit:
        try:
            model = torch.jit.trace(model, (inputs, input_lengths), check_trace=False, strict=False)
            print("---- JIT trace enable.")
            model = torch.jit.freeze(model)
        except Exception as e:
            print("---- JIT trace disable.")
            print("failed to use PyTorch jit mode due to: ", e)

    total_time = 0.0
    total_sample = 0
    for i in range(args.num_iter):
        # Forward propagate
        elapsed = time.time()
        outputs, output_lengths = model(inputs, input_lengths)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        elapsed = time.time() - elapsed
        if args.profile:
            args.p.step()
        # Calculate CTC Loss
        loss = criterion(outputs.transpose(0, 1), targets, output_lengths, target_lengths)
        print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
        
        if i >= args.num_warmup:
            total_time += elapsed
            total_sample += args.batch_size

    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference Latency: {:.3f} ms".format(latency))
    print("inference Throughput: {} samples/s".format(throughput))

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
            '-Conformer-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)


if __name__ == "__main__":

    args = parse_args()

    with torch.inference_mode(), torch.no_grad():
        if args.precision == "bfloat16" and args.device == "cpu":
            print("---- Use AMP autocast bfloat16 cpu")
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                main(args)
        elif args.precision == "float16" and args.device == "cpu":
            print("---- Use AMP autocast float16 cpu")
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.half):
                main(args)
        elif args.precision == "float16" and args.device == "cuda":
            print("---- Use AMP autocast float16 cuda")
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                main(args)
        elif args.precision == "float16" and args.device == "xpu":
            print("---- Use AMP autocast float16 xpu")
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
                main(args)
        else:
            main(args)
