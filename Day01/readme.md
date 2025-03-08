Objective: 
- To understand DataParallel and Distributed Data Parallel(DDP) Training
- Test with Kaggle T4 x2 ✅

- DataParallel (DP) replicates the model on each GPU and uses a single process to handle the training across multiple GPUs.
It splits the input batch, processes it on each GPU, and then averages the gradients at the end of the backward pass.
While this is effective for small-scale tasks, DDP is more scalable as it creates a separate process for each GPU, with each process having its own model replica.
- DDP performs gradient synchronization across processes using more efficient communication (via NCCL), which scales better, especially when training on many GPUs or across multiple nodes. 
DP, on the other hand, becomes less efficient with a larger number of GPUs because it involves a bottleneck in synchronizing gradients using a single process, making it harder to scale efficiently.


Reference:

1. https://github.com/dnddnjs/pytorch-multigpu/issues/6 
