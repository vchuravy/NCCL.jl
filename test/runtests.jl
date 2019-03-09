using NCCL
using Test
using CUDAdrv

@show id = NCCL.UniqueID()
@show comm = NCCL.Communicator(length(CUDAdrv.devices()), id, 0)
@show comm = NCCL.Communicator(CUDAdrv.devices())

