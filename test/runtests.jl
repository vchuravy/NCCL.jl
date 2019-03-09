using NCCL
using Test
using CUDAdrv
using CUDAnative

CUDAnative.initialize()

@show id = NCCL.UniqueID()
# initialise a communcator for nranks=1
# @show comm = NCCL.Communicator(1, id, 0)

@show comm = NCCL.Communicator(CUDAdrv.devices())

