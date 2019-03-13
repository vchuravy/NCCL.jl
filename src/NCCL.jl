module NCCL

import CUDAdrv
import CuArrays
using Printf

import CUDAdrv: CuPtr, CuStream_t
import CuArrays: CuArray

const ext = joinpath(dirname(@__DIR__), "deps", "ext.jl")
isfile(ext) || error("NCCL.jl has not been built, please run Pkg.build(\"NCCL\").")
include(ext)
if !configured
    # default (non-functional) values for critical variables,
    # making it possible to _load_ the package at all times.
    const libnccl = nothing
end

include("base.jl")

function version()
    version = Ref{Cint}()
    @apicall(:ncclGetVersion, (Ref{Cint},), version)
    version[]
end

include("communicator.jl")
include("group.jl")
include("collectives.jl")

end
