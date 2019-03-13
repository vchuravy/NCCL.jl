const ncclDatatype_t = Cint
datatype(::Type{Int8})    = ncclDatatype_t(0)
datatype(::Type{UInt8})   = ncclDatatype_t(1)
datatype(::Type{Int32})   = ncclDatatype_t(2)
datatype(::Type{UInt32})  = ncclDatatype_t(3)
datatype(::Type{Int64})   = ncclDatatype_t(4)
datatype(::Type{UInt64})  = ncclDatatype_t(5)
datatype(::Type{Float16}) = ncclDatatype_t(6)
datatype(::Type{Float32}) = ncclDatatype_t(7)
datatype(::Type{Float64}) = ncclDatatype_t(8)
datatype(T::DataType) = error("NCCL doesn't support datatype $T")

const ncclRedOp_t = Cint
op(::Type{typeof(+)})   = ncclRedOp_t(0)
op(::Type{typeof(*)})   = ncclRedOp_t(1)
op(::Type{typeof(min)}) = ncclRedOp_t(2)
op(::Type{typeof(max)}) = ncclRedOp_t(3)
op(T::DataType) = error("NCCL doesn't support reduction operator $T")

function allReduce(::F, send::CuArray{T}, recv::CuArray{T}, comm::Communicator, stream=CUDAdrv.CuDefaultStream()) where {F, T}
    @assert size(send) == size(recv)
    @apicall(:ncclAllReduce, (CuPtr, CuPtr, Csize_t, ncclDatatype_t, ncclRedOp_t, ncclComm_t, CuStream_t),
                             send, rec, length(send), datatype(T), op(F), comm, stream)
end


