# Communicator

const NCCL_UNIQUE_ID_BYTES = 128
const ncclUniqueId_t = NTuple{NCCL_UNIQUE_ID_BYTES, Cchar}

"""
    UniqueID()

Generates an Id to be used in `Communicator`. `UniqueID` should be
called once and the Id should be distributed to all ranks in the
communicator before calling `Communicator(nranks, uid, rank)`.
"""
struct UniqueID
    internal::ncclUniqueId_t

    function UniqueID()
        buf = zeros(Cchar, NCCL_UNIQUE_ID_BYTES)
        @apicall(:ncclGetUniqueId, (Ptr{Cchar},), buf)
        new(Tuple(buf))
    end
end

Base.convert(::Type{ncclUniqueId_t}, id::UniqueID) = id.internal


const ncclComm_t = Ptr{Cvoid}

struct Communicator
    handle::ncclComm_t
    Communicator(handle::ncclComm_t) = new(handle)
end

# creates a new communicator (multi thread/process version)
"""
   Communicator(nranks, uid, rank)

Creates a new Communicator (multi thread/process version)
`rank` must be between `0` and `nranks-1` and unique within a communicator
clique. Each rank is associated to a CUDA device which has to be set before
calling `Communicator`. Implicitly synchroniszed with other ranks so it must
be called by different threads/processes or used within `group`.
"""
function Communicator(nranks, comm_id, rank)
    handle_ref = Ref{ncclComm_t}()
    @apicall(:ncclCommInitRank, (Ptr{ncclComm_t}, Cint, ncclUniqueId_t, Cint), 
             handle_ref, nranks, comm_id, rank)

    Communicator(handle_ref[])
end 

Base.cconvert(::Type{ncclComm_t}, comm::Communicator) = comm.handle

# creates a clique of communicators (single process version)
function Communicator(devices)
    ndev = length(devices)
    comms = Vector{ncclComm_t}(undef, ndev)
    devlist = Cint[dev.handle for dev in devices]
    @apicall(:ncclCommInitAll, (Ptr{ncclComm_t}, Cint, Ptr{Cint}), comms, ndev, devlist)
    Communicator.(comms)
end

function destroy(comm::Communicator)
    @apicall(:ncclCommDestroy, (ncclComm_t,), comm)
end

function abort(comm::Communicator)
    @apicall(:ncclCommAbort, (ncclComm_t,), comm)
end

function getError(comm::Communicator)
    ref = Ref{ncclResult_t}()
    @apicall(:ncclCommGetAsyncError, (ncclComm_t, Ref{ncclResult_t}), comm, ref)
    return NCCLError(ref[])
end

function count(comm::Communicator)
    ref = Ref{Cint}()
    @apicall(:ncclCommCount, (ncclComm_t, Ref{Cint}), comm, ref)
    return ref[]
end

function rank(comm::Communicator)
    ref = Ref{Cint}()
    @apicall(:ncclUserRank, (ncclComm_t, Ref{Cint}), comm, ref)
    return ref[]
end

function device(comm::Communicator)
    ref = Ref{Cint}()
    @apicall(:ncclCommCuDevice, (ncclComm_t, Ref{Cint}), comm, ref)
    return CUDAdrv.CuDevice(ref[])
end

