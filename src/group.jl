groupStart() = @apicall(:ncclGroupStart, ())
groupEnd() = @apicall(:ncclGroupEnd, ())
function group(f)
    groupStart()
    f()
    groupEnd()
end
