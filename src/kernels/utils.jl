function set_thread_block(len, nthreads=256)
    nblocks = ceil(Int, len / nthreads)
    return nthreads, nblocks
end
