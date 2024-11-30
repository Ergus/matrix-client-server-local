# Readme

## Project parts

The project contains 4 main parts

1. Server-Client infrastructure (server_lib.rs)
2. Communication infrastructure (shared_buffer.rs)
3. Matrix class (matrix.rs)
4. Server and client programs (server.rs, client.rs)

There are not sub-projects or anything like that in order to make thing simpler.

## Execution

The execution is pretty simple.

1. Open a terminal and start the server: `cargo run -r --bin server`
2. Open another terminal and start a client: `cargo run -r --bin client 14 14 3 5`

Remember that the command line arguments are:

- m: rows = $2^m$

- n: rows = $2^n$

- set_size: number of random matrices generated in the client

- n_request: number of requests the client will send to the server.

You can start more than one client concurrently on different terminals.

## Implementation

### Matrix transpose

The Matrix class includes a trait to enforce that only numeric 64 bits
types can be used.

The class include 4 transposition algorithms that work out of place
generating a new transposed matrix.

1. **transpose_small_rectangle**: The dummy transposition algorithm
   iterating rows and columns. The Server uses it for small matrices.

2. **transpose_big**: Block based transposition algorithm sequentially.
   This version uses a temporal small squared matrix to perform
   inplace transposition. This version is at least ~3x faster than the
   previous one because reads and writes the memory in cache friendly
   order. But also because the unfriendly transpose operations are
   executed in small blocks that fit in cache lines.

3. **transpose_parallel_static**: Derived form previous version, but
   adds multi-threading. The total number of blocks/thread is
   precomputed assigned from the beginning.
   
   The code handles remainders blocks distributing it fairly.
   
   This algorithm may perform better in traditional hardware where all
   the cores have the same speed

4. **transpose_parallel_dynamic**: Performs the work distribution
   dynamically. The threads behave like ``workers'' that ask for work
   and when finish ask for more until there is not anything else to do.
   
   The work distribution uses a simple atomic counter that increments
   on every call.
   
   This algorithm may perform better in modern hardware with high
   performance and energy efficient cores.

The matrix class also include:

- Infrastructure to serialize-deserialize memory buffers into matrices
  efficiently. (with the unsafe ptr::copy_nonoverlapping)

- Infrastructure to extract and insert sub-blocks (with unsafe optimizations)

- A some heuristic to select correct block sizes

- A some heuristic to select correct algorithm based on matrix dimension.

The parallel function creates a maximum number of threads based on
system cores. While this is a common approach maybe a smarter
thread-pool could be implemented in order to avoid fork-join overhead
and over-subscription when the server has many clients.

### Server-Client

The server and client use an hybrid approach to communicate.

1. The server main function is reading (nix::unistd::read) from a unix
   socket waiting for connections. (The exercise explicitly said that
   server and client are in the same machine).
   
2. When a connection request is received the server creates a new
   thread and assigns a unique id to that client.
   
3. The new thread creates a shared memory buffer for that client based
   on the initial request information.
   
4. Then the server thread replies to the client with the id
   information.  If there was an error and the server cannot satisfy
   this client it replies 0.
   
5. The communication sync process is based on an atomic bool at the
   beginning of the shared memory region. The server sets it to zero
   in the creation moment.
   
   The client will check that value before writing a request. The
   request basically involves a write into the memory region and set
   the flag to 1 at the end.
   
   The server symmetrically checks for the flag to be true, when that
   happens it starts reading the data into a matrix with the
   Matrix::from_buffer function.
   
   If matrix dimension is zero it means that the client is done and
   the connection can be closed. In this case the communication buffer
   is destroyed a the thread finalizes.

The client uses multithread to initialize the matrices set in
parallel. This was not requires, but saves testing time.

### Shared buffer

The shared buffer is a shared memory wrapper that performs the io
operation described above and ensures the correct cleanup.

I chose shared memory because it is the most efficient IPC for large
data chunks and reduces the system calls and kernel latency. As the
buffers are created within the ``worker threads'' the whole
communication is lock free and there is not bottle neck when the
server has multiple clients connected. Every thread has a private
buffer and it is released when the client disconnects.

The only potential limitation is the amount of shared memory in the
server given by the OS.

## Optimizations

1. The Communication with shared memory reduces communication overhead
   and polling cost in server and client (only checks a flag).
   
2. The matrix transpose parallelization and heuristics.
   
   Using the dynamic dispatch and block dimension optimization. The
   current default blockdim is 64 because it is the most performing
   value in my machine (when running release). Another machine may
   perform better on different values.
   
3. The memory copies use ptr::copy_nonoverlapping which is optimized
   to read-write in cache order, but also to use vectorized
   instructions.
   
   The `copy_from_block` is also optimized with an unsafe call. As we
   are confident that the threads won't write in same memory regions,
   we perform the write without taking the lock. Otherwise the write
   operation becomes a bottle neck when the number of threads grow and
   all of them try to write back it's block after every block
   transpose.
   
## Known issues

1. The current code performs 3 memory copies. From shared memory to
   main memory, from main to transposed and from transposed to shared.
   
   One of these copies could be removed by using another matrix class
   where data is a slice instead of a vector.
   
   However, the trade of here is that access to shared memory is more
   expensive, so multi-thread access to it in non-contiguous accesses
   (by blocks) either reading or writing may not give a performance
   boost we expect.
   
2. The number of threads in the parallel transposition may produce
   over-subscription when there are many clients connected.
   
   The right approach involves using a thread pool and to deliver
   tasks.
   
   A workaround if this becomes an issue could be to take a lock in
   the `parallel_transpose_` functions just before
   `std::thread::scope`, in order to assert that only one
   ``computation'' is creating threads at the time, but not blocking
   other operations like copy from/to memory.
   
3. The stats are collected by thread-client not globally because I
   thing it is more useful in that way.
   
4. This doesn't have a robust error handling on the server or client
   to manage peer disconnections.

I spend about 2 days (Thursday: made it work, 1/2 Friday:
Optimization, 1/2 Saturday: Readme, testing and optimizing a bit more)
