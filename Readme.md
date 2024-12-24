# Readme

This was a 4 days code challenge I have extended for fun and to test
some rust features and patterns.

## Requirements:

1. Server client architecture
2. Running on the same linux host
3. No external crates except `rand` and `nix`
4. Optimize for throughput including parallelism.
5. Matrices dimensions are powers of two up to 16.

The code includes different versions in **different branches**.  The
**cpp** branch has a matrix api version implemented in C++ (with some C
tricks to beat rust performance)

## Project parts

The project contains 5 main parts

1. Server-Client infrastructure ([server_lib](src/server_lib.rs))
2. Communication infrastructure ([shared_buffer](src/shared_buffer.rs))
3. Matrix class ([matrix](src/matrix.rs))
4. Thread pool to avoid over-suscriptions and fork-join ([thread_pool](src/thread\_pool.rs))
5. Server and client programs ([server](src/server.rs), [client](src/client.rs))


Extra:

C++ matrix class ([matrix.h](cpp/matrix.hpp)) (Requires extra
dependency to connect with rust, so that code is disabled)

There are not sub-projects in order to make thing simpler.

## Execution

The execution is pretty simple.

1. Open a terminal and start the server: `cargo run -r --bin server`
2. Open another terminal and start a client: `cargo run -r --bin client 14 14 3 5`

The command line arguments are:

- m: rows = $2^m$

- n: rows = $2^n$

- set_size: number of random matrices generated in the client

- n_request: number of requests the client will send to the server.

You can start more than one client concurrently on different terminals.

## Implementation

### Matrix transpose

The Matrix class includes a trait to enforce that only numeric 64 bits
types can be used.

The class include 5 transposition algorithms that work generating a
transposed matrix with different methods.

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
   
5. **transpose_parallel_square_inplace**: Perform a dynamic transpose
   by block couples inplace for big squared matrices with reduced
   auxiliar space. This out performs by 2x the other parallel
   algorithms, but only applies to squared matrices.

The matrix class also includes:

- Infrastructure to serialize-deserialize memory buffers into matrices
  efficiently. (with the unsafe ptr::copy_nonoverlapping)
  
- Infrastructure to extract and insert sub-blocks in parallel (with
  unsafe optimizations to bypass some Rust constrains)
  
- Some heuristic to select correct block sizes (to fit in the L1 cache)
  
- Some heuristic to select correct algorithm based on matrix
  dimension.

The parallel functions implement a fork-join approach or use the made
in home thread-pool (as I cannot use external dependencies, but
functionally equivalent to the ThreadPool crate).

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

The shared buffer is a shared memory wrapper that performs the IO
operation described above and ensures the correct cleanup.

I chose shared memory because it is the most efficient IPC for large
data chunks and reduces the system calls and kernel latency. As the
buffers are created within the ``worker threads'' the whole
communication is **lock free** and there is not bottle neck when the
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
   
4. The matrix internally can store the data in a Vector or just use a
   Slice to access buffered data. This is a new optimization to avoid
   one of the copies from shared memory to local.
   
5. The `to_buffer` method includes an heuristic to parallelize IO for
   big matrices. There is a new function `to_buffer_parallel` that
   Implements a fixed data IO operation in parallel.
   
6. The client pre-computes also the transposes in order to stress more
   the server, but keep checking correctness.

Last code times:

```
Matrix Dim: 16384x16384
CopyOut                 	 count: 10       avg: 0.0        min: 0          max: 0          sum: 0
Transpose               	 count: 10       avg: 113895.3   min: 102280     max: 216675     sum: 1138953
CopyIn                  	 count: 10       avg: 0.0        min: 0          max: 0          sum: 0
Total                   	 count: 10       avg: 113902.2   min: 102287     max: 216685     sum: 1139022
Throughput (MFLOPS): net:2356.86 gross:2356.86 full:2356.72
```
