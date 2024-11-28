use nix::sys::mman::{mmap, munmap, shm_open, MapFlags, ProtFlags};
use nix::unistd::close;
use std::sync::atomic::{AtomicBool, Ordering};

use std::{env, ptr};
use std::os::unix::io::RawFd;
use rand::seq::SliceRandom;

use matrix::Matrix;
use std::ffi::c_void;

const SHM_NAME: &str = "/rust_ipc_shm";
const SHM_SIZE: usize = 4096; // Size of shared memory

#[derive(Debug, Clone)]
pub struct Client<'a> {
    shm_fd: RawFd,
    ptr: *mut c_void,
    ready_flag: &'a AtomicBool,
    payload: *mut c_void,
}

impl Client<'_> {
    /// Constructor
    pub fn new() -> Self {
        let shm_fd: RawFd = shm_open(SHM_NAME, nix::fcntl::OFlag::O_RDWR, nix::sys::stat::Mode::empty()).unwrap();

        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                SHM_SIZE,
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                shm_fd,
                0,
            ).unwrap()
        } as *mut c_void;

        let ready_flag = unsafe { &*(ptr as *mut AtomicBool) };
        let payload: *mut c_void = unsafe { ptr.add(8)};

        Self { shm_fd, ptr, ready_flag, payload}
    }

    // TODO: Error handling
    pub fn send(&self, matrix: &Matrix<f64>)
    {
        matrix.to_buffer(self.payload);
    }

    pub fn wait_response(&self)
    {
        while self.ready_flag.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
    }

    pub fn read<T>(&mut self) -> Matrix<f64>
    {
        Matrix::<f64>::from_buffer(self.payload)
    }

}

impl Drop for Client<'_> {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr, SHM_SIZE).unwrap();
        }
        close(self.shm_fd).unwrap();
    }
}

// fn client_init()  {
//     // Open shared memory

//     // Access shared memory
//     let n_ptr = unsafe { &*(ptr.add(N_OFFSET) as *mut AtomicUsize) };
//     let array_ptr = unsafe { ptr.add(ARRAY_OFFSET) as *mut f64 };

//     // Prepare request
//     let data = vec![1.1, 2.2, 3.3, 4.4, 5.5];
//     let n = data.len();

//     // Write the number of elements and the array to shared memory
//     n_ptr.store(n, Ordering::SeqCst);

//     unsafe {
//         for (i, &value) in data.iter().enumerate() {
//             *array_ptr.add(i) = value;
//         }
//     }

//     // Signal the server
//     ready_flag.store(true, Ordering::SeqCst);

//     // Wait for the server to clear the flag
//     while ready_flag.load(Ordering::SeqCst) {
//         std::thread::yield_now();
//     }

//     // Read the response
//     let result = unsafe { *array_ptr };
//     println!("Client received sum: {}", result);

//     // Clean up resources
//     unsafe {
//         munmap(ptr, SHM_SIZE)?;
//     }
//     close(shm_fd)?;

// }


fn main() -> nix::Result<()> {

    let args: Vec<String> = env::args().collect();

    if args.len() != 5 {
        eprintln!("Usage: {} m n set_size n_requests", args[0]);
        std::process::exit(1);
    }

    // Parse the first four arguments as usize
    let parsed_args: Vec<u32> = args[1..]  // skip the first element (program name)
        .iter()
        .map(|x| x.parse::<u32>().unwrap()) // parse each argument
        .collect();

    // Destructure the parsed arguments
    let (m, n, set_size, n_requests) = (
        parsed_args[0],
        parsed_args[1],
        parsed_args[2],
        parsed_args[3],
    );

    assert!(m >= 4);
    assert!(m <= 14);
    assert!(n >= 4);
    assert!(n <= 14);

    let rows = 2_usize.pow(m);
    let cols = 2_usize.pow(n);

    // Initialize vector set
    let data: Vec<Matrix<f64>> = (0..set_size).map(|_| Matrix::<f64>::random(rows, cols)).collect();


    let mut rng = rand::thread_rng();


    let client = Client::new();


    for _ in 0..n_requests {

        let tmp = data.choose(&mut rng);

        client.send(tmp.unwrap());

    }





    Ok(())
}
