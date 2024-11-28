use nix::sys::mman::{mmap, munmap, shm_open, shm_unlink, MapFlags, ProtFlags};
use nix::sys::stat::Mode;
use nix::unistd::ftruncate;
use std::os::unix::io::RawFd;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::ffi::c_void;

use matrix::Matrix;

const SHM_NAME: &str = "/rust_ipc_shm";
const SHM_SIZE: usize = 4096; // Size of shared memory

fn main() -> nix::Result<()> {
    // Create shared memory
    let shm_fd: RawFd = shm_open(SHM_NAME, nix::fcntl::OFlag::O_CREAT | nix::fcntl::OFlag::O_RDWR, Mode::S_IRWXU)?;
    ftruncate(shm_fd, SHM_SIZE as i64)?;

    // Map shared memory into the process's address space
    let ptr: *mut u8 = unsafe {
        mmap(
            ptr::null_mut(),
            SHM_SIZE,
            ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
            MapFlags::MAP_SHARED,
            shm_fd,
            0,
        )?
    } as *mut u8;

    // Initialize shared memory
    let ready_flag = unsafe { &*(ptr as *mut AtomicBool) };
    let payload: *mut c_void = unsafe { ptr.add(8) as *mut c_void };

    // Clear the ready flag
    ready_flag.store(false, Ordering::SeqCst);

    println!("Server is waiting for client...");

    loop {
        // Poll the ready flag
        if ready_flag.load(Ordering::SeqCst) {

            let rows_a: &AtomicUsize = unsafe { &*(ptr.add(8) as *mut AtomicUsize) };
            let rows = rows_a.load(Ordering::SeqCst);

            // This is not save but useful for testing
            if rows == 0 {
                break;
            }

            let matrix = Matrix::<f64>::from_buffer(payload);

            print!("{}", matrix);

            let transpose: Matrix::<f64> = matrix.transpose(64);

            // Write the result back into shared memory
            transpose.to_buffer(payload);

            // Clear the ready flag to wait for the next client
            ready_flag.store(false, Ordering::SeqCst);
        } else {
            // Avoid busy-waiting
            std::thread::yield_now();
        }
    }

    // Clean up resources
    unsafe {
        munmap(ptr as *mut _, SHM_SIZE)?;
    }
    shm_unlink(SHM_NAME)?;
    // close(shm_fd).unwrap();

    Ok(())
}
