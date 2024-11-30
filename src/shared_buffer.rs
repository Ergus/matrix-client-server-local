use nix::sys::mman::{mmap, munmap, shm_open, MapFlags, ProtFlags};
use nix::unistd::close;
use std::sync::atomic::{AtomicBool, Ordering};
use nix::unistd::ftruncate;
use nix::sys::stat::Mode;

use std::ptr;
use std::os::unix::io::RawFd;

use crate::Matrix;
use std::ffi::{c_void, CString};

/// A class to perform matrix interchange between processes.
///
/// This buffer uses shared memory as the mos efficient IPC within a
/// node.  I use socket communication only for the initial connection.
/// After that initial "sync" the Server creates an instance of this
/// class and all the following communications between the two processes
/// use this shared memory region which is more than two times faster.
///
/// Using shared memory also avoids the bottle neck in the socket when
/// more than one client is connected.
/// 
/// The communicatio process is pretty simple. There is an atomic
/// memory region in the beginning of the buffer (ready_flag).
///
/// 0. When the server receives a connection request (in the socket).
///    It creates a new thread and that thread attempts to construct this buffer.
/// 1. The server's thread initializes the flag to FALSE after the reserve.
/// 2. Then informs the client (using the socket)
/// 3. The client with the id information creates a "mirror" buffer sharing the memory.
/// 4. The client sets the information in the payload and sets the flag to TRUE.
/// 5. When the server's thread finds that the flag is on true. It starts the
///    read process and writes back the information in the same place when done.
/// 6. The sets the flag to false again
#[derive(Debug, Clone)]
pub struct SharedBuffer<'a> {
    // shm_name: String,
    id: u64,
    shm_full_size: usize,
    is_client: bool,
    shm_fd: RawFd,
    ptr: *mut c_void,
    pub ready_flag: &'a AtomicBool,
    pub payload: *mut c_void,
}

impl SharedBuffer<'_> {

    /// Constructor
    ///
    /// The constructor reserves a shared memory region and assumes
    /// and initializes
    pub fn new(is_client: bool, id: u64, payload_size: usize) -> nix::Result<Self>
    {
        let shm_name: String = format!("/rust_ipc_shm_{}", id);
        let shm_full_size: usize = size_of::<AtomicBool>() + payload_size;

        let shm_fd: RawFd;

        let c_shm_name = CString::new(&*shm_name).expect("CString::new failed");

        if is_client {
            shm_fd = shm_open(c_shm_name.as_c_str(), nix::fcntl::OFlag::O_RDWR, nix::sys::stat::Mode::empty()).unwrap();
        } else {
            shm_fd = shm_open(c_shm_name.as_c_str(), nix::fcntl::OFlag::O_CREAT | nix::fcntl::OFlag::O_RDWR, Mode::S_IRWXU).unwrap();
            ftruncate(shm_fd, shm_full_size as i64).unwrap();
        }

        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                shm_full_size,
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                shm_fd,
                0,
            ).unwrap()
        } as *mut c_void;

        let ready_flag = unsafe { &*(ptr as *mut AtomicBool) };
        let payload: *mut c_void = unsafe { ptr.add(8)};

        ready_flag.store(false, Ordering::SeqCst);

        Ok(Self { id, shm_full_size, is_client, shm_fd, ptr, ready_flag, payload})
    }

    /// Get the buffer id (same as client id)
    pub fn id(&self) -> u64
    {
        self.id
    }

    /// Get the number of 64 bits entries in the payload
    pub fn n_elements(&self) -> usize
    {
        (self.shm_full_size - 8) / 8
    }

    /// Change the flag value to notify the peer we are done.
    pub fn notify(&self)
    {
        self.ready_flag.store(self.is_client, Ordering::SeqCst);
    }

    /// Effectively write the matrix to the shared payload
    pub fn send(&self, matrix: &Matrix<f64>)
    {
        matrix.to_buffer(self.payload);
        self.notify();
    }

    /// Pooling check the flag value for replies.
    pub fn wait_response(&self)
    {
        while self.ready_flag.load(Ordering::SeqCst) == self.is_client {
            std::thread::yield_now();
        }
    }

    /// Effectively read the matrix from the shared payload
    pub fn receive(&mut self) -> Matrix<f64>
    {
        Matrix::<f64>::from_buffer(self.payload)
    }

}

impl Drop for SharedBuffer<'_> {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr, self.shm_full_size).unwrap();
        }
        close(self.shm_fd).unwrap();
    }
}

