use std::ptr;
use std::sync::atomic::{AtomicI8, Ordering};
use std::os::unix::io::RawFd;
use std::ffi::c_void;

use nix::sys::mman::{mmap, munmap, shm_open, shm_unlink, MapFlags, ProtFlags};
use nix::sys::stat::Mode;
use nix::unistd::{close,ftruncate};

use crate::matrix;

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
    shm_name: String,
    id: i8,
    rid: i8,
    shm_full_size: usize,
    shm_fd: RawFd,
    ptr: *mut c_void,
    ready_flag: &'a AtomicI8,
    payload: *mut c_void,
}

impl SharedBuffer<'_> {

    /// Constructor
    ///
    /// The constructor reserves a shared memory region and assumes
    /// and initializes
    pub fn new(id: i8, rid: i8, payload_size: usize) -> nix::Result<Self>
    {
        let shm_name: String = format!("/rust_ipc_shm_{}", std::cmp::max(id, rid));
        let shm_full_size: usize = align_of::<u128>()  + payload_size;

        let shm_fd: RawFd;

        if id == 0 {
            shm_fd = shm_open(shm_name.as_str(), nix::fcntl::OFlag::O_CREAT | nix::fcntl::OFlag::O_RDWR, Mode::S_IRWXU)
                .expect("Server failed to create shared memory");
            ftruncate(shm_fd, shm_full_size as i64).unwrap();
        } else {
            shm_fd = shm_open(shm_name.as_str(), nix::fcntl::OFlag::O_RDWR, nix::sys::stat::Mode::empty())
                .expect("Client failed to open shared memory");
        }

        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                shm_full_size,
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                shm_fd,
                0,
            ).expect("Failed to map shared memory")
        } as *mut c_void;

        let ready_flag: &AtomicI8 = unsafe { &*(ptr as *mut AtomicI8) };
        let payload: *mut c_void = unsafe { ptr.byte_add(align_of::<u128>()) };

        // Server initializes the flag on creation.
        if id == 0 {
            ready_flag.store(0, Ordering::SeqCst);
        }

        Ok(Self { shm_name, id, rid, shm_full_size, shm_fd, ptr, ready_flag, payload})
    }

    /// Get the buffer id (same as client id)
    pub fn id(&self) -> i8
    {
        self.rid
    }

    /// Get the payload start address
    pub fn payload(&self) -> *mut c_void
    {
        self.payload
    }

    /// Change the flag value to notify the peer we are done.
    pub fn notify(&self) -> bool
    {
        let notified = self.ready_flag.compare_exchange(
            self.rid,         // Expected value
            self.id,          // New value
            Ordering::SeqCst, // Success memory ordering
            Ordering::SeqCst, // Failure memory ordering
        );

        match notified {
            Ok(_) => true,
            Err(_) => false
        }
    }

    /// Change the flag value to notify the peer we are done.
    pub fn notify_action(&self, action: i8)
    {
        assert!(action < 0, "Action notification value must be negative");
        self.ready_flag.store(action, Ordering::SeqCst);
    }

    /// Effectively write the matrix to the shared payload
    pub fn send<T, S>(&self, matrix: &matrix::MatrixTemp<T, S>)
      where
        T: matrix::Numeric64,
        S: matrix::SliceOrVec<T>,
        rand::distributions::Standard: rand::prelude::Distribution<T>,
    {
        matrix.to_buffer(self.payload);
    }

    /// Pooling check the flag value for replies.
    /// This function was extended to handle peers disconnections
    pub fn wait_response(&self) -> bool
    {
        loop {
            let ready: i8 = self.ready_flag.load(Ordering::SeqCst);

            if ready == self.id {
                // I wrote, so yield and sleep
                std::thread::yield_now();
            } else if ready == self.rid {
                // the remote wrote, to I have work to do
                return true;
            } else if ready < 0 {
                // The remote sets negative values on error, so, close
                // this connection
                return false;
            } else {
                // We should never be here, no other client is
                // intended to put its id here
                return false;
            }
        }
    }

    /// Effectively read the matrix from the shared payload
    pub fn receive(&mut self) -> matrix::MatrixBorrow::<f64>
    {
        matrix::MatrixBorrow::<f64>::from_buffer(self.payload)
    }

}

impl Drop for SharedBuffer<'_> {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr, self.shm_full_size).unwrap();
        }
        close(self.shm_fd).unwrap();

        if self.id == 0 {
            shm_unlink(self.shm_name.as_str()).expect("Failed to unlink shared memory");
        }
    }
}

