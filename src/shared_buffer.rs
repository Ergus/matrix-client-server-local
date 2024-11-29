use nix::sys::mman::{mmap, munmap, shm_open, MapFlags, ProtFlags};
use nix::unistd::close;
use std::sync::atomic::{AtomicBool, Ordering};
use nix::unistd::ftruncate;
use nix::sys::stat::Mode;

use std::ptr;
use std::os::unix::io::RawFd;

use crate::Matrix;
use std::ffi::{c_void, CString};

#[derive(Debug, Clone)]
pub struct SharedBuffer<'a> {
    shm_name: String,
    shm_full_size: usize,
    is_client: bool,
    shm_fd: RawFd,
    ptr: *mut c_void,
    pub ready_flag: &'a AtomicBool,
    pub payload: *mut c_void,
}

impl SharedBuffer<'_> {

    /// Constructor
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

        Ok(Self { shm_name, shm_full_size, is_client, shm_fd, ptr, ready_flag, payload})
    }

    pub fn notify(&self)
    {
        self.ready_flag.store(self.is_client, Ordering::SeqCst);
    }

    // TODO: Error handling
    pub fn send(&self, matrix: &Matrix<f64>)
    {
        matrix.to_buffer(self.payload);
        self.notify();
    }

    pub fn wait_response(&self)
    {
        while self.ready_flag.load(Ordering::SeqCst) == self.is_client {
            std::thread::yield_now();
        }
    }

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

