#![allow(dead_code)]

use crate::stats::TimeGuard;
use cxx::CxxString;

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("irreducible/cpp/matrix.hpp");

        #[cxx_name = "Matrix"]
        type CMatrix;

        unsafe fn from_buffer(buffer: *mut u8) -> UniquePtr<CMatrix>;
        unsafe fn to_buffer(self: &CMatrix, buffer: *mut u8);

        fn transpose(self: &CMatrix) -> UniquePtr<CMatrix>;
    }

    extern "Rust" {
        type TimeGuard;

        fn new_time_guard(key: &CxxString) -> Box<TimeGuard>;
    }
}

fn new_time_guard(key: &CxxString) -> Box<TimeGuard> {
    return Box::new(TimeGuard::new(key.to_str().unwrap()));
}

pub use ffi::*;


