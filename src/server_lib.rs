use nix::unistd::{read, write, close};
use nix::sys::socket::{
    accept, bind, listen, socket, AddressFamily, SockFlag, SockType, connect
};
use std::os::unix::io::{AsRawFd};
use std::os::fd::RawFd;

use crate::{Matrix, SharedBuffer};

pub struct Server {
    counter: u64,
    fd: RawFd,
}

impl Server {

    pub const SOCKET_PATH: &'static str = "/tmp/rust_unix_socket";

    pub fn new() -> Self
    {
        // Clean up any existing socket file
        if std::path::Path::new(Self::SOCKET_PATH).exists() {
            std::fs::remove_file(Self::SOCKET_PATH).unwrap();
        }

        // Create the Unix socket
        let fd = socket(
            AddressFamily::Unix,
            SockType::Stream,
            SockFlag::empty(),
            None,
        ).unwrap();

        // Bind the socket to the file system path
        let sockaddr = nix::sys::socket::UnixAddr::new(Self::SOCKET_PATH).unwrap();
        bind(fd, &sockaddr).unwrap();

        // Start listening for incoming connections
        listen(fd, 10).unwrap();

        println!("Server listening on {}", Self::SOCKET_PATH);


        Self { counter: 0,  fd}
    }

    pub fn server_thread(shared_buffer: &mut SharedBuffer) -> nix::Result<()>
    {
        // Clear the ready flag
        println!("Server is waiting for client...");

        loop {
            // Poll the ready flag
            shared_buffer.wait_response();

            let matrix = {
                let __guard = stats::TimeGuard::new("CopyIn");
                Matrix::<f64>::from_buffer(shared_buffer.payload)
            };

            // This is not save but useful for testing
            // When we receive an empty matrix, then we break this thread and exit.
            if matrix.data().len() == 0 {
                break;
            }

            // TODO: Add debug prints
            // println!("Received:");
            // print!("{}", matrix);

            let transpose: Matrix::<f64> = {
                let __guard = stats::TimeGuard::new("Transpose");
                matrix.transpose()
            };

            // Write the result back into shared memory
            {
                let __guard = stats::TimeGuard::new("CopyOut");
                transpose.to_buffer(shared_buffer.payload);
                shared_buffer.notify();
            }
        }

        stats::print_stats();

        Ok(())
    }

    /// This is not "elegant" to return a tuple, but it is simple enough
    pub fn wait_client(&mut self) -> nix::Result<(RawFd, u64, u64)>
    {
        let mut buf = [0u8; 8];

        let client_fd = accept(self.fd)?;

        // Read the number from the client
        match read(client_fd.as_raw_fd(), &mut buf) {
            Ok(n) if n > 0 => {
                let payload_size = u64::from_be_bytes(buf);
                self.counter = self.counter + 1;
                Ok((client_fd, self.counter, payload_size))
            }
            Ok(_) => {
                eprintln!("Client disconnected");
                Err(nix::errno::Errno::EPIPE)
            },
            Err(err) => {
                eprintln!("Error reading from client: {}", err);
                Err(err)
            },
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        // Close the listening socket
        if let Err(err) = close(self.fd) {
            eprintln!("Failed to close listening socket: {}", err);
        }

        // Remove the socket file
        if let Err(err) = std::fs::remove_file(Self::SOCKET_PATH) {
            eprintln!("Failed to remove socket file {}: {}", Self::SOCKET_PATH, err);
        }
    }
}


pub struct Client<'a> {
    id: u64,
    pub shared_buffer: SharedBuffer<'a>,
}

impl Client<'_> {

    pub fn new(payload_size: u64) -> Self
    {
        // Create the socket
        let fd = socket(AddressFamily::Unix, SockType::Stream, SockFlag::empty(), None).unwrap();

        // Create a Unix socket address
        let sockaddr = nix::sys::socket::UnixAddr::new(Server::SOCKET_PATH).unwrap();

        // Connect to the server
        connect(fd, &sockaddr).unwrap();

        // Send a number
        let bytes = payload_size.to_be_bytes();
        write(fd, &bytes).unwrap();

        let mut buf = [0u8; 8];

        let id = match read(fd, &mut buf) {
            Ok(n) if n == 8 => Ok(u64::from_be_bytes(buf)),
            Ok(_) => Err(nix::errno::Errno::EIO), // Handle incomplete reads
            Err(err) => Err(err)
        }.unwrap();

        // Close the socket
        nix::unistd::close(fd).unwrap();


        let shared_buffer = SharedBuffer::new(true, id, payload_size as usize)
            .expect("Client couldn't create shared buffer");

        Self { id, shared_buffer }
    }

}

mod stats {

    use std::time::Instant;
    use std::collections::HashMap;
    use std::cell::RefCell;

    thread_local! {
        static TIMESMAP: RefCell<HashMap<String, Vec<u128>>> = RefCell::new(HashMap::new());
    }

    /// Use a time guard to collect times easier
    pub struct TimeGuard {
        key: String,
        start: Instant,
    }

    impl TimeGuard {
        pub fn new(key: &str) -> Self {
            Self { key: key.to_string(), start: Instant::now() }
        }
    }

    impl Drop for TimeGuard {
        fn drop(&mut self) {
            let duration: u128 = self.start.elapsed().as_micros() ;

            TIMESMAP.with(|map| {
                let mut map = map.borrow_mut();
                map.entry(self.key.clone())
                    .and_modify(|existing| existing.push(duration))
                    .or_insert_with(|| vec![duration]);
            });
        }
    }


    pub fn print_stats()
    {
        println!("Client stats: (times in microseconds)");
        TIMESMAP.with(|map| {
            for (key, timesvec) in map.borrow_mut().iter() {
                let sum: u128 = timesvec.iter().sum(); // Sum of all elements
                let count = timesvec.len() as f64;    // Convert to f64 for division

                let avg: f64 = sum as f64 / count;

                let max = timesvec.iter().max().expect("Vector is empty"); // Find the max element
                let min = timesvec.iter().min().expect("Vector is empty"); // Find the min element

                println!("{:16}\t count:{:8} avg:{:8.1} min:{:8} max:{:8}", key, count, avg, min, max);
            }
        })
    }



}


