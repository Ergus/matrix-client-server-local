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

            let matrix = Matrix::<f64>::from_buffer(shared_buffer.payload);

            // This is not save but useful for testing
            // When we receive an empty matrix, then we break this thread and exit.
            if matrix.data().len() == 0 {
                break;
            }

            println!("Received:");
            print!("{}", matrix);

            println!("\n");

            let transpose: Matrix::<f64> = matrix.transpose();

            // Write the result back into shared memory
            transpose.to_buffer(shared_buffer.payload);
            // Clear the ready flag to wait for the next client
            shared_buffer.notify();
        }

        Ok(())
    }


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


    // pub fn run(&mut self) -> nix::Result<()>
    // {
    //     let mut buf = [0u8; 8];

    //     loop {
    //         // Accept a new connection
    //         let client_fd = accept(self.fd)?;

    //         let client_fd = unsafe { std::fs::File::from_raw_fd(client_fd) };

    //         // Read the number from the client
    //         match read(client_fd.as_raw_fd(), &mut buf) {
    //             Ok(n) if n > 0 => {

    //                 let payload_size = u64::from_be_bytes(buf);
    //                 self.counter = self.counter + 1;
    //                 let id: u64 = self.counter;

    //                 std::thread::spawn(move || {

    //                     let mut shared_buffer = SharedBuffer::new(false, id, payload_size as usize);

    //                     let _ = write(client_fd.as_raw_fd(), &id.to_be_bytes());

    //                     let _ = Self::server_thread(&mut shared_buffer);

    //                     println!("Server exiting...");
    //                 });
    //             }
    //             Ok(_) => println!("Client disconnected"),
    //             Err(err) => eprintln!("Error reading from client: {}", err),
    //         };
    //     }
    // }
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
