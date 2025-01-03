use std::os::{fd::{FromRawFd, OwnedFd}, unix::io::AsRawFd};

use nix::sys::socket::Backlog;

use crate::{stats, bridge, Matrix, MatrixBorrow, SharedBuffer};

/// A server class to construct from server instances.
///
/// This only stores the counter (to get tracks of the unique ids) and
/// the file descriptor for the socket connection
pub struct Server {
    counter: i8,
    fd: OwnedFd,
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
        let fd: OwnedFd = nix::sys::socket::socket(
            nix::sys::socket::AddressFamily::Unix,
            nix::sys::socket::SockType::Stream,
            nix::sys::socket::SockFlag::empty(),
            None,
        ).unwrap();

        // Bind the socket to the file system path
        let sockaddr = nix::sys::socket::UnixAddr::new(Self::SOCKET_PATH).unwrap();
        nix::sys::socket::bind(fd.as_raw_fd(), &sockaddr).unwrap();

        // Start listening for incoming connections
        nix::sys::socket::listen(&fd, Backlog::new(10).unwrap()).unwrap();

        println!("Server listening on {}", Self::SOCKET_PATH);

        Self { counter: 0,  fd }
    }

    /// Main thread server function
    ///
    /// This function is the action spawned in the server side for
    /// every client.
    ///
    /// The function basically performs the
    /// check-read-traspose-write-notify in the shared buffer.
    /// See: [`SharedBuffer`] - Link to a type
    pub fn server_thread(shared_buffer: &mut SharedBuffer) -> nix::Result<()>
    {
        // Clear the ready flag
        println!("Server is waiting for client: {}...", shared_buffer.id());

        let mut counter: usize = 0;

        loop {
            // Poll the ready flag
            if !shared_buffer.wait_response() {
                println!("Client disconnected");
                break;
            }

            println!("Received matrix: {} from client: {}", counter, shared_buffer.id());
            counter += 1;

            let mut __guard_total = stats::TimeGuard::new("Total");

            let mut matrix = { // Read Matrix from shared memory
                let mut __guard = stats::TimeGuard::new("CopyIn");
                MatrixBorrow::<f64>::from_buffer(shared_buffer.payload())
            };

            let transpose: Option<Matrix::<f64>> = {
                let __guard = stats::TimeGuard::new(
                    format!("Transpose_{}X{}", matrix.rows(), matrix.cols()).as_str()
                );
                matrix.transpose()
            };

            // let _ctranspose = unsafe {
            //     let cmatrix = bridge::from_buffer(shared_buffer.payload() as *mut u8);

            //     let __guard = stats::TimeGuard::new("CTranspose");
            //     cmatrix.transpose()
            // };


            { // Write the result back into shared memory
                let __guard = stats::TimeGuard::new("CopyOut");
                match transpose {
                    Some(mtranspose) => { shared_buffer.send(&mtranspose) },
                    None => {}
                }
                if !shared_buffer.notify() {
                    println!("Matrix sent back, but client seems disconnected");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Function to wait for new connections. This is basically the
    /// place where the server `main' is most of the time
    ///
    ///This is not "elegant" to return a tuple, but it is simple enough
    pub fn wait_client(&mut self) -> nix::Result<(OwnedFd, i8, u64)>
    {
        let mut buf = [0u8; 8];

        let client_fd = nix::sys::socket::accept(self.fd.as_raw_fd()).unwrap();

        // Read the number from the client
        match nix::unistd::read(client_fd, &mut buf) {
            Ok(n) if n > 0 => {
                let payload_size = u64::from_be_bytes(buf);
                self.counter += 1;

                let owned_client_fd = unsafe { OwnedFd::from_raw_fd(client_fd) };
                Ok((owned_client_fd, self.counter, payload_size))
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
        if let Err(err) = nix::unistd::close(self.fd.as_raw_fd()) {
            eprintln!("Failed to close listening socket: {}", err);
        }

        // Remove the socket file
        if let Err(err) = std::fs::remove_file(Self::SOCKET_PATH) {
            eprintln!("Failed to remove socket file {}: {}", Self::SOCKET_PATH, err);
        }
    }
}

/// A client class to construct from client instances.
///
/// Only stores the id (receivef from the server) and the shared
/// buffer (constructed from it)
pub struct Client<'a> {
    pub id: i8,
    pub shared_buffer: SharedBuffer<'a>,
}

impl Client<'_> {

    /// Constructor
    ///
    /// Basically opens a socket connection and writes to the server asking for an id.
    /// The id number is used to construct a shared buffer.
    pub fn new(payload_size: u64) -> Self
    {
        // Create the socket
        let fd = nix::sys::socket::socket(
            nix::sys::socket::AddressFamily::Unix,
            nix::sys::socket::SockType::Stream,
            nix::sys::socket::SockFlag::empty(),
            None
        ).unwrap();

        // Create a Unix socket address
        let sockaddr = nix::sys::socket::UnixAddr::new(Server::SOCKET_PATH).unwrap();

        // Connect to the server
        nix::sys::socket::connect(fd.as_raw_fd(), &sockaddr).unwrap();

        // Send a number
        let bytes = payload_size.to_be_bytes();
        nix::unistd::write(&fd, &bytes).unwrap();

        let mut buf = [0u8; 1];

        // I send and receive 8 bytes because it was the same size we
        // sent and We can use the extra space for extra info in the
        // future.
        let id = match nix::unistd::read(fd.as_raw_fd(), &mut buf) {
            Ok(n) if n == 1 => Ok(i8::from_be_bytes(buf)),
            Ok(_) => Err(nix::errno::Errno::EIO), // Handle incomplete reads
            Err(err) => Err(err)
        }.unwrap();

        // Close the socket
        nix::unistd::close(fd.as_raw_fd()).unwrap();

        // TODO: Error handling here when receive a zero or negative value.
        // Which means that the server could not accept out connection.

        let shared_buffer = SharedBuffer::new(id, 0, payload_size as usize)
            .expect("Client couldn't create shared buffer");

        Self { id: id as i8, shared_buffer }
    }

}

