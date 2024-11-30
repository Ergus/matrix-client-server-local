use std::os::{unix::io::AsRawFd, fd::RawFd};

use crate::{Matrix, SharedBuffer};

/// A server class to construct from server instances.
///
/// This only stores the counter (to get tracks of the unique ids) and
/// the file descriptor for the socket connection
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
        let fd = nix::sys::socket::socket(
            nix::sys::socket::AddressFamily::Unix,
            nix::sys::socket::SockType::Stream,
            nix::sys::socket::SockFlag::empty(),
            None,
        ).unwrap();

        // Bind the socket to the file system path
        let sockaddr = nix::sys::socket::UnixAddr::new(Self::SOCKET_PATH).unwrap();
        nix::sys::socket::bind(fd, &sockaddr).unwrap();

        // Start listening for incoming connections
        nix::sys::socket::listen(fd, 10).unwrap();

        println!("Server listening on {}", Self::SOCKET_PATH);


        Self { counter: 0,  fd}
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
            shared_buffer.wait_response();
            println!("Received matrix: {} from client: {}", counter, shared_buffer.id());
            counter = counter + 1;

            let mut __guard_total = stats::TimeGuard::new("Total");

            let matrix = { // Read Matrix from shared memory
                let mut __guard = stats::TimeGuard::new("CopyIn");
                let matrix = Matrix::<f64>::from_buffer(shared_buffer.payload);

                // When we receive an empty matrix, then we break this thread and exit.
                if matrix.datalen() == 0 {
                    __guard_total.disable();
                    __guard.disable();
                    break;
                }

                matrix
            };


            let transpose: Matrix::<f64> = {
                let __guard = stats::TimeGuard::new(
                    format!("Transpose_{}X{}", matrix.rows(), matrix.cols()).as_str()
                );
                matrix.transpose()
            };

            { // Write the result back into shared memory
                let __guard = stats::TimeGuard::new("CopyOut");
                transpose.to_buffer(shared_buffer.payload);
                shared_buffer.notify();
            }
        }

        Ok(())
    }

    /// Function to wait for new connections. This is basically the
    /// place where the server `main' is most of the time
    ///
    ///This is not "elegant" to return a tuple, but it is simple enough
    pub fn wait_client(&mut self) -> nix::Result<(RawFd, u64, u64)>
    {
        let mut buf = [0u8; 8];

        let client_fd = nix::sys::socket::accept(self.fd)?;

        // Read the number from the client
        match nix::unistd::read(client_fd.as_raw_fd(), &mut buf) {
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
        if let Err(err) = nix::unistd::close(self.fd) {
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
    pub id: u64,
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
        nix::sys::socket::connect(fd, &sockaddr).unwrap();

        // Send a number
        let bytes = payload_size.to_be_bytes();
        nix::unistd::write(fd, &bytes).unwrap();

        let mut buf = [0u8; 8];

        let id = match nix::unistd::read(fd, &mut buf) {
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

/// A class to collect statistics
///
/// This can be tunned, but in general shows the main times at the end
/// of every connection.
/// The stats are stored as thread local information and the
/// information gets printed when the thread_local destructor is called
mod stats {

    use std::time::Instant;
    use std::collections::HashMap;
    use std::cell::RefCell;

    struct ThreadInfo {
        pub times_map: HashMap<String, Vec<u128>>,
    }

    impl Drop for ThreadInfo {
        fn drop(&mut self) {
           let summary = Summary::new(&self.times_map);

            summary.print(&["CopyIn", "Transpose", "CopyOut", "Total"].to_vec());

        }
    }

    // The RefCell cntains a hash map with the times information.
    thread_local! {
        static THREAD_INFO: RefCell<ThreadInfo>
        = RefCell::new(ThreadInfo {times_map: HashMap::new()});
    }

    /// Use a time guard to collect times easier with RAII.
    pub struct TimeGuard {
        enabled: bool,
        key: String,
        start: Instant,
    }

    impl TimeGuard {
        pub fn new(key: &str) -> Self {
            Self { enabled: true, key: key.to_string(), start: Instant::now() }
        }

        pub fn disable(&mut self)
        {
            self.enabled = false;
        }
    }

    impl Drop for TimeGuard {
        fn drop(&mut self) {
            if !self.enabled {
                return
            }

            let duration: u128 = self.start.elapsed().as_micros() ;

            THREAD_INFO.with(|thread_info| {

                thread_info.borrow_mut().times_map.entry(self.key.clone())
                    .and_modify(|existing| existing.push(duration))
                    .or_insert_with(|| vec![duration]);
            }
            );
        }
    }

    // This is for private use.
    struct StatsEntry {
        count: usize,
        avg: f64,
        min: u128,
        max: u128,
        sum: u128
    }

    impl StatsEntry {
        pub fn new(timesvec: &Vec<u128>) -> Self
        {
            let sum: u128 = timesvec.iter().sum(); // Sum of all elements
            let count = timesvec.len();    // Convert to f64 for division

            let avg: f64 = (sum as f64) / (count as f64);

            let max = *timesvec.iter().max().expect("Vector is empty"); // Find the max element
            let min = *timesvec.iter().min().expect("Vector is empty"); // Find the min element

            Self {count, avg, min, max, sum}

        }
    }

    /// Helper for print
    impl std::fmt::Display for StatsEntry {

        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {

            write!(f, "count: {:<8} avg: {:<10.1} min: {:<10} max: {:<10}",
                self.count, self.avg, self.min, self.max)?;
            Ok(())
        }
    }

    struct Summary {
        rows: usize,
        cols: usize,
        stats_map: HashMap<String, StatsEntry>
    }

    impl Summary {
        /// Parse the collected times and construct a Summary object.
        ///
        /// This function is called in the THREAD_INFO destructor.
        pub fn new(map: &HashMap<String, Vec<u128>>) -> Self
        {
            let mut stats_map: HashMap<String, StatsEntry> = HashMap::new();

            let mut rows: usize = 0;
            let mut cols: usize = 0;

            for (key, timesvec) in map.iter() {

                let stats = StatsEntry::new(timesvec);

                if key.starts_with("Transpose_") {
                    // This is a hacky workaround, but time...
                    stats_map.insert("Transpose".to_string(), stats);

                    // Parse Transpose_rowsXcols
                    let parts: Vec<&str> = key
                        .strip_prefix("Transpose_")
                        .unwrap()
                        .split('X')
                        .collect();

                    (rows, cols) = (parts[0].parse::<usize>().unwrap(), parts[1].parse::<usize>().unwrap())
                } else {
                    stats_map.insert(key.clone(), stats);
                }
            }

            Self{ rows, cols, stats_map }
        }

        pub fn print(&self, keys: &Vec<&str>)
        {
            println!("Printing Stats collected.  \nMatrix Dim: {}x{}", self.rows, self.cols);

            for key in keys.iter() {
                println!("{:24}\t {}", key, self.stats_map[&key.to_string()]);
            }

            let ops = (self.rows * self.cols) as f64;

            // 1.0E6 because time is in micro seconds
            let net = self.stats_map.get("Transpose").unwrap().avg;
            let gross = net + self.stats_map.get("CopyOut").unwrap().avg;
            let full = self.stats_map.get("Total").unwrap().avg;

            println!("Throughput (MFLOPS): net: {} gross: {} full: {}\n",
                ops / net, ops / gross, ops / full);
        }

    }

}


