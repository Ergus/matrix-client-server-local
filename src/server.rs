use nix::unistd::write;
use irreducible::{Server, SharedBuffer};

fn main() -> nix::Result<()> {

    let mut server = Server::new();

    loop {
        match server.wait_client() {
            Ok((client_fd, id, payload_size)) => {

                std::thread::spawn(move || {
                    match SharedBuffer::new(false, id, payload_size as usize) {
                        Ok(mut shared_buffer) => {
                            let _ = write(client_fd, &id.to_be_bytes());
                            let _ = Server::server_thread(&mut shared_buffer);
                            println!("Client: {} finished and disconnected...", shared_buffer.id());
                        }
                        Err(err) => {
                            let _ = write(client_fd, &0_u64.to_be_bytes());
                            eprintln!("Couldn't create shared buffer: {}", err);
                        }
                    }});
            },
            Err(err) => {
                eprintln!("Error reading client initial connection: {}", err);
            }
        }
    }
}
