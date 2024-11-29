
use irreductible::Server;

fn main() -> nix::Result<()> {

    let mut server = Server::new();


    server.run()
}
