
use std::env;
use rand::seq::SliceRandom;

use irreductible::{Matrix, Client};

fn parse_cl(args: &Vec<String>) -> (usize, usize, usize, usize)
{
    if args.len() != 5 {
        eprintln!("Usage: {} m n set_size n_requests", args[0]);
        std::process::exit(1);
    }

    // Parse the first four arguments as usize
    let parsed_args: Vec<usize> = args[1..]  // skip the first element (program name)
        .iter()
        .map(|x| x.parse::<usize>().unwrap()) // parse each argument
        .collect();

    // Destructure the parsed arguments
    let ret = (
        parsed_args[0],
        parsed_args[1],
        parsed_args[2],
        parsed_args[3],
    );

    // Validate 
    match ret {
        (m, n, set_size, n_requests) => {
            assert!(m >= 4 && m <= 14, "m is out of range");
            assert!(n >= 4 && n <= 14, "m is out of range");
            assert!(set_size > 0, "Set size cannot be zero");
            assert!(n_requests > 0, "The number of requests cannot be zero");
        }
    };

    ret
}

fn main() -> nix::Result<()>
{
    let args: Vec<String> = env::args().collect();

    let (m, n, set_size, n_requests) = parse_cl(&args);

    let rows = 2_usize.pow(m as u32);
    let cols = 2_usize.pow(n as u32);

    println!("Initializing matrices");
    let data: Vec<Matrix<f64>> = (0..set_size).map(|_| Matrix::<f64>::random(rows, cols)).collect();

    let payload_size: u64 = data.first().expect("The data array is empty?").payload_size() as u64;

    println!("Connecting to server");
    let mut client = Client::new(payload_size);

    let mut rng = rand::thread_rng();


    println!("Lets go!!");
    for _ in 0..n_requests {

        let tmp = data.choose(&mut rng);

        // println!("Sent:");
        // print!("{}", tmp.unwrap());

        client.shared_buffer.send(tmp.unwrap());

        client.shared_buffer.wait_response();
        let received = client.shared_buffer.receive();

        let transpose = tmp.expect("Error").transpose();

        println!("Received:");
        print!("{}", received);

        println!("transpose:");
        print!("{}", transpose);

        println!("\n");

        if received != transpose {
            let diff: Vec<_> = received.data().into_iter().zip(transpose.data()).map(|(a, b)| a - b).collect();

            for i in 0..rows {
                for j in 0..cols {
                    if diff[i * cols + j] != 0. {
                        println!("i={} j={} rec={} exp={} diff={}",
                            i, j, received[(i,j)], transpose[(i,j)], diff[i * cols + j]
                        );
                    }
                }
            }
        }

        debug_assert_eq!(received, transpose, "Received matrix is not the transpose");
    }

    println!("Inform the server that I am leaving!!");
    client.shared_buffer.send(&Matrix::<f64>::new(0, 0));

    Ok(())
}
