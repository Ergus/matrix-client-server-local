use std::{env,mem};
use std::sync::{Arc, Mutex};

use rand::Rng; // Import the Rng trait

use irreducible::{Matrix, Client};

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

/// Initialize the matrix vector in parallel.
fn init_matrix_set(set_size: usize, rows: usize, cols: usize) -> Vec<Box<Matrix<f64>>>
{
    let num_threads: usize = std::thread::available_parallelism().unwrap().into();

    let result_vec = Arc::new(Mutex::new(Vec::<Box::<Matrix<f64>>>::with_capacity(set_size)));

    // Calculate how many items each thread should process
    let chunk_size = set_size / num_threads;
    let rest = set_size - chunk_size * num_threads;

    std::thread::scope(|s| {

        for i in 0..num_threads {
            let vec_clone = Arc::clone(&result_vec);

            let size = chunk_size + ((i < rest) as usize);

            if size == 0 {
                break;
            }

            s.spawn(move || {
                println!("Thread {} initializes {} matrices", i, size);
                // Initialize a portion of the Vec
                let mut local_vec = Vec::new();

                for _ in 0..size {
                    local_vec.push(Box::new(Matrix::<f64>::random(rows, cols)));
                }

                // Lock the mutex and append the local vector to the shared result vector
                let mut result = vec_clone.lock().unwrap();
                result.extend(local_vec);

                println!("Thread {} done.", i)
            });
        };
    });

    let mut guard = result_vec.lock().unwrap();
    mem::take(&mut *guard)
}

fn main() -> nix::Result<()>
{
    let args: Vec<String> = env::args().collect();

    let (m, n, set_size, n_requests) = parse_cl(&args);

    let rows = 2_usize.pow(m as u32);
    let cols = 2_usize.pow(n as u32);

    println!("Initializing matrices");
    let data = init_matrix_set(set_size, rows, cols);

    let transposes = data.iter().map(|box_ref| box_ref.transpose_parallel_static(64)).collect::<Vec<_>>();

    let payload_size: u64 = data.first().expect("The data array is empty?").payload_size() as u64;

    println!("Connecting to server");
    let mut client = Client::new(payload_size);

    if client.id == 0 {
        eprintln!("Could not establish connection, server returned 0");
        std::process::exit(1);
    }

    println!("Connection established with client id: {}", client.id);
    let mut rng = rand::thread_rng();


    println!("Lets go!!");
    for _ in 0..n_requests {

        let rand_i: usize = rng.gen_range(1..set_size);

        let tmp = &data[rand_i];
        let transpose = &transposes[rand_i];

        // println!("Sent:");
        // print!("{}", tmp.unwrap());

        client.shared_buffer.send(tmp);

        client.shared_buffer.wait_response();
        let received = client.shared_buffer.receive();

        println!("Received!");

        if received != *transpose {
            let difference = received.substract(&transpose);

            for i in 0..rows {
                for j in 0..cols {
                    let diff = difference.get(i, j);

                    if diff != 0. {

                        println!("i={} j={} rec={} exp={} diff={}",
                            i, j, received.get(i,j), transpose.get(i,j), diff
                        );
                    }
                }
            }
        }

        debug_assert_eq!(received, *transpose, "Received matrix is not the transpose");
    }

    println!("Inform the server that I am leaving!!");
    client.shared_buffer.send(&Matrix::<f64>::new(0, 0));

    Ok(())
}
