/// A class to collect statistics
///
/// This can be tunned, but in general shows the main times at the end
/// of every connection.
/// The stats are stored as thread local information and the
/// information gets printed when the thread_local destructor is called

use std::time::Instant;
use std::collections::HashMap;
use std::cell::RefCell;

struct ThreadInfo {
    pub times_map: HashMap<String, Vec<u128>>,
}

impl Drop for ThreadInfo {
    fn drop(&mut self) {
        let summary = Summary::new(&self.times_map);

        // This is hacky, but good enough for an assignment (it works)
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
    pub fn new(key: &str) -> Self
    {
        Self { enabled: true, key: key.to_string(), start: Instant::now() }
    }

    pub fn disable(&mut self)
    {
        self.enabled = false;
    }
}

impl Drop for TimeGuard {
    fn drop(&mut self)
    {
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
}

impl StatsEntry {
    pub fn new(timesvec: &Vec<u128>) -> Self
    {
        let sum: u128 = timesvec.iter().sum(); // Sum of all elements
        let count = timesvec.len();    // Convert to f64 for division

        let avg: f64 = (sum as f64) / (count as f64);

        let max = *timesvec.iter().max().expect("Vector is empty"); // Find the max element
        let min = *timesvec.iter().min().expect("Vector is empty"); // Find the min element

        Self {count, avg, min, max}

    }
}

/// Helper for print
impl std::fmt::Display for StatsEntry {

    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result
    {
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

        // I refer here to the 3 throughput relative to the user
        // measure method
        // net: refers to the transpse algorithm
        // gross: to the transpose + send back
        // full: Refers to all Copy_in + transpose + Copy_out
        // 1.0E6 because time is in micro seconds -> Mega
        let net = self.stats_map.get("Transpose").unwrap().avg;
        let gross = net + self.stats_map.get("CopyOut").unwrap().avg;
        let full = self.stats_map.get("Total").unwrap().avg;

        println!("Throughput (MFLOPS): net:{:.2} gross:{:.2} full:{:.2}\n",
            ops / net, ops / gross, ops / full);
    }

}


