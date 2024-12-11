use std::sync::{Arc, Mutex, Condvar};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use std::collections::VecDeque;

#[cfg(target_arch = "x86")]
use core::arch::x86::_mm_pause;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm_pause;

#[cfg(target_arch = "arm")]
use core::arch::arm::__yield;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::__yield;

// General spin_wait function
fn spin_wait() {
    unsafe {
        // Use platform-specific pause/yield instruction
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        _mm_pause();

        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
        __yield();
    }

    // Fall back to std::thread::yield_now() if architecture is unsupported
    #[cfg(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64"
    )))]
    {
        std::thread::yield_now();
    }
}

struct Task {
    id: usize,
    job: Box<dyn FnOnce() + Send>,
}

impl Task {
    // Static-like counter specific to UniqueStruct.
    fn counter() -> &'static AtomicUsize
    {
        static COUNTER: AtomicUsize = AtomicUsize::new(0); // Static counter within the `impl`.
        &COUNTER
    }

    // Constructor for creating a new instance with a unique ID.
    fn new(job: Box<dyn FnOnce() + Send>) -> Self
    {
        let id = Self::counter().fetch_add(1, Ordering::Relaxed); // Increment counter atomically.
        Self { id, job }
    }

    fn execute(self)
    {
        (self.job)();
    }
}


struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn new(
        id: usize,
        controler: Arc<ThreadPoolControlers>
    ) -> Self {
        println!("Creating Worker: {}", id);

        let thread = thread::spawn(move || {

            println!("Worker {} Running", id);
            while controler.active.load(Ordering::Relaxed) {

                loop {
                    // The task needs to be taken independently of the
                    // match, because when inlined in the match it
                    // looks like the match holds the lock.
                    let task = controler.queue.lock().unwrap().pop_front();

                    match task {
                        Some(task) => {
                            println!(" -> Worker {id} got task {}", task.id);
                            task.execute();
                        },
                        None => {
                            // We break here because the queue is empty,
                            // So we can check the active condition.
                            spin_wait();
                            break;
                        }
                    }
                }
            }
            println!("Worker {} Done", id);
        });

        Self {id, thread: Some(thread),}
    }
}

struct ThreadPoolControlers {
    queue: Mutex<VecDeque<Task>>,
    active: AtomicBool,
}

pub struct ThreadPool {
    workers: Vec<Worker>,
    controler: Arc<ThreadPoolControlers>,
}

impl ThreadPool {
    /// Create a new ThreadPool with a fixed number of threads.
    pub fn new(size: usize) -> Self
    {
        assert!(size > 0);

        let controler = Arc::new(
            ThreadPoolControlers {
                queue: Mutex::new(VecDeque::<Task>::new()),
                active: AtomicBool::new(true)}
        );

        let workers = (0..size)
            .map(|i| Worker::new(i, Arc::clone(&controler)))
            .collect();

        Self {workers, controler}
    }

    /// Execute a task by sending it to the thread pool.
    pub fn submit(&self, job: Box<dyn FnOnce() + Send>)
    {
        self.controler.queue.lock().unwrap().push_back(Task::new(job));
    }

    /// Execute a task by sending it to the thread pool.
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.submit(Box::new(f));
    }

    pub fn scope<'scope, F>(&'scope self, fout: F)
    where
      F: FnOnce(&PoolScopeData<'scope>),
    {
        // We put the `ScopeData` into an `Arc` so that other threads can finish their
        // `decrement_num_running_threads` even after this function returns.
        let scope = PoolScopeData::new(&self);

        fout(&scope);
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {

        self.controler.active.store(false, Ordering::Relaxed);

        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}

struct PoolScopeControls {
    num_pending_tasks: AtomicUsize,
    shutdown_signal: (Mutex<bool>, Condvar),
}

pub struct PoolScopeData<'scope> {
    thread_pool: &'scope ThreadPool,
    controls: Arc<PoolScopeControls>
}

impl<'scope> PoolScopeData<'scope> {

    pub fn new(pool: &'scope ThreadPool) -> Self {

        Self {
            thread_pool: &pool,
            controls: Arc::new(
                PoolScopeControls {
                    num_pending_tasks: AtomicUsize::new(0),
                    shutdown_signal: (Mutex::new(false), Condvar::new()),
                }
            ),
        }
    }


    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'scope + 'static,
    {
        self.controls.num_pending_tasks.fetch_add(1, Ordering::SeqCst);
        let controls = Arc::clone(&self.controls);

        let job = Box::new(move || {

            f();

            if controls.num_pending_tasks.fetch_sub(1, Ordering::SeqCst) == 1 {
                let mut shutdown_complete = controls.shutdown_signal.0.lock().unwrap();
                *shutdown_complete = true;
                controls.shutdown_signal.1.notify_one();
            }

        });

        self.thread_pool.submit(job);
    }
}

impl<'a> Drop for PoolScopeData<'a> {

    fn drop(&mut self) {

        let mut shutdown_complete = self.controls.shutdown_signal.0.lock().unwrap();
        while self.controls.num_pending_tasks.load(Ordering::SeqCst) > 0 {
            shutdown_complete = self.controls.shutdown_signal.1.wait(shutdown_complete).unwrap();
        }
    }

}


#[cfg(test)]
mod matrix_borrow {

    use super::*;
    use std::time::Duration;

    #[test]
    fn pool_test()
    {
        let pool = ThreadPool::new(4);

        for i in 0..8 {
            pool.execute(move || {
                println!("Task {} is running.", i);
            });
        }
    }

    #[test]
    fn pool_scope()
    {
        let pool = ThreadPool::new(4);

        pool.scope(move |s| {
            for i in 0..8 {
                s.spawn(move || {
                    println!("Task {} is running.", i);
                });
            }
        }
        );

    }


    #[test]
    fn pool_scope_timer()
    {
        let pool = ThreadPool::new(4);

        pool.scope(move |s| {
            for i in 0..16 {
                s.spawn(move || {
                    println!("Task {} is running.", i);
                    std::thread::sleep(Duration::from_millis(500));
                });
            }
        });

    }
}

