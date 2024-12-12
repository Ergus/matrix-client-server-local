use std::sync::{Arc, Mutex, Condvar};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use std::collections::VecDeque;

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
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Worker {
    fn new(
        id: usize,
        controler: Arc<ThreadPoolControlers>
    ) -> Self {
        println!("Creating Worker: {}", id);

        let thread = std::thread::spawn(move || {

            println!("Worker {} Running", id);
            'outter: loop {

                loop {
                    // The task needs to be taken independently of the
                    // match, because when inlined in the match it
                    // looks like the match holds the lock.
                    let task = {
                        let mut lock = controler.queue.lock().unwrap();

                        // I use a condition variable here to avoid
                        // wasting resources on active waiting
                        // (pooling)

                        // The thread will be here in the condition
                        // variable until someone notifies that there
                        // is some work to do or sets active to false.
                        while lock.is_empty() && controler.active.load(Ordering::Relaxed) {
                            lock = controler.cv.wait(lock).unwrap();
                        };

                        if lock.is_empty() {
                            break 'outter
                        }

                        // Update the atomic, this trick is to avoid
                        // taking the lock on every task finalization
                        // to notify taskwait.
                        controler.running.fetch_add(1, Ordering::Relaxed);

                        // We can pop_front even in empty queues.
                        // When the queue is empty we are here because
                        // active was set to false and we want to
                        // exit.  However, We don't do that check here
                        // to not-hold the lock for longer.
                        lock.pop_front()
                    };

                    match task {
                        Some(task) => {
                            println!(" -> Worker {id} got task {}", task.id);
                            task.execute();

                            if controler.running.fetch_sub(1, Ordering::Relaxed) == 1 {
                                let _lock = controler.queue.lock().unwrap();
                                controler.cv.notify_all();
                            }
                        },
                        None => {break 'outter} // We are exiting
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
    running: AtomicUsize,
    active: AtomicBool,
    cv: Condvar,
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
                running: AtomicUsize::new(0),
                active: AtomicBool::new(true),
                cv: Condvar::new()
            }
        );

        let workers = (0..size)
            .map(|i| Worker::new(i, Arc::clone(&controler)))
            .collect();

        Self {workers, controler}
    }

    /// Execute a task by sending it to the thread pool.
    pub fn submit(&self, job: Box<dyn FnOnce() + Send>)
    {
        let mut guard = self.controler.queue.lock().unwrap();
        guard.push_back(Task::new(job));

        if guard.len() == 1 {
            // wake up threads that may be sleeping
            self.controler.cv.notify_all();
        }

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

        drop(scope);
    }

    pub fn taskwait(&self)
    {
        let mut guard = self.controler.queue.lock().unwrap();
        while guard.len() > 0 {
            guard = self.controler.cv.wait(guard).unwrap();
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {

        self.controler.active.store(false, Ordering::Relaxed);

        // This notify is to stop all inactive threads that may be
        // waiting in the condvar.
        // The wait condition include both conditions:
        // !queue_empty + active == true
        self.controler.cv.notify_all();

        // Now wait all pending tasks
        self.taskwait();

        self.workers
            .iter_mut()
            .for_each(
                |worker| worker.thread.take().unwrap().join().unwrap()
            );
    }
}

struct PoolScopeControls {
    num_pending_tasks: AtomicUsize,
    scope_mutex: Mutex<()>,
    cv: Condvar,
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
                    scope_mutex: Mutex::new(()),
                    cv: Condvar::new()
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
                let _guard = controls.scope_mutex.lock().unwrap();
                controls.cv.notify_one();
            }

        });

        self.thread_pool.submit(job);
    }
}

impl<'a> Drop for PoolScopeData<'a> {

    fn drop(&mut self) {

        let mut guard = self.controls.scope_mutex.lock().unwrap();

        while self.controls.num_pending_tasks.load(Ordering::SeqCst) > 0 {
            guard = self.controls.cv.wait(guard).unwrap();
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
                println!("\tTask {} is running.", i);
            });
        }
    }

    #[test]
    fn pool_taskwait_fast()
    {
        let pool = ThreadPool::new(4);

        for i in 0..8 {
            pool.execute(move || {
                println!("\tBefore {} is running.", i);
            });
        }

        pool.taskwait();
        println!("=== Taskwait!!!===");

        for i in 9..18 {
            pool.execute(move || {
                println!("\tAfter {} is running.", i);
            });
        }

    }

    #[test]
    fn pool_taskwait_timer()
    {
        let pool = ThreadPool::new(4);

        for i in 0..8 {
            pool.execute(move || {
                println!("\tBefore {} is running.", i);
                std::thread::sleep(Duration::from_millis(500));
            });
        }

        pool.taskwait();
        println!("=== Taskwait!!!===");

        for i in 9..18 {
            pool.execute(move || {
                println!("\tAfter {} is running.", i);
                std::thread::sleep(Duration::from_millis(500));
            });
        }

    }

    #[test]
    fn pool_scope_fast()
    {
        let pool = ThreadPool::new(4);

        pool.scope(move |s| {
            for i in 0..8 {
                s.spawn(move || {
                    println!("\tTask {} is running.", i);
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
                    println!("\tTask {} is running.", i);
                    std::thread::sleep(Duration::from_millis(500));
                });
            }
        });
    }

}

