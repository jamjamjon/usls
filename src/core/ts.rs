use std::time::Duration;

#[derive(Debug, Default)]
pub struct Ts {
    n: usize,
    ts: Vec<Duration>,
}

impl Ts {
    pub fn total(&self) -> Duration {
        self.ts.iter().sum::<Duration>()
    }

    pub fn n(&self) -> usize {
        self.n / self.ts.len()
    }

    pub fn avg(&self) -> Duration {
        self.total() / self.n() as u32
    }

    pub fn avgi(&self, i: usize) -> Duration {
        if i >= self.ts.len() {
            panic!("Index out of bound");
        }
        self.ts[i] / self.n() as u32
    }

    pub fn ts(&self) -> &Vec<Duration> {
        &self.ts
    }

    // TODO: overhead?
    pub fn add_or_push(&mut self, i: usize, x: Duration) {
        match self.ts.get_mut(i) {
            Some(elem) => *elem += x,
            None => {
                if i >= self.ts.len() {
                    self.ts.push(x)
                }
            }
        }
        self.n += 1;
    }

    pub fn clear(&mut self) {
        self.n = Default::default();
        self.ts = Default::default();
    }
}
