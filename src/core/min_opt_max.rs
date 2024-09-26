/// A value composed of Min-Opt-Max
#[derive(Clone)]
pub struct MinOptMax {
    pub min: isize,
    pub opt: isize,
    pub max: isize,
}

impl Default for MinOptMax {
    fn default() -> Self {
        Self {
            min: -1,
            opt: -1,
            max: -1,
        }
    }
}

impl std::fmt::Debug for MinOptMax {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("")
            .field("Min", &self.min)
            .field("Opt", &self.opt)
            .field("Max", &self.max)
            .finish()
    }
}

impl From<(isize, isize, isize)> for MinOptMax {
    fn from((min, opt, max): (isize, isize, isize)) -> Self {
        let min = min.min(opt);
        let max = max.max(opt);
        Self { min, opt, max }
    }
}

impl From<[isize; 3]> for MinOptMax {
    fn from([min, opt, max]: [isize; 3]) -> Self {
        let min = min.min(opt);
        let max = max.max(opt);
        Self { min, opt, max }
    }
}

impl From<isize> for MinOptMax {
    fn from(opt: isize) -> Self {
        Self::new(opt)
    }
}

impl MinOptMax {
    pub fn new(opt: isize) -> Self {
        Self {
            min: opt,
            opt,
            max: opt,
        }
    }

    pub fn update(&mut self, opt: isize) {
        self.opt = opt;
        if self.min > opt {
            self.min = opt;
        }
        if self.max < opt {
            self.max = opt;
        }
    }

    pub fn is_dyn(&self) -> bool {
        self.opt == -1 && self.max == -1 && self.min == -1
    }
}
