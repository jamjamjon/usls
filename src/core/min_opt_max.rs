use aksr::Builder;

/// A value composed of Min-Opt-Max
#[derive(Builder, Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub struct MinOptMax {
    #[args(except(setter))]
    min: usize,
    #[args(except(setter))]
    opt: usize,
    #[args(except(setter))]
    max: usize,
}

impl Default for MinOptMax {
    fn default() -> Self {
        Self {
            min: 1,
            opt: 1,
            max: 1,
        }
    }
}

impl MinOptMax {
    pub fn ones() -> Self {
        Default::default()
    }

    pub fn zeros() -> Self {
        Self {
            min: 0,
            opt: 0,
            max: 0,
        }
    }

    pub fn update_opt(&mut self, opt: usize) {
        // `opt` can be any valid usize number, even if it is smaller than `self.min` or greater than `self.max`.
        self.opt = opt;
        self.auto_tune();
    }

    pub fn try_update_min(&mut self, x: usize) -> anyhow::Result<()> {
        if x > self.opt {
            anyhow::bail!(
                "Newly assigned `min`: {} must be smaller than the current `self.opt`: {}",
                x,
                self.opt
            );
        } else {
            self.min = x;
            Ok(())
        }
    }

    pub fn try_update_max(&mut self, x: usize) -> anyhow::Result<()> {
        if x < self.opt {
            anyhow::bail!(
                "Newly assigned `max`: {} must be greater than the current `self.opt`: {}",
                x,
                self.opt
            );
        } else {
            self.max = x;
            Ok(())
        }
    }

    pub fn auto_tune(&mut self) {
        // Rule 1: min <= opt <= max
        // Rule 2: opt is unchangeable here
        self.min = self.min.min(self.opt);
        self.max = self.max.max(self.opt);
    }

    pub fn auto_tuned(mut self) -> Self {
        self.auto_tune();
        self
    }

    pub fn is_dyn(&self) -> bool {
        self.min + self.max + self.opt == 0
    }
}

// TODO: min = 1?????
impl From<i32> for MinOptMax {
    fn from(opt: i32) -> Self {
        let opt = opt.max(0) as usize;
        let min = opt;
        let max = opt;
        Self { min, opt, max }
    }
}

// TODO: min = 1?????
impl From<i64> for MinOptMax {
    fn from(opt: i64) -> Self {
        let opt = opt.max(0) as usize;
        let min = opt;
        let max = opt;
        Self { min, opt, max }
    }
}

impl From<u32> for MinOptMax {
    fn from(opt: u32) -> Self {
        let opt = opt as usize;
        let min = opt;
        let max = opt;
        Self { min, opt, max }
    }
}

impl From<u64> for MinOptMax {
    fn from(opt: u64) -> Self {
        let opt = opt as usize;
        let min = opt;
        let max = opt;
        Self { min, opt, max }
    }
}

impl From<usize> for MinOptMax {
    fn from(opt: usize) -> Self {
        let min = opt;
        let max = opt;
        Self { min, opt, max }
    }
}

// TODO: min = 1?????
impl From<isize> for MinOptMax {
    fn from(opt: isize) -> Self {
        let opt = opt.max(0) as usize;
        let min = opt;
        let max = opt;
        Self { min, opt, max }
    }
}

impl From<f32> for MinOptMax {
    fn from(opt: f32) -> Self {
        let opt = opt.max(0.).round() as usize;
        let min = opt;
        let max = opt;
        Self { min, opt, max }
    }
}

impl From<f64> for MinOptMax {
    fn from(opt: f64) -> Self {
        let opt = opt.max(0.).round() as usize;
        let min = opt;
        let max = opt;
        Self { min, opt, max }
    }
}

impl<T> From<(T, T, T)> for MinOptMax
where
    T: Into<MinOptMax>,
{
    fn from((min, opt, max): (T, T, T)) -> Self {
        let min = min.into().min;
        let opt = opt.into().opt;
        let max = max.into().max;
        Self { min, opt, max }.auto_tuned()
    }
}

impl<T> From<[T; 3]> for MinOptMax
where
    T: Into<MinOptMax>,
{
    fn from([min, opt, max]: [T; 3]) -> Self {
        Self::from((min, opt, max))
    }
}

#[cfg(test)]
mod tests_minoptmax {
    use super::MinOptMax;

    #[test]
    fn test_default() {
        let default_mom = MinOptMax::default();
        assert_eq!(default_mom.min, 1);
        assert_eq!(default_mom.opt, 1);
        assert_eq!(default_mom.max, 1);
    }

    #[test]
    fn test_ones() {
        let ones_mom = MinOptMax::ones();
        assert_eq!(ones_mom.min, 1);
        assert_eq!(ones_mom.opt, 1);
        assert_eq!(ones_mom.max, 1);
    }

    #[test]
    fn test_zeros() {
        let zeros_mom = MinOptMax::zeros();
        assert_eq!(zeros_mom.min, 0);
        assert_eq!(zeros_mom.opt, 0);
        assert_eq!(zeros_mom.max, 0);
    }

    #[test]
    fn test_update_opt() {
        let mut mom = MinOptMax::default();
        mom.update_opt(5);
        assert_eq!(mom.opt, 5);
        assert_eq!(mom.min, 1);
        assert_eq!(mom.max, 5);

        let mut mom = MinOptMax::from((5, 6, 7));
        mom.update_opt(2);
        assert_eq!(mom.opt, 2);
        assert_eq!(mom.min, 2);
        assert_eq!(mom.max, 7);
    }

    #[test]
    fn test_try_update_min_success() {
        let mut mom = MinOptMax::default();
        let result = mom.try_update_min(0);
        assert!(result.is_ok());
        assert_eq!(mom.min, 0);

        let result = mom.try_update_min(1);
        assert!(result.is_ok());
        assert_eq!(mom.min, 1);
    }

    #[test]
    fn test_try_update_min_failure() {
        let mut mom = MinOptMax::default(); // 1
        let result = mom.try_update_min(6);
        assert!(result.is_err());
        assert_eq!(mom.min, 1);
        assert_eq!(mom.opt, 1);
        assert_eq!(mom.max, 1);
    }

    #[test]
    fn test_try_update_max_success() {
        let mut mom = MinOptMax::default();
        let result = mom.try_update_max(20);
        assert!(result.is_ok());
        assert_eq!(mom.max, 20);
        assert_eq!(mom.opt, 1);
        assert_eq!(mom.min, 1);
    }

    #[test]
    fn test_try_update_max_failure() {
        let mut mom = MinOptMax::default();
        mom.update_opt(5);
        let result = mom.try_update_max(4);
        assert!(result.is_err());
        assert_eq!(mom.max, 5);
        assert_eq!(mom.opt, 5);
        assert_eq!(mom.min, 1);
    }

    #[test]
    fn test_combined_updates() {
        let mut mom = MinOptMax::default();
        mom.update_opt(5);
        assert_eq!(mom.max, 5);
        assert_eq!(mom.opt, 5);
        assert_eq!(mom.min, 1);

        assert!(mom.try_update_min(3).is_ok());
        assert_eq!(mom.min, 3);
        assert_eq!(mom.opt, 5);
        assert_eq!(mom.max, 5);

        assert!(mom.try_update_max(6).is_ok());
        assert_eq!(mom.max, 6);
        assert_eq!(mom.opt, 5);
        assert_eq!(mom.min, 3);

        // unchanged
        assert!(mom.try_update_min(7).is_err());
        assert_eq!(mom.max, 6);
        assert_eq!(mom.opt, 5);
        assert_eq!(mom.min, 3);

        // unchanged
        assert!(mom.try_update_max(4).is_err());
        assert_eq!(mom.max, 6);
        assert_eq!(mom.opt, 5);
        assert_eq!(mom.min, 3);
    }

    #[test]
    fn test_auto_tune() {
        let mut mom = MinOptMax {
            min: 5,
            opt: 3,
            max: 7,
        };
        mom.auto_tune();
        assert_eq!(mom.min, 3);
        assert_eq!(mom.max, 7);
        assert_eq!(mom.opt, 3);
    }

    #[test]
    fn test_auto_tuned() {
        let mom = MinOptMax {
            min: 5,
            opt: 3,
            max: 7,
        }
        .auto_tuned();
        assert_eq!(mom.min, 3);
        assert_eq!(mom.max, 7);
        assert_eq!(mom.opt, 3);
    }

    #[test]
    fn test_is_dyn() {
        let dyn_mom = MinOptMax::zeros();
        assert!(dyn_mom.is_dyn());

        let non_dyn_mom = MinOptMax {
            min: 1,
            opt: 1,
            max: 1,
        };
        assert!(!non_dyn_mom.is_dyn());
    }

    #[test]
    fn test_from_integer_types() {
        let from_i32: MinOptMax = MinOptMax::from(-5);
        assert_eq!(from_i32, MinOptMax::zeros());

        let from_i64: MinOptMax = MinOptMax::from(-3);
        assert_eq!(from_i64, MinOptMax::zeros());

        let from_u32: MinOptMax = MinOptMax::from(4u32);
        assert_eq!(
            from_u32,
            MinOptMax {
                min: 4,
                opt: 4,
                max: 4
            }
        );

        let from_u64: MinOptMax = MinOptMax::from(7u64);
        assert_eq!(
            from_u64,
            MinOptMax {
                min: 7,
                opt: 7,
                max: 7
            }
        );

        let from_usize: MinOptMax = MinOptMax::from(10);
        assert_eq!(
            from_usize,
            MinOptMax {
                min: 10,
                opt: 10,
                max: 10
            }
        );

        let from_isize: MinOptMax = MinOptMax::from(-1isize);
        assert_eq!(from_isize, MinOptMax::zeros());

        let from_f32: MinOptMax = MinOptMax::from(-2.0);
        assert_eq!(from_f32, MinOptMax::zeros());

        let from_f64: MinOptMax = MinOptMax::from(3.9);
        assert_eq!(
            from_f64,
            MinOptMax {
                min: 4,
                opt: 4,
                max: 4
            }
        );
    }

    #[test]
    fn test_from_tuple() {
        let tuple_mom: MinOptMax = MinOptMax::from((1, 2, 3));
        assert_eq!(
            tuple_mom,
            MinOptMax {
                min: 1,
                opt: 2,
                max: 3
            }
        );
    }

    #[test]
    fn test_from_array() {
        let array_mom: MinOptMax = [1, 2, 3].into();
        assert_eq!(
            array_mom,
            MinOptMax {
                min: 1,
                opt: 2,
                max: 3
            }
        );
    }
}
