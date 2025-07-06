use std::collections::HashMap;
use std::time::Duration;

/// A macro to measure the execution time of a given code block and optionally log the result.
#[macro_export]
macro_rules! elapsed {
    ($code:expr) => {{
        let t = std::time::Instant::now();
        let ret = $code;
        let duration = t.elapsed();
        (duration, ret)
    }};
    ($label:expr, $ts:expr, $code:expr) => {{
        let t = std::time::Instant::now();
        let ret = $code;
        let duration = t.elapsed();
        $ts.push($label, duration);
        ret
    }};
}

/// Time series collection for performance measurement and profiling.
#[derive(aksr::Builder, Debug, Default, Clone, PartialEq)]
pub struct Ts {
    // { k1: [d1,d1,d1,..], k2: [d2,d2,d2,..], k3: [d3,d3,d3,..], ..}
    map: HashMap<String, Vec<Duration>>,
    names: Vec<String>,
}

impl std::ops::Index<&str> for Ts {
    type Output = Vec<Duration>;

    fn index(&self, index: &str) -> &Self::Output {
        self.map.get(index).unwrap_or_else(|| {
            let available_keys: Vec<&str> = self.map.keys().map(|s| s.as_str()).collect();
            panic!(
                "Key '{}' was not found in Ts. Available keys: {:?}",
                index, available_keys
            )
        })
    }
}

impl std::ops::Index<usize> for Ts {
    type Output = Vec<Duration>;

    fn index(&self, index: usize) -> &Self::Output {
        self.names
            .get(index)
            .and_then(|key| self.map.get(key))
            .unwrap_or_else(|| {
                panic!(
                    "Index {} was not found in Ts. Available indices: 0..{}",
                    index,
                    self.names.len()
                )
            })
    }
}

impl Ts {
    pub fn summary(&self) {
        let decimal_places = 4;
        let place_holder = '-';
        let width_count = 10;
        let width_time = 15;
        let width_task = self
            .names
            .iter()
            .map(|s| s.len())
            .max()
            .map(|x| x + 8)
            .unwrap_or(60);

        let sep = "-".repeat(width_task + 66);

        // cols
        println!(
            "\n\n{:<width_task$}{:<width_count$}{:<width_time$}{:<width_time$}{:<width_time$}{:<width_time$}",
            "Task", "Count", "Mean", "Min", "Max", "Total",
        );
        println!("{}", sep);

        if self.is_empty() {
            println!("No data available");
        } else {
            // rows
            let total_name = "(Total)".to_string();
            let mut iter = self
                .names
                .iter()
                .chain(std::iter::once(&total_name))
                .peekable();
            while let Some(task) = iter.next() {
                if iter.peek().is_none() {
                    let avg = self
                        .avg()
                        .map_or(place_holder.into(), |x| format!("{:.decimal_places$?}", x));
                    let total = format!("{:.decimal_places$?}", self.sum());
                    println!(
                        "{:<width_task$}{:<width_count$}{:<width_time$}{:<width_time$}{:<width_time$}{:<width_time$}",
                        task, place_holder, avg, place_holder, place_holder, total
                    );
                } else {
                    let durations = &self.map[task];
                    let count = durations.len();
                    let total = format!("{:.decimal_places$?}", self.sum_by_key(task));
                    let avg = self
                        .avg_by_key(task)
                        .map_or(place_holder.into(), |x| format!("{:.decimal_places$?}", x));
                    let min = durations
                        .iter()
                        .min()
                        .map_or(place_holder.into(), |x| format!("{:.decimal_places$?}", x));
                    let max = durations
                        .iter()
                        .max()
                        .map_or(place_holder.into(), |x| format!("{:.decimal_places$?}", x));

                    println!(
                        "{:<width_task$}{:<width_count$}{:<width_time$}{:<width_time$}{:<width_time$}{:<width_time$}",
                        task, count, avg, min, max, total
                    );
                }
            }
        }
    }

    pub fn merge(xs: &[&Ts]) -> Self {
        let mut names = Vec::new();
        let mut map: HashMap<String, Vec<Duration>> = HashMap::new();
        for x in xs.iter() {
            names.extend_from_slice(x.get_names());
            map.extend(x.get_map().to_owned());
        }

        Self { names, map }
    }

    pub fn push(&mut self, k: &str, v: Duration) {
        if !self.names.contains(&k.to_string()) {
            self.names.push(k.to_string());
        }
        self.map
            .entry(k.to_string())
            .and_modify(|x| x.push(v))
            .or_insert(vec![v]);
    }

    pub fn numit(&self) -> anyhow::Result<usize> {
        // num of iterations
        if self.names.is_empty() {
            anyhow::bail!("Empty Ts");
        }

        let len = self[0].len();
        for v in self.map.values() {
            if v.len() != len {
                anyhow::bail!(
                    "Invalid Ts: The number of elements in each values entry is inconsistent"
                );
            }
        }

        Ok(len)
    }

    pub fn is_valid(&self) -> bool {
        let mut iter = self.map.values();
        if let Some(first) = iter.next() {
            let len = first.len();
            iter.all(|v| v.len() == len)
        } else {
            true
        }
    }

    pub fn sum_by_index(&self, i: usize) -> Duration {
        self[i].iter().sum::<Duration>()
    }

    pub fn sum_by_key(&self, i: &str) -> Duration {
        self[i].iter().sum::<Duration>()
    }

    pub fn avg_by_index(&self, i: usize) -> anyhow::Result<Duration> {
        let len = self[i].len();
        if len == 0 {
            anyhow::bail!("Cannot compute average for an empty duration vector.")
        } else {
            Ok(self.sum_by_index(i) / len as u32)
        }
    }

    pub fn avg_by_key(&self, i: &str) -> anyhow::Result<Duration> {
        let len = self[i].len();
        if len == 0 {
            anyhow::bail!("Cannot compute average for an empty duration vector.")
        } else {
            Ok(self.sum_by_key(i) / len as u32)
        }
    }

    pub fn sum_column(&self, i: usize) -> Duration {
        self.map
            .values()
            .filter_map(|vec| vec.get(i))
            .copied()
            .sum()
    }

    pub fn sum(&self) -> Duration {
        self.map.values().flat_map(|vec| vec.iter()).copied().sum()
    }

    pub fn avg(&self) -> anyhow::Result<Duration> {
        self.names.iter().map(|x| self.avg_by_key(x)).sum()
    }

    pub fn skip(mut self, n: usize) -> Self {
        self.map.iter_mut().for_each(|(_, vec)| {
            *vec = vec.iter().skip(n).copied().collect();
        });
        self
    }

    pub fn clear(&mut self) {
        self.names.clear();
        self.map.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty() && self.map.is_empty()
    }

    /// Get reference to the names vector
    pub fn get_names(&self) -> &Vec<String> {
        &self.names
    }

    /// Get reference to the internal map
    pub fn get_map(&self) -> &HashMap<String, Vec<Duration>> {
        &self.map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_push_and_indexing() {
        let mut ts = Ts::default();

        ts.push("task1", Duration::new(1, 0));
        ts.push("task1", Duration::new(2, 0));
        ts.push("task2", Duration::new(3, 0));

        assert_eq!(ts["task1"], vec![Duration::new(1, 0), Duration::new(2, 0)]);
        assert_eq!(ts["task2"], vec![Duration::new(3, 0)]);
    }

    #[test]
    fn test_numit() {
        let mut ts = Ts::default();

        ts.push("task1", Duration::new(1, 0));
        ts.push("task1", Duration::new(2, 0));
        ts.push("task2", Duration::new(3, 0));
        ts.push("task2", Duration::new(4, 0));

        assert_eq!(ts.numit().unwrap(), 2);
    }

    #[test]
    fn test_is_valid() {
        let mut ts = Ts::default();

        ts.push("task1", Duration::new(1, 0));
        ts.push("task1", Duration::new(2, 0));
        ts.push("task2", Duration::new(3, 0));

        assert!(!ts.is_valid());

        ts.push("task2", Duration::new(4, 0));
        ts.push("task3", Duration::new(5, 0));

        assert!(!ts.is_valid());

        ts.push("task3", Duration::new(6, 0));
        assert!(ts.is_valid());
    }

    #[test]
    fn test_sum_by_index() {
        let mut ts = Ts::default();

        ts.push("task1", Duration::new(1, 0));
        ts.push("task1", Duration::new(2, 0));
        ts.push("task2", Duration::new(3, 0));
        ts.push("task2", Duration::new(3, 0));
        ts.push("task2", Duration::new(3, 0));

        assert_eq!(ts.sum_by_index(0), Duration::new(3, 0)); // 1 + 2
        assert_eq!(ts.sum_by_index(1), Duration::new(9, 0)); // 1 + 2
    }

    #[test]
    fn test_sum_by_key() {
        let mut ts = Ts::default();

        ts.push("task1", Duration::new(1, 0));
        ts.push("task1", Duration::new(2, 0));
        ts.push("task2", Duration::new(3, 0));
        ts.push("task2", Duration::new(3, 0));
        ts.push("task2", Duration::new(3, 0));

        assert_eq!(ts.sum_by_key("task1"), Duration::new(3, 0)); // 1 + 2
        assert_eq!(ts.sum_by_key("task2"), Duration::new(9, 0)); // 1 + 2
    }

    #[test]
    fn test_avg_by_index() {
        let mut ts = Ts::default();

        ts.push("task1", Duration::new(1, 0));
        ts.push("task1", Duration::new(2, 0));
        ts.push("task2", Duration::new(2, 0));
        ts.push("task2", Duration::new(2, 0));
        ts.push("task3", Duration::new(2, 0));

        assert_eq!(ts.avg_by_index(0).unwrap(), Duration::new(1, 500_000_000));
        assert_eq!(ts.avg_by_index(1).unwrap(), Duration::new(2, 0));
        assert_eq!(ts.avg_by_index(2).unwrap(), Duration::new(2, 0));
    }

    #[test]
    fn test_avg_by_key() {
        let mut ts = Ts::default();

        ts.push("task1", Duration::new(1, 0));
        ts.push("task1", Duration::new(2, 0));

        let avg = ts.avg_by_key("task1").unwrap();
        assert_eq!(avg, Duration::new(1, 500_000_000));
    }

    #[test]
    fn test_sum_column() {
        let mut ts = Ts::default();

        ts.push("task1", Duration::new(1, 0));
        ts.push("task1", Duration::new(2, 0));
        ts.push("task2", Duration::new(3, 0));

        assert_eq!(ts.sum_column(0), Duration::new(4, 0)); // 1 + 3
    }

    #[test]
    fn test_sum() {
        let mut ts = Ts::default();

        ts.push("task1", Duration::new(1, 0));
        ts.push("task1", Duration::new(2, 0));
        ts.push("task2", Duration::new(3, 0));

        assert_eq!(ts.sum(), Duration::new(6, 0));
    }

    #[test]
    fn test_avg() {
        let mut ts = Ts::default();

        ts.push("task1", Duration::new(1, 0));
        ts.push("task1", Duration::new(2, 0));
        ts.push("task2", Duration::new(3, 0));
        ts.push("task2", Duration::new(4, 0));

        assert_eq!(ts.avg().unwrap(), Duration::new(5, 0));
    }

    #[test]
    fn test_skip() {
        let mut ts = Ts::default();

        ts.push("task1", Duration::new(1, 0));
        ts.push("task1", Duration::new(2, 0));
        ts.push("task2", Duration::new(3, 0));
        ts.push("task2", Duration::new(4, 0));
        ts.push("task2", Duration::new(4, 0));

        let ts_skipped = ts.skip(1);

        assert_eq!(ts_skipped["task1"], vec![Duration::new(2, 0)]);
        assert_eq!(
            ts_skipped["task2"],
            vec![Duration::new(4, 0), Duration::new(4, 0)]
        );

        let ts_skipped = ts_skipped.skip(1);

        assert!(ts_skipped["task1"].is_empty());
        assert_eq!(ts_skipped["task2"], vec![Duration::new(4, 0)]);
    }

    #[test]
    fn test_clear() {
        let mut ts = Ts::default();

        ts.push("task1", Duration::new(1, 0));
        ts.push("task2", Duration::new(2, 0));

        ts.clear();
        assert!(ts.names.is_empty());
        assert!(ts.map.is_empty());
    }
}
