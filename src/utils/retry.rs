/// A macro to retry an expression multiple times with configurable delays between attempts.
///
/// This macro supports three forms:
///
/// 1. `retry!(max_attempts, base_delay, max_delay, code)`  
///    - Customizes the retry behavior:
///      - `max_attempts`: Maximum number of retry attempts. Set to `0` for infinite retries.
///      - `base_delay`: Initial delay (in milliseconds) before retrying. Delays increase exponentially.
///      - `max_delay`: Maximum delay (in milliseconds) between retries.
///
/// 2. `retry!(max_attempts, code)`  
///    - Retries the provided `code` up to `max_attempts` times, with a default base delay of `80ms`
///      and a maximum delay of `1000ms` between attempts.
///
/// 3. `retry!(code)`  
///    - Retries the provided `code` indefinitely until it succeeds, using a default base delay of `80ms`
///      and a maximum delay of `1000ms` between attempts.
///
/// # Examples
///
/// ## Example 1: Retry with a default delay configuration
/// ```rust,ignore
/// use anyhow::Result;
/// use usls::retry;
///
/// fn main() -> Result<()> {
///     println!(
///         "{}",
///         retry!(3.9, {
///             Err::<usize, anyhow::Error>(anyhow::anyhow!("Failed message"))
///         })?
///     );
///     Ok(())
/// }
/// ```
///
/// ## Example 2: Retry until a random condition is met
/// ```rust,ignore
/// use anyhow::Result;
/// use usls::retry;
///
/// fn main() -> Result<()> {
///     let _n = retry!({
///         let n = rand::random::<f32>();
///         if n < 0.7 {
///             Err(anyhow::anyhow!(format!("Random failure: {}", n)))
///         } else {
///             Ok(1)
///         }
///     })?;
///     Ok(())
/// }
/// ```
///
/// ## Example 3: Retry with custom delays and a stateful condition
/// ```rust,ignore
/// use anyhow::Result;
/// use usls::retry;
///
/// fn main() -> Result<()> {
///     let mut cnt = 5;
///     fn example_function(cnt: usize) -> Result<usize> {
///         if cnt < 10 {
///             anyhow::bail!("Failed")
///         } else {
///             Ok(42)
///         }
///     }
///
///     println!(
///         "Result: {}",
///         retry!(20, 10, 100, {
///             cnt += 1;
///             example_function(cnt)
///         })?
///     );
///     Ok(())
/// }
/// ```
#[macro_export]
macro_rules! retry {
    ($code:expr) => {
        retry!(0, 80, 1000, $code)
    };
    ($max_attempts:expr, $code:expr) => {
        retry!($max_attempts, 80, 1000, $code)
    };
    ($max_attempts:expr, $base_delay:expr, $max_delay:expr, $code:expr) => {{
        let max_attempts: u64 = ($max_attempts as f64).round() as u64;
        let base_delay: u64 = ($base_delay as f64).round() as u64;
        let max_delay: u64 = ($max_delay as f64).round() as u64;
        if base_delay == 0 {
            anyhow::bail!(
                "[retry!] `base_delay` cannot be zero. Received: {}",
                $base_delay
            );
        }
        if max_delay == 0 {
            anyhow::bail!(
                "[retry!] `max_delay` cannot be zero. Received: {}",
                $max_delay
            );
        }
        if max_delay <= base_delay {
            anyhow::bail!(
                "[retry!] `max_delay`: {} must be greater than `base_delay`: {}.",
                $base_delay,
                $max_delay
            );
        }

        let mut n = 1;
        loop {
            match $code {
                Ok(result) => {
                    log::debug!("[retry!] Attempt {} succeeded.", n);
                    break Ok::<_, anyhow::Error>(result);
                }
                Err(err) => {
                    let message = format!(
                        "[retry!] Attempt {}/{} failed with error: {:?}.",
                        n,
                        if max_attempts == 0 {
                            "inf".to_string()
                        } else {
                            max_attempts.to_string()
                        },
                        err,
                    );
                    if max_attempts > 0 && n >= max_attempts {
                        log::error!("{} Stopping after {} attempts.", message, n);
                        anyhow::bail!(err);
                    }

                    let delay = (base_delay * (1 << (n - 1))).min(max_delay);
                    let delay = std::time::Duration::from_millis(delay);
                    log::debug!("{} Retrying in {:?}..", message, delay);
                    std::thread::sleep(delay);
                    n += 1;
                }
            }
        }
    }};
}
