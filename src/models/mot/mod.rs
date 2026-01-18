//! Multi-object tracking (MOT) utilities.
//!
//! This implementation borrows heavily from the jamtrack-rs project by kadu-v:
//! https://github.com/kadu-v/jamtrack-rs
//!
//! This module provides tracking algorithms and utilities for tracking objects across video frames:
//! - **ByteTracker**: ByteTrack algorithm for multi-object tracking
//! - **KalmanFilter**: Kalman filter implementation for object state prediction
//! - **LAPJV**: Hungarian algorithm implementation for data association
//! - **STrack**: Single object track representation

mod bytetrack;
mod kalman_filter;
mod lapjv;
mod strack;

pub use bytetrack::*;
pub use kalman_filter::*;
pub use lapjv::*;
pub use strack::*;
