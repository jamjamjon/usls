//! Multi-object tracking (MOT) utilities.
//!
//! This module provides tracking algorithms and utilities for tracking objects across video frames:
//! - **ByteTracker**: ByteTrack algorithm for multi-object tracking
//! - **KalmanFilter**: Kalman filter implementation for object state prediction
//! - **LAPJV**: Hungarian algorithm implementation for data association
//! - **STrack**: Single object track representation

pub mod bytetrack;
pub mod kalman_filter;
pub mod lapjv;
pub mod strack;

pub use bytetrack::*;
pub use kalman_filter::*;
pub use lapjv::*;
pub use strack::*;
