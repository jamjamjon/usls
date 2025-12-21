use crate::{Hbb, KalmanFilterXYAH, StateCov, StateMean};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global track ID counter for unique track identification
pub(crate) static GLOBAL_TRACK_ID: AtomicUsize = AtomicUsize::new(0);

/// Track state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrackState {
    New,
    Tracked,
    Lost,
    Removed,
}

/// Single object tracker implementing Kalman filter-based tracking
///
/// This struct represents a tracked object with state management, Kalman filtering,
/// and bounding box tracking capabilities.
#[derive(Debug, Clone)]
pub struct STrack {
    pub(crate) kalman_filter: KalmanFilterXYAH,
    pub(crate) mean: StateMean,
    pub(crate) covariance: StateCov,
    pub(crate) hbb: Hbb,
    pub(crate) state: TrackState,
    pub(crate) is_activated: bool,
    pub(crate) track_id: usize,
    pub(crate) frame_id: usize,
    pub(crate) start_frame_id: usize,
    pub(crate) tracklet_len: usize,
}

impl PartialEq for STrack {
    fn eq(&self, other: &Self) -> bool {
        self.track_id == other.track_id
    }
}

impl STrack {
    /// Creates a new track from a detection bounding box
    pub fn new(hbb: Hbb) -> Self {
        Self {
            kalman_filter: KalmanFilterXYAH::default(),
            mean: [0.0; 8],
            covariance: [[0.0; 8]; 8],
            hbb,
            state: TrackState::New,
            is_activated: false,
            track_id: 0,
            frame_id: 0,
            start_frame_id: 0,
            tracklet_len: 0,
        }
    }

    /// Returns the confidence score of the track
    #[inline(always)]
    pub fn score(&self) -> f32 {
        self.hbb.confidence().unwrap_or(1.0)
    }

    /// Computes bounding box from Kalman filter state (cx, cy, aspect_ratio, height)
    fn compute_hbb_from_mean(&self) -> Hbb {
        let (cx, cy, a, h) = (self.mean[0], self.mean[1], self.mean[2], self.mean[3]);
        let w = a * h;
        let x = (cx - w / 2.0).max(0.0);
        let y = (cy - h / 2.0).max(0.0);

        Hbb::from_xywh(x, y, w, h)
    }

    /// Returns current bounding box computed from Kalman filter state
    pub fn current_hbb(&self) -> Hbb {
        let mut hbb = self.compute_hbb_from_mean();

        // Preserve metadata from stored hbb
        if let Some(conf) = self.hbb.confidence() {
            hbb = hbb.with_confidence(conf);
        }
        if let Some(id) = self.hbb.id() {
            hbb = hbb.with_id(id);
        }
        if let Some(name) = self.hbb.name() {
            hbb = hbb.with_name(name);
        }
        if self.track_id > 0 {
            hbb = hbb.with_track_id(self.track_id);
        }

        hbb
    }

    /// Predicts the next state using Kalman filter
    pub fn predict(&mut self) {
        if self.state != TrackState::Tracked {
            self.mean[7] = 0.0; // Reset velocity component for non-tracked states
        }
        let (new_mean, new_covariance) = self.kalman_filter.predict(&self.mean, &self.covariance);
        self.mean = new_mean;
        self.covariance = new_covariance;
    }

    /// Updates track with new detection
    pub fn update(&mut self, new_track: &STrack, frame_id: usize) {
        // Use detection bbox as measurement in cxcyah format
        let (cx, cy, a, h) = new_track.hbb.cxcyah();
        let measurement = [cx, cy, a, h];
        let (new_mean, new_covariance) =
            self.kalman_filter
                .update(&self.mean, &self.covariance, &measurement);
        self.mean = new_mean;
        self.covariance = new_covariance;

        self.frame_id = frame_id;
        self.tracklet_len += 1;
        self.state = TrackState::Tracked;
        self.is_activated = true;

        // Update bbox from Kalman filter state and preserve metadata
        self.update_hbb_from_detection(new_track);
    }

    /// Updates bounding box from Kalman filter prediction state
    fn update_hbb_from_prediction(&mut self) {
        let mut hbb = self.compute_hbb_from_mean();

        // Preserve existing metadata
        if let Some(conf) = self.hbb.confidence() {
            hbb = hbb.with_confidence(conf);
        }
        if let Some(id) = self.hbb.id() {
            hbb = hbb.with_id(id);
        }
        if let Some(name) = self.hbb.name() {
            hbb = hbb.with_name(name);
        }
        if self.track_id > 0 {
            hbb = hbb.with_track_id(self.track_id);
        }

        self.hbb = hbb;
    }

    /// Updates bounding box from Kalman filter state and detection metadata
    fn update_hbb_from_detection(&mut self, detection: &STrack) {
        let mut hbb = self.compute_hbb_from_mean();

        // Update metadata from detection
        if let Some(conf) = detection.hbb.confidence() {
            hbb = hbb.with_confidence(conf);
        }
        if let Some(name) = detection.hbb.name() {
            hbb = hbb.with_name(name);
        }
        if let Some(id) = detection.hbb.id() {
            hbb = hbb.with_id(id);
        }
        if self.track_id > 0 {
            hbb = hbb.with_track_id(self.track_id);
        }

        self.hbb = hbb;
    }

    /// Activates a new track
    pub fn activate(&mut self, kalman_filter: KalmanFilterXYAH, frame_id: usize) {
        self.kalman_filter = kalman_filter;
        self.track_id = Self::next_id();

        // Initialize Kalman filter with detection bbox
        let (cx, cy, a, h) = self.hbb.cxcyah();
        let measurement = [cx, cy, a, h];
        let (new_mean, new_covariance) = self.kalman_filter.initiate(&measurement);
        self.mean = new_mean;
        self.covariance = new_covariance;

        self.tracklet_len = 0;
        self.state = TrackState::Tracked;
        self.is_activated = frame_id == 1;
        self.frame_id = frame_id;
        self.start_frame_id = frame_id;

        // Update bbox and assign track ID
        self.update_hbb_from_prediction();
        self.hbb = self.hbb.clone().with_track_id(self.track_id);
    }

    /// Re-activates a lost track
    pub fn re_activate(&mut self, new_track: &STrack, frame_id: usize, new_id: bool) {
        // Use detection bbox as measurement
        let (cx, cy, a, h) = new_track.hbb.cxcyah();
        let measurement = [cx, cy, a, h];
        let (new_mean, new_covariance) =
            self.kalman_filter
                .update(&self.mean, &self.covariance, &measurement);
        self.mean = new_mean;
        self.covariance = new_covariance;

        self.tracklet_len = 0;
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.frame_id = frame_id;

        if new_id {
            self.track_id = Self::next_id();
        }

        // Update bbox from Kalman filter state and detection metadata
        self.update_hbb_from_detection(new_track);
    }

    /// Generates next unique track ID
    pub fn next_id() -> usize {
        GLOBAL_TRACK_ID.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Resets global track ID counter
    pub fn reset_id() {
        GLOBAL_TRACK_ID.store(0, Ordering::SeqCst);
    }

    /// Returns current track ID counter value
    pub fn get_current_count() -> usize {
        GLOBAL_TRACK_ID.load(Ordering::SeqCst)
    }

    /// Returns the most recent frame ID
    pub fn end_frame(&self) -> usize {
        self.frame_id
    }
}
