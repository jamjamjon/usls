use anyhow::Result;
use std::{collections::HashMap, vec};

use crate::{lapjv, Hbb, KalmanFilterXYAH, STrack, TrackState};

/// Result of linear assignment algorithm for track-detection matching
///
/// This type alias represents the output of the Hungarian algorithm (linear assignment)
/// used to optimally match tracks with detections in multi-object tracking.
///
/// # Tuple Fields
///
/// The tuple contains three vectors in the following order:
///
/// 1. **`Vec<(usize, isize)>`** - **Matches**: Pairs of (track_index, detection_index)
///    - Each tuple represents a successful match between a track and a detection
///    - `usize` is the index of the track in the tracks vector
///    - `isize` is the index of the detection in the detections vector
///    - Example: `(0, 2)` means track[0] is matched with detection[2]
///
/// 2. **`Vec<usize>`** - **Unmatched Tracks**: Indices of tracks that couldn't be matched
///    - Contains track indices that have no corresponding detection
///    - These tracks will be marked as "lost" and may be removed after max_time_lost
///    - Example: `[1, 3]` means tracks[1] and tracks[3] are unmatched
///
/// 3. **`Vec<usize>`** - **Unmatched Detections**: Indices of detections that couldn't be matched
///    - Contains detection indices that have no corresponding track
///    - These detections will be used to create new tracks (if confidence is high enough)
///    - Example: `[0, 4]` means detections[0] and detections[4] are unmatched
///
/// # Algorithm Context
///
/// This result is produced by the Hungarian algorithm (implemented via LAPJV)
/// which finds the optimal assignment between tracks and detections to minimize
/// the total cost (typically IoU distance). The algorithm ensures:
///
/// - Each track can be matched with at most one detection
/// - Each detection can be matched with at most one track
/// - The total cost of all matches is minimized
/// - Unmatched tracks and detections are identified for further processing
type LinearAssignmentResult = (Vec<(usize, isize)>, Vec<usize>, Vec<usize>);

/// ByteTracker implementation for multi-object tracking
///
/// This tracker uses Kalman filtering and Hungarian algorithm for data association.
/// It follows the ByteTracker algorithm with two-stage association for robust tracking.
#[derive(Debug)]
pub struct ByteTracker {
    kalman_filter: KalmanFilterXYAH,
    tracked_stracks: Vec<STrack>,
    lost_stracks: Vec<STrack>,
    removed_stracks: Vec<STrack>,
    frame_id: usize,
    track_id_count: usize,
    track_high_thresh: f32,
    track_low_thresh: f32,
    new_track_thresh: f32,
    match_thresh: f32,
    second_match_thresh: f32,
    unconfirmed_match_thresh: f32,
    fuse_score: bool,
    max_age: usize,
}

impl Default for ByteTracker {
    fn default() -> Self {
        Self {
            kalman_filter: KalmanFilterXYAH::default(),
            tracked_stracks: Vec::new(),
            lost_stracks: Vec::new(),
            removed_stracks: Vec::new(),
            frame_id: 0,
            track_id_count: 0,
            track_high_thresh: 0.25,
            track_low_thresh: 0.1,
            new_track_thresh: 0.25,
            match_thresh: 0.8,
            second_match_thresh: 0.5,
            unconfirmed_match_thresh: 0.7,
            fuse_score: true,
            max_age: 30,
        }
    }
}

impl ByteTracker {
    /// Creates a new ByteTracker with specified parameters
    pub fn new(
        max_age: usize,
        track_high_thresh: f32,
        track_low_thresh: f32,
        new_track_thresh: f32,
        match_thresh: f32,
        fuse_score: bool,
    ) -> Self {
        Self {
            track_high_thresh,
            track_low_thresh,
            new_track_thresh,
            match_thresh,
            fuse_score,
            max_age,
            ..Default::default()
        }
    }

    /// Updates tracker with detections
    pub fn update(&mut self, hbbs: &[Hbb]) -> Result<Vec<Hbb>> {
        if hbbs.is_empty() {
            return Ok(Vec::new());
        }

        self.frame_id += 1;

        // Split detections by confidence thresholds
        let mut det_stracks = Vec::new();
        let mut det_low_stracks = Vec::new();

        for hbb in hbbs {
            let confidence = hbb.confidence().unwrap_or(1.0);
            let strack = STrack::new(hbb.clone());

            if confidence >= self.track_high_thresh {
                det_stracks.push(strack);
            } else if confidence > self.track_low_thresh {
                det_low_stracks.push(strack);
            }
        }

        let mut activated_stracks = Vec::new();
        let mut refind_stracks = Vec::new();
        let mut lost_stracks = Vec::new();
        let mut removed_stracks = Vec::new();

        // Split tracked_stracks into active and unconfirmed
        let mut unconfirmed = Vec::new();
        let mut tracked_stracks = Vec::new();
        for track in self.tracked_stracks.iter() {
            if !track.is_activated {
                unconfirmed.push(track.clone());
            } else {
                tracked_stracks.push(track.clone());
            }
        }

        // First association with high score detections
        let mut strack_pool = Self::joint_stracks(&tracked_stracks, &self.lost_stracks);
        self.multi_predict(&mut strack_pool);

        let dists = self.get_dists(&strack_pool, &det_stracks);
        let (matches, u_track, u_detection) =
            self.linear_assignment(&dists, self.match_thresh, det_stracks.len())?;

        for (itracked, idet) in matches {
            let mut track = strack_pool[itracked].clone();
            let det = &det_stracks[idet as usize];
            if track.state == TrackState::Tracked {
                track.update(det, self.frame_id);
                activated_stracks.push(track.clone());
            } else {
                track.re_activate(det, self.frame_id, false);
                refind_stracks.push(track.clone());
            }
        }

        // Second association with low score detections
        let detections_second = det_low_stracks;
        let r_tracked_stracks: Vec<STrack> = u_track
            .iter()
            .filter_map(|&i| {
                if strack_pool[i].state == TrackState::Tracked {
                    Some(strack_pool[i].clone())
                } else {
                    None
                }
            })
            .collect();

        let dists = Self::calc_iou_distance(&r_tracked_stracks, &detections_second);
        let (matches, u_track, _u_detection_second) =
            self.linear_assignment(&dists, self.second_match_thresh, detections_second.len())?;

        for (itracked, idet) in matches {
            let mut track = r_tracked_stracks[itracked].clone();
            let det = &detections_second[idet as usize];
            if track.state == TrackState::Tracked {
                track.update(det, self.frame_id);
                activated_stracks.push(track.clone());
            } else {
                track.re_activate(det, self.frame_id, false);
                refind_stracks.push(track.clone());
            }
        }

        for &it in u_track.iter() {
            let mut track = r_tracked_stracks[it].clone();
            if track.state != TrackState::Lost {
                track.state = TrackState::Lost;
                lost_stracks.push(track.clone());
            }
        }

        // Handle unconfirmed tracks
        let detections: Vec<STrack> = u_detection
            .iter()
            .map(|&i| det_stracks[i].clone())
            .collect();

        let dists = self.get_dists(&unconfirmed, &detections);
        let (matches, u_unconfirmed, u_detection) =
            self.linear_assignment(&dists, self.unconfirmed_match_thresh, detections.len())?;

        for (itracked, idet) in matches {
            unconfirmed[itracked].update(&detections[idet as usize], self.frame_id);
            activated_stracks.push(unconfirmed[itracked].clone());
        }

        for &it in u_unconfirmed.iter() {
            let mut track = unconfirmed[it].clone();
            track.state = TrackState::Removed;
            removed_stracks.push(track.clone());
        }

        // Initialize new tracks
        for &inew in u_detection.iter() {
            let mut track = detections[inew].clone();
            if track.score() < self.new_track_thresh {
                continue;
            }
            track.activate(self.kalman_filter.clone(), self.frame_id);
            activated_stracks.push(track.clone());
        }

        // Update state - remove old lost tracks
        for track in self.lost_stracks.iter() {
            if self.frame_id - track.end_frame() > self.max_age {
                let mut track = track.clone();
                track.state = TrackState::Removed;
                removed_stracks.push(track.clone());
            }
        }

        // Update tracked_stracks
        self.tracked_stracks = activated_stracks;
        self.tracked_stracks.extend(refind_stracks);

        // Update lost_stracks
        self.lost_stracks = Self::sub_stracks(&self.lost_stracks, &self.tracked_stracks);
        self.lost_stracks.extend(lost_stracks);
        self.lost_stracks = Self::sub_stracks(&self.lost_stracks, &removed_stracks);

        // Remove duplicate stracks
        let (tracked_stracks_out, lost_stracks_out) =
            self.remove_duplicate_stracks(&self.tracked_stracks, &self.lost_stracks);
        self.tracked_stracks = tracked_stracks_out;
        self.lost_stracks = lost_stracks_out;

        // Update removed_stracks
        self.removed_stracks.extend(removed_stracks);
        if self.removed_stracks.len() > 1000 {
            self.removed_stracks = self
                .removed_stracks
                .iter()
                .rev()
                .take(999)
                .cloned()
                .collect();
        }

        // Return activated tracks
        let output_hbbs: Vec<Hbb> = self
            .tracked_stracks
            .iter()
            .filter(|track| track.is_activated)
            .map(|track| track.hbb.clone())
            .collect();

        Ok(output_hbbs)
    }

    pub fn with_max_age(mut self, n: usize) -> Self {
        self.max_age = n;
        self
    }

    /// Resets the global track ID counter
    pub fn reset_id(&mut self) {
        STrack::reset_id();
        self.track_id_count = 0;
    }

    /// Resets the tracker to initial state
    pub fn reset(&mut self) {
        self.tracked_stracks.clear();
        self.lost_stracks.clear();
        self.removed_stracks.clear();
        self.frame_id = 0;
        self.kalman_filter = KalmanFilterXYAH::default();
        self.reset_id();
    }

    /// Calculates distances between tracks and detections
    pub fn get_dists(&self, tracks: &[STrack], detections: &[STrack]) -> Vec<Vec<f32>> {
        let mut dists = Self::calc_iou_distance(tracks, detections);
        if self.fuse_score {
            Self::fuse_score(&mut dists, detections);
        }
        dists
    }

    /// Predicts next states for multiple tracks
    pub fn multi_predict(&self, tracks: &mut [STrack]) {
        for track in tracks.iter_mut() {
            track.predict();
        }
    }

    /// Combines two track lists, removing duplicates by track_id
    pub(crate) fn joint_stracks(a_tracks: &[STrack], b_tracks: &[STrack]) -> Vec<STrack> {
        let mut exists = HashMap::new();
        let mut res = Vec::new();

        for a in a_tracks.iter() {
            exists.insert(a.track_id, 1);
            res.push(a.clone());
        }

        for b in b_tracks.iter() {
            let tid = b.track_id;
            if exists.insert(tid, 1).is_none() {
                res.push(b.clone());
            }
        }
        res
    }

    /// Removes tracks from first list that exist in second list
    pub(crate) fn sub_stracks(a_tracks: &[STrack], b_tracks: &[STrack]) -> Vec<STrack> {
        let mut stracks = HashMap::new();
        for a in a_tracks.iter() {
            stracks.insert(a.track_id, a.clone());
        }

        for b in b_tracks.iter() {
            let tid = b.track_id;
            if stracks.contains_key(&tid) {
                stracks.remove(&tid);
            }
        }

        stracks.values().cloned().collect::<Vec<_>>()
    }

    /// Removes duplicate tracks based on IoU overlap
    pub(crate) fn remove_duplicate_stracks(
        &self,
        a_stracks: &[STrack],
        b_stracks: &[STrack],
    ) -> (Vec<STrack>, Vec<STrack>) {
        let mut a_res = Vec::new();
        let mut b_res = Vec::new();

        let ious = Self::calc_iou_distance(a_stracks, b_stracks);
        let mut overlapping_combinations = Vec::new();

        for (i, row) in ious.iter().enumerate() {
            for (j, &iou) in row.iter().enumerate() {
                if iou < 0.15 {
                    overlapping_combinations.push((i, j));
                }
            }
        }

        let mut a_overlapping = vec![false; a_stracks.len()];
        let mut b_overlapping = vec![false; b_stracks.len()];

        for &(i, j) in overlapping_combinations.iter() {
            let timep = a_stracks[i].frame_id - a_stracks[i].start_frame_id;
            let timeq = b_stracks[j].frame_id - b_stracks[j].start_frame_id;
            if timep > timeq {
                b_overlapping[j] = true;
            } else {
                a_overlapping[i] = true;
            }
        }

        for (i, a_strack) in a_stracks.iter().enumerate() {
            if !a_overlapping[i] {
                a_res.push(a_strack.clone());
            }
        }

        for (i, b_strack) in b_stracks.iter().enumerate() {
            if !b_overlapping[i] {
                b_res.push(b_strack.clone());
            }
        }

        (a_res, b_res)
    }

    /// Performs linear assignment using Hungarian algorithm
    pub(crate) fn linear_assignment(
        &self,
        cost_matrix: &[Vec<f32>],
        thresh: f32,
        n_detections: usize,
    ) -> Result<LinearAssignmentResult> {
        // Pre-allocate vectors with estimated capacity for better performance
        let mut matches = Vec::new();
        let mut a_unmatched = Vec::new();
        let mut b_unmatched = Vec::new();

        if cost_matrix.is_empty() {
            // If no tracks exist, all detections are unmatched
            b_unmatched.reserve(n_detections);
            for i in 0..n_detections {
                b_unmatched.push(i);
            }
            return Ok((matches, a_unmatched, b_unmatched));
        }

        // Convert f32 to f64 and prepare cost matrix
        let n_rows = cost_matrix.len();
        let n_cols = cost_matrix[0].len();

        // Pre-allocate vectors with known capacity
        matches.reserve(n_rows.min(n_cols)); // Maximum possible matches
        a_unmatched.reserve(n_rows);
        b_unmatched.reserve(n_cols);
        let mut cost_f64 = vec![vec![0.0; n_cols]; n_rows];

        for i in 0..n_rows {
            for j in 0..n_cols {
                cost_f64[i][j] = cost_matrix[i][j] as f64;
            }
        }

        // Extend cost matrix if needed
        let mut extended_cost = cost_f64;

        if n_rows != n_cols || thresh < f32::MAX {
            let n = n_rows + n_cols;
            extended_cost = vec![vec![0.0; n]; n];

            // Fill with threshold value
            let threshold_value = if thresh < f32::MAX {
                thresh as f64 / 2.0
            } else {
                // Find max cost and add 1
                let mut max_cost = -1.0;
                for row in cost_matrix.iter().take(n_rows) {
                    for &val in row.iter().take(n_cols) {
                        if val as f64 > max_cost {
                            max_cost = val as f64;
                        }
                    }
                }
                max_cost + 1.0
            };

            // Fill entire matrix with threshold
            for row in &mut extended_cost {
                for cell in row.iter_mut() {
                    *cell = threshold_value;
                }
            }

            // Set diagonal elements to 0 for extended part
            for (i, row) in extended_cost.iter_mut().enumerate().take(n).skip(n_rows) {
                row[i] = 0.0;
            }

            // Copy original cost matrix
            for (i, row) in cost_matrix.iter().enumerate().take(n_rows) {
                for (j, &val) in row.iter().enumerate().take(n_cols) {
                    extended_cost[i][j] = val as f64;
                }
            }
        }

        // Call lapjv algorithm
        let (row_indices, col_indices) = lapjv(&extended_cost)?;

        // Process results
        for (&row, &col) in row_indices.iter().zip(col_indices.iter()) {
            let row_idx = row as usize;
            let col_idx = col as usize;

            // Only consider assignments within original matrix bounds
            if row_idx < n_rows && col_idx < n_cols {
                matches.push((row_idx, col_idx as isize));
            }
        }

        // Find unmatched rows and columns using boolean arrays
        let mut matched_rows = vec![false; n_rows];
        let mut matched_cols = vec![false; n_cols];

        // Mark matched rows and columns
        for &(row, col) in &matches {
            matched_rows[row] = true;
            matched_cols[col as usize] = true;
        }

        // Collect unmatched rows and columns in a single pass

        for (i, &matched) in matched_rows.iter().enumerate() {
            if !matched {
                a_unmatched.push(i);
            }
        }

        for (i, &matched) in matched_cols.iter().enumerate() {
            if !matched {
                b_unmatched.push(i);
            }
        }

        Ok((matches, a_unmatched, b_unmatched))
    }

    /// Calculates IoU between two sets of bounding boxes
    pub fn calc_ious(a_hbbs: &[Hbb], b_hbbs: &[Hbb]) -> Vec<Vec<f32>> {
        let mut ious = vec![vec![0.0; b_hbbs.len()]; a_hbbs.len()];
        if a_hbbs.is_empty() || b_hbbs.is_empty() {
            return ious;
        }

        for (bi, b_hbb) in b_hbbs.iter().enumerate() {
            for (ai, a_hbb) in a_hbbs.iter().enumerate() {
                ious[ai][bi] = a_hbb.iou(b_hbb);
            }
        }

        ious
    }

    /// Calculates IoU distance between tracks
    pub(crate) fn calc_iou_distance(a_tracks: &[STrack], b_tracks: &[STrack]) -> Vec<Vec<f32>> {
        let a_len = a_tracks.len();
        let b_len = b_tracks.len();

        let mut cost_matrix = vec![vec![1.0; b_len]; a_len];
        if a_tracks.is_empty() || b_tracks.is_empty() {
            return cost_matrix;
        }

        for (ai, a_track) in a_tracks.iter().enumerate() {
            for (bi, b_track) in b_tracks.iter().enumerate() {
                let iou = a_track.hbb.iou(&b_track.hbb);
                cost_matrix[ai][bi] = 1.0 - iou;
            }
        }

        cost_matrix
    }

    /// Fuses detection scores with IoU distance for better matching
    pub(crate) fn fuse_score(cost_matrix: &mut [Vec<f32>], detections: &[STrack]) {
        if cost_matrix.is_empty() || detections.is_empty() {
            return;
        }

        for row in cost_matrix.iter_mut() {
            for (j, det) in detections.iter().enumerate() {
                if j < row.len() {
                    let iou_sim = 1.0 - row[j];
                    let fuse_sim = iou_sim * det.score();
                    row[j] = 1.0 - fuse_sim;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Hbb;

    #[test]
    fn test_calc_ious() {
        let hbbs_a = vec![
            Hbb::from_xywh(0.0, 0.0, 10.0, 10.0),
            Hbb::from_xywh(5.0, 5.0, 10.0, 10.0),
        ];
        let hbbs_b = vec![
            Hbb::from_xywh(0.0, 0.0, 10.0, 10.0),
            Hbb::from_xywh(20.0, 20.0, 10.0, 10.0),
        ];

        let ious = ByteTracker::calc_ious(&hbbs_a, &hbbs_b);

        // First hbb_a should have perfect overlap with first hbb_b
        assert!((ious[0][0] - 1.0).abs() < 0.01);
        // First hbb_a should have no overlap with second hbb_b
        assert!(ious[0][1] < 0.01);
    }

    #[test]
    fn test_linear_assignment_3x3() {
        let cost = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let tracker = ByteTracker::new(30, 0.5, 1.0, 0.8, 0.7, true);
        let (matches, a_unmatched, b_unmatched) =
            tracker.linear_assignment(&cost, f32::MAX, 3).unwrap();

        // Should have 3 matches
        assert_eq!(matches.len(), 3);
        assert!(a_unmatched.is_empty());
        assert!(b_unmatched.is_empty());

        // Check specific assignments
        let mut matched_rows: Vec<usize> = matches.iter().map(|(r, _)| *r).collect();
        let mut matched_cols: Vec<usize> = matches.iter().map(|(_, c)| *c as usize).collect();
        matched_rows.sort();
        matched_cols.sort();
        assert_eq!(matched_rows, vec![0, 1, 2]);
        assert_eq!(matched_cols, vec![0, 1, 2]);
    }

    #[test]
    fn test_linear_assignment_4x4() {
        let cost = vec![
            vec![1., 2., 3., 4.],
            vec![5., 6., 7., 8.],
            vec![9., 10., 11., 12.],
            vec![13., 14., 15., 16.],
        ];
        let tracker = ByteTracker::new(30, 0.5, 1.0, 0.8, 0.7, true);
        let (matches, a_unmatched, b_unmatched) =
            tracker.linear_assignment(&cost, f32::MAX, 3).unwrap();

        // Should have 4 matches
        assert_eq!(matches.len(), 4);
        assert!(a_unmatched.is_empty());
        assert!(b_unmatched.is_empty());

        // Check specific assignments
        let mut matched_rows: Vec<usize> = matches.iter().map(|(r, _)| *r).collect();
        let mut matched_cols: Vec<usize> = matches.iter().map(|(_, c)| *c as usize).collect();
        matched_rows.sort();
        matched_cols.sort();
        assert_eq!(matched_rows, vec![0, 1, 2, 3]);
        assert_eq!(matched_cols, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_tracker_new() {
        let tracker = ByteTracker::new(30, 0.25, 0.1, 0.25, 0.8, true);
        assert_eq!(tracker.frame_id, 0);
        assert_eq!(tracker.track_id_count, 0);
    }

    #[test]
    fn test_tracker_with_defaults() {
        let tracker = ByteTracker::default().with_max_age(30);
        assert_eq!(tracker.track_high_thresh, 0.25);
        assert_eq!(tracker.track_low_thresh, 0.1);
        assert_eq!(tracker.new_track_thresh, 0.25);
        assert_eq!(tracker.match_thresh, 0.8);
        assert!(tracker.fuse_score);
    }

    #[test]
    fn test_update() {
        // Reset global track ID counter
        STrack::reset_id();

        let mut tracker = ByteTracker::default().with_max_age(30);

        let hbbs = vec![
            Hbb::from_xywh(10.0, 10.0, 50.0, 100.0).with_confidence(0.9),
            Hbb::from_xywh(200.0, 200.0, 60.0, 120.0).with_confidence(0.85),
        ];

        let result = tracker.update(&hbbs);
        assert!(result.is_ok());

        let tracked = result.unwrap();
        assert_eq!(tracked.len(), 2, "Should create 2 tracks from 2 detections");

        // Verify track_ids are assigned and sequential
        let track_ids: Vec<usize> = tracked.iter().filter_map(|hbb| hbb.track_id()).collect();
        assert_eq!(track_ids.len(), 2, "Should have 2 track IDs");
        assert!(
            track_ids[0] > 0 && track_ids[1] > 0,
            "Track IDs should be positive"
        );
        assert!(
            track_ids[0] != track_ids[1],
            "Track IDs should be different"
        );
    }

    #[test]
    fn test_frame_id_starts_at_one() {
        // Reset global track ID counter
        STrack::reset_id();

        let mut tracker = ByteTracker::default().with_max_age(30);
        assert_eq!(tracker.frame_id, 0, "Initial frame_id should be 0");

        let hbbs = vec![Hbb::from_xywh(10.0, 10.0, 50.0, 100.0).with_confidence(0.9)];
        let _ = tracker.update(&hbbs);

        assert_eq!(
            tracker.frame_id, 1,
            "After first update, frame_id should be 1"
        );
    }

    #[test]
    fn test_detection_splitting() {
        // Reset global track ID counter
        STrack::reset_id();

        let mut tracker = ByteTracker::default().with_max_age(30);

        // Create detections with various scores
        let hbbs = vec![
            Hbb::from_xywh(10.0, 10.0, 50.0, 100.0).with_confidence(0.9), // >= 0.25 (high)
            Hbb::from_xywh(20.0, 20.0, 50.0, 100.0).with_confidence(0.3), // >= 0.25 (high)
            Hbb::from_xywh(30.0, 30.0, 50.0, 100.0).with_confidence(0.15), // > 0.1 && < 0.25 (low)
            Hbb::from_xywh(40.0, 40.0, 50.0, 100.0).with_confidence(0.05), // <= 0.1 (ignored)
        ];

        let result = tracker.update(&hbbs);
        assert!(result.is_ok());

        let tracked = result.unwrap();
        // Should only create tracks for the 2 high-confidence detections in first frame
        assert_eq!(
            tracked.len(),
            2,
            "Should create 2 tracks from high-confidence detections"
        );
    }

    #[test]
    fn test_track_persistence() {
        // Reset global track ID counter
        STrack::reset_id();

        let mut tracker = ByteTracker::default().with_max_age(30);

        // Frame 1: Create initial track
        let hbbs1 = vec![Hbb::from_xywh(10.0, 10.0, 50.0, 100.0).with_confidence(0.9)];
        let tracked1 = tracker.update(&hbbs1).unwrap();
        assert_eq!(tracked1.len(), 1);
        let track_id1 = tracked1[0].track_id().unwrap();
        let pos1 = (tracked1[0].x(), tracked1[0].y());

        // Frame 2: Same object moved slightly
        let hbbs2 = vec![Hbb::from_xywh(12.0, 12.0, 50.0, 100.0).with_confidence(0.9)];
        let tracked2 = tracker.update(&hbbs2).unwrap();
        assert_eq!(tracked2.len(), 1);
        let track_id2 = tracked2[0].track_id().unwrap();
        let pos2 = (tracked2[0].x(), tracked2[0].y());

        // Should maintain the same track ID
        assert_eq!(
            track_id1, track_id2,
            "Track ID should persist across frames"
        );

        // Position should update (Kalman filter smoothing)
        assert!(
            (pos2.0 - pos1.0).abs() > 0.0 || (pos2.1 - pos1.1).abs() > 0.0,
            "Track position should update! pos1={:?}, pos2={:?}",
            pos1,
            pos2
        );
    }
}
