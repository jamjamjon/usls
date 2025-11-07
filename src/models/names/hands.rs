//! Hand keypoint labels used by pose models.

/// 21 hand keypoints from wrist to fingertips (index-aligned, base→tip).
pub const NAMES_HAND_KEYPOINTS_21: [&str; 21] = [
    "wrist",          // Central point
    "thumb1",         // Thumb base
    "thumb2",         // Thumb joint 1
    "thumb3",         // Thumb joint 2
    "thumb4",         // Thumb tip
    "forefinger1",    // Index finger base
    "forefinger2",    // Index finger joint 1
    "forefinger3",    // Index finger joint 2
    "forefinger4",    // Index finger tip
    "middle_finger1", // Middle finger base
    "middle_finger2", // Middle finger joint 1
    "middle_finger3", // Middle finger joint 2
    "middle_finger4", // Middle finger tip
    "ring_finger1",   // Ring finger base
    "ring_finger2",   // Ring finger joint 1
    "ring_finger3",   // Ring finger joint 2
    "ring_finger4",   // Ring finger tip
    "pinky_finger1",  // Pinky finger base
    "pinky_finger2",  // Pinky finger joint 1
    "pinky_finger3",  // Pinky finger joint 2
    "pinky_finger4",  // Pinky finger tip
];
