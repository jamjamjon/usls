use crate::Color;

/// Connection between two keypoints with optional color.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Connection {
    pub indices: (usize, usize),
    pub color: Option<Color>,
}

impl From<(usize, usize)> for Connection {
    fn from(indices: (usize, usize)) -> Self {
        Self {
            indices,
            color: None,
        }
    }
}

impl From<(usize, usize, Color)> for Connection {
    fn from((a, b, color): (usize, usize, Color)) -> Self {
        Self {
            indices: (a, b),
            color: Some(color),
        }
    }
}

/// Skeleton structure containing keypoint connections.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Skeleton {
    pub connections: Vec<Connection>,
}

impl std::ops::Deref for Skeleton {
    type Target = Vec<Connection>;

    fn deref(&self) -> &Self::Target {
        &self.connections
    }
}

impl Skeleton {
    pub fn with_connections<C: Into<Connection> + Clone>(mut self, connections: &[C]) -> Self {
        self.connections = connections.iter().cloned().map(|c| c.into()).collect();
        self
    }

    pub fn with_colors(mut self, colors: &[Color]) -> Self {
        for (i, connection) in self.connections.iter_mut().enumerate() {
            if i < colors.len() {
                connection.color = Some(colors[i]);
            }
        }
        self
    }
}

impl From<&[(usize, usize)]> for Skeleton {
    fn from(connections: &[(usize, usize)]) -> Self {
        Self {
            connections: connections.iter().map(|&c| c.into()).collect(),
        }
    }
}

impl<const N: usize> From<[(usize, usize); N]> for Skeleton {
    fn from(arr: [(usize, usize); N]) -> Self {
        Self::from(arr.as_slice())
    }
}

impl From<(&[(usize, usize)], &[Color])> for Skeleton {
    fn from((connections, colors): (&[(usize, usize)], &[Color])) -> Self {
        Self {
            connections: connections
                .iter()
                .zip(colors.iter())
                .map(|(&(a, b), &c)| (a, b, c).into())
                .collect(),
        }
    }
}

impl<const N: usize> From<([(usize, usize); N], [Color; N])> for Skeleton {
    fn from((connections, colors): ([(usize, usize); N], [Color; N])) -> Self {
        Skeleton::from((&connections[..], &colors[..]))
    }
}

/// Defines the keypoint connections for the COCO person skeleton with 19 connections.
/// Each tuple (a, b) represents a connection between keypoint indices a and b.
/// The connections define the following body parts:
/// - Upper body: shoulders, elbows, wrists
/// - Torso: shoulders to hips
/// - Lower body: hips, knees, ankles
pub const SKELETON_COCO_19: [(usize, usize); 19] = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
];

/// Defines colors for visualizing each connection in the COCO person skeleton.
/// Colors are grouped by body parts:
/// - Blue (0x3399ff): Upper limbs
/// - Pink (0xff33ff): Torso
/// - Orange (0xff8000): Lower limbs  
/// - Green (0x00ff00): Head and neck
pub const SKELETON_COLOR_COCO_19: [Color; 19] = [
    Color(0x3399ffff),
    Color(0x3399ffff),
    Color(0x3399ffff),
    Color(0x3399ffff),
    Color(0xff33ffff),
    Color(0xff33ffff),
    Color(0xff33ffff),
    Color(0xff8000ff),
    Color(0xff8000ff),
    Color(0xff8000ff),
    Color(0xff8000ff),
    Color(0xff8000ff),
    Color(0x00ff00ff),
    Color(0x00ff00ff),
    Color(0x00ff00ff),
    Color(0x00ff00ff),
    Color(0x00ff00ff),
    Color(0x00ff00ff),
    Color(0x00ff00ff),
];

/// Defines the keypoint connections for the HALPE person skeleton with 27 connections.
/// Each tuple (a, b) represents a connection between keypoint indices a and b.
/// The connections define the following body parts:
/// - Upper body:
///   - Head and neck connections
///   - Shoulders to elbows to wrists
/// - Torso:
///   - Neck to hip
///   - Shoulders to neck
/// - Lower body:
///   - Hip to knees to ankles
///   - Ankles to toes and heels
pub const SKELETON_HALPE_27: [(usize, usize); 27] = [
    (15, 13), // left_ankle -> left_knee
    (13, 11), // left_knee -> left_hip
    (11, 19), // left_hip -> hip
    (16, 14), // right_ankle -> right_knee
    (14, 12), // right_knee -> right_hip
    (12, 19), // right_hip -> hip
    (17, 18), // head -> neck
    (18, 19), // neck -> hip
    (18, 5),  // neck -> left_shoulder
    (5, 7),   // left_shoulder -> left_elbow
    (7, 9),   // left_elbow -> left_wrist
    (18, 6),  // neck -> right_shoulder
    (6, 8),   // right_shoulder -> right_elbow
    (8, 10),  // right_elbow -> right_wrist
    (1, 2),   // left_eye -> right_eye
    (0, 1),   // nose -> left_eye
    (0, 2),   // nose -> right_eye
    (1, 3),   // left_eye -> left_ear
    (2, 4),   // right_eye -> right_ear
    (3, 5),   // left_ear -> left_shoulder
    (4, 6),   // right_ear -> right_shoulder
    (15, 20), // left_ankle -> left_big_toe
    (15, 22), // left_ankle -> left_small_toe
    (15, 24), // left_ankle -> left_heel
    (16, 21), // right_ankle -> right_big_toe
    (16, 23), // right_ankle -> right_small_toe
    (16, 25), // right_ankle -> right_heel
];

/// Defines colors for visualizing each connection in the HALPE person skeleton.
/// Colors are grouped by body parts and sides:
/// - Blue (0x3399ff): Face and central connections (eyes, neck to hip)
/// - Green (0x00ff00): Left side limbs and extremities
/// - Orange (0xff8000): Right side limbs and extremities
pub const SKELETON_COLOR_HALPE_27: [Color; 27] = [
    Color(0x00ff00ff), // left_ankle -> left_knee
    Color(0x00ff00ff), // left_knee -> left_hip
    Color(0x00ff00ff), // left_hip -> hip
    Color(0xff8000ff), // right_ankle -> right_knee
    Color(0xff8000ff), // right_knee -> right_hip
    Color(0xff8000ff), // right_hip -> hip
    Color(0x3399ffff), // head -> neck
    Color(0x3399ffff), // neck -> hip
    Color(0x00ff00ff), // neck -> left_shoulder
    Color(0x00ff00ff), // left_shoulder -> left_elbow
    Color(0x00ff00ff), // left_elbow -> left_wrist
    Color(0xff8000ff), // neck -> right_shoulder
    Color(0xff8000ff), // right_shoulder -> right_elbow
    Color(0xff8000ff), // right_elbow -> right_wrist
    Color(0x3399ffff), // left_eye -> right_eye
    Color(0x3399ffff), // nose -> left_eye
    Color(0x3399ffff), // nose -> right_eye
    Color(0x3399ffff), // left_eye -> left_ear
    Color(0x3399ffff), // right_eye -> right_ear
    Color(0x3399ffff), // left_ear -> left_shoulder
    Color(0x3399ffff), // right_ear -> right_shoulder
    Color(0x00ff00ff), // left_ankle -> left_big_toe
    Color(0x00ff00ff), // left_ankle -> left_small_toe
    Color(0x00ff00ff), // left_ankle -> left_heel
    Color(0xff8000ff), // right_ankle -> right_big_toe
    Color(0xff8000ff), // right_ankle -> right_small_toe
    Color(0xff8000ff), // right_ankle -> right_heel
];

/// Defines the keypoint connections for the hand skeleton with 20 connections.
/// Each tuple (a, b) represents a connection between keypoint indices a and b.
/// The connections define the following parts:
/// - Thumb: wrist -> thumb1 -> thumb2 -> thumb3 -> thumb4
/// - Index: wrist -> forefinger1 -> forefinger2 -> forefinger3 -> forefinger4
/// - Middle: wrist -> middle_finger1 -> middle_finger2 -> middle_finger3 -> middle_finger4
/// - Ring: wrist -> ring_finger1 -> ring_finger2 -> ring_finger3 -> ring_finger4
/// - Pinky: wrist -> pinky_finger1 -> pinky_finger2 -> pinky_finger3 -> pinky_finger4
pub const SKELETON_HAND_21: [(usize, usize); 20] = [
    (0, 1),   // wrist -> thumb1
    (1, 2),   // thumb1 -> thumb2
    (2, 3),   // thumb2 -> thumb3
    (3, 4),   // thumb3 -> thumb4
    (0, 5),   // wrist -> forefinger1
    (5, 6),   // forefinger1 -> forefinger2
    (6, 7),   // forefinger2 -> forefinger3
    (7, 8),   // forefinger3 -> forefinger4
    (0, 9),   // wrist -> middle_finger1
    (9, 10),  // middle_finger1 -> middle_finger2
    (10, 11), // middle_finger2 -> middle_finger3
    (11, 12), // middle_finger3 -> middle_finger4
    (0, 13),  // wrist -> ring_finger1
    (13, 14), // ring_finger1 -> ring_finger2
    (14, 15), // ring_finger2 -> ring_finger3
    (15, 16), // ring_finger3 -> ring_finger4
    (0, 17),  // wrist -> pinky_finger1
    (17, 18), // pinky_finger1 -> pinky_finger2
    (18, 19), // pinky_finger2 -> pinky_finger3
    (19, 20), // pinky_finger3 -> pinky_finger4
];

/// Defines colors for visualizing each connection in the hand skeleton.
/// Colors are grouped by fingers:
/// - Thumb: Orange (0xff8000)
/// - Index: Pink (0xff99ff)
/// - Middle: Light Blue (0x66b2ff)
/// - Ring: Red (0xff3333)
/// - Pinky: Green (0x00ff00)
pub const SKELETON_COLOR_HAND_21: [Color; 20] = [
    Color(0xff8000ff), // wrist -> thumb1
    Color(0xff8000ff), // thumb1 -> thumb2
    Color(0xff8000ff), // thumb2 -> thumb3
    Color(0xff8000ff), // thumb3 -> thumb4
    Color(0xff99ffff), // wrist -> forefinger1
    Color(0xff99ffff), // forefinger1 -> forefinger2
    Color(0xff99ffff), // forefinger2 -> forefinger3
    Color(0xff99ffff), // forefinger3 -> forefinger4
    Color(0x66b2ffff), // wrist -> middle_finger1
    Color(0x66b2ffff), // middle_finger1 -> middle_finger2
    Color(0x66b2ffff), // middle_finger2 -> middle_finger3
    Color(0x66b2ffff), // middle_finger3 -> middle_finger4
    Color(0xff3333ff), // wrist -> ring_finger1
    Color(0xff3333ff), // ring_finger1 -> ring_finger2
    Color(0xff3333ff), // ring_finger2 -> ring_finger3
    Color(0xff3333ff), // ring_finger3 -> ring_finger4
    Color(0x00ff00ff), // wrist -> pinky_finger1
    Color(0x00ff00ff), // pinky_finger1 -> pinky_finger2
    Color(0x00ff00ff), // pinky_finger2 -> pinky_finger3
    Color(0x00ff00ff), // pinky_finger3 -> pinky_finger4
];

/// Defines the keypoint connections for the COCO-133 person skeleton with 65 connections.
/// Each tuple (a, b) represents a connection between keypoint indices a and b.
/// The connections define the following parts:
/// - Body parts: shoulders, elbows, wrists, hips, knees, ankles, toes
/// - Face: detailed face landmarks
/// - Hands: detailed finger joints for both hands
pub const SKELETON_COCO_65: [(usize, usize); 65] = [
    (15, 13),   // left_ankle -> left_knee
    (13, 11),   // left_knee -> left_hip
    (16, 14),   // right_ankle -> right_knee
    (14, 12),   // right_knee -> right_hip
    (11, 12),   // left_hip -> right_hip
    (5, 11),    // left_shoulder -> left_hip
    (6, 12),    // right_shoulder -> right_hip
    (5, 6),     // left_shoulder -> right_shoulder
    (5, 7),     // left_shoulder -> left_elbow
    (6, 8),     // right_shoulder -> right_elbow
    (7, 9),     // left_elbow -> left_wrist
    (8, 10),    // right_elbow -> right_wrist
    (1, 2),     // left_eye -> right_eye
    (0, 1),     // nose -> left_eye
    (0, 2),     // nose -> right_eye
    (1, 3),     // left_eye -> left_ear
    (2, 4),     // right_eye -> right_ear
    (3, 5),     // left_ear -> left_shoulder
    (4, 6),     // right_ear -> right_shoulder
    (15, 17),   // left_ankle -> left_big_toe
    (15, 18),   // left_ankle -> left_small_toe
    (15, 19),   // left_ankle -> left_heel
    (16, 20),   // right_ankle -> right_big_toe
    (16, 21),   // right_ankle -> right_small_toe
    (16, 22),   // right_ankle -> right_heel
    (91, 92),   // left_hand_root -> left_thumb1
    (92, 93),   // left_thumb1 -> left_thumb2
    (93, 94),   // left_thumb2 -> left_thumb3
    (94, 95),   // left_thumb3 -> left_thumb4
    (91, 96),   // left_hand_root -> left_forefinger1
    (96, 97),   // left_forefinger1 -> left_forefinger2
    (97, 98),   // left_forefinger2 -> left_forefinger3
    (98, 99),   // left_forefinger3 -> left_forefinger4
    (91, 100),  // left_hand_root -> left_middle_finger1
    (100, 101), // left_middle_finger1 -> left_middle_finger2
    (101, 102), // left_middle_finger2 -> left_middle_finger3
    (102, 103), // left_middle_finger3 -> left_middle_finger4
    (91, 104),  // left_hand_root -> left_ring_finger1
    (104, 105), // left_ring_finger1 -> left_ring_finger2
    (105, 106), // left_ring_finger2 -> left_ring_finger3
    (106, 107), // left_ring_finger3 -> left_ring_finger4
    (91, 108),  // left_hand_root -> left_pinky_finger1
    (108, 109), // left_pinky_finger1 -> left_pinky_finger2
    (109, 110), // left_pinky_finger2 -> left_pinky_finger3
    (110, 111), // left_pinky_finger3 -> left_pinky_finger4
    (112, 113), // right_hand_root -> right_thumb1
    (113, 114), // right_thumb1 -> right_thumb2
    (114, 115), // right_thumb2 -> right_thumb3
    (115, 116), // right_thumb3 -> right_thumb4
    (112, 117), // right_hand_root -> right_forefinger1
    (117, 118), // right_forefinger1 -> right_forefinger2
    (118, 119), // right_forefinger2 -> right_forefinger3
    (119, 120), // right_forefinger3 -> right_forefinger4
    (112, 121), // right_hand_root -> right_middle_finger1
    (121, 122), // right_middle_finger1 -> right_middle_finger2
    (122, 123), // right_middle_finger2 -> right_middle_finger3
    (123, 124), // right_middle_finger3 -> right_middle_finger4
    (112, 125), // right_hand_root -> right_ring_finger1
    (125, 126), // right_ring_finger1 -> right_ring_finger2
    (126, 127), // right_ring_finger2 -> right_ring_finger3
    (127, 128), // right_ring_finger3 -> right_ring_finger4
    (112, 129), // right_hand_root -> right_pinky_finger1
    (129, 130), // right_pinky_finger1 -> right_pinky_finger2
    (130, 131), // right_pinky_finger2 -> right_pinky_finger3
    (131, 132), // right_pinky_finger3 -> right_pinky_finger4
];

/// Defines colors for visualizing each connection in the COCO-133 person skeleton.
/// Colors are grouped by body parts:
/// - Blue (0x3399ff): Face and central body connections
/// - Green (0x00ff00): Left side body parts
/// - Orange (0xff8000): Right side body parts
///
/// For hands:
/// - Orange (0xff8000): Thumb
/// - Pink (0xff99ff): Index finger
/// - Light Blue (0x66b2ff): Middle finger
/// - Red (0xff3333): Ring finger
/// - Green (0x00ff00): Pinky finger
pub const SKELETON_COLOR_COCO_65: [Color; 65] = [
    Color(0x00ff00ff), // left_ankle -> left_knee
    Color(0x00ff00ff), // left_knee -> left_hip
    Color(0xff8000ff), // right_ankle -> right_knee
    Color(0xff8000ff), // right_knee -> right_hip
    Color(0x3399ffff), // left_hip -> right_hip
    Color(0x3399ffff), // left_shoulder -> left_hip
    Color(0x3399ffff), // right_shoulder -> right_hip
    Color(0x3399ffff), // left_shoulder -> right_shoulder
    Color(0x00ff00ff), // left_shoulder -> left_elbow
    Color(0xff8000ff), // right_shoulder -> right_elbow
    Color(0x00ff00ff), // left_elbow -> left_wrist
    Color(0xff8000ff), // right_elbow -> right_wrist
    Color(0x3399ffff), // left_eye -> right_eye
    Color(0x3399ffff), // nose -> left_eye
    Color(0x3399ffff), // nose -> right_eye
    Color(0x3399ffff), // left_eye -> left_ear
    Color(0x3399ffff), // right_eye -> right_ear
    Color(0x3399ffff), // left_ear -> left_shoulder
    Color(0x3399ffff), // right_ear -> right_shoulder
    Color(0x00ff00ff), // left_ankle -> left_big_toe
    Color(0x00ff00ff), // left_ankle -> left_small_toe
    Color(0x00ff00ff), // left_ankle -> left_heel
    Color(0xff8000ff), // right_ankle -> right_big_toe
    Color(0xff8000ff), // right_ankle -> right_small_toe
    Color(0xff8000ff), // right_ankle -> right_heel
    // Left hand
    Color(0xff8000ff), // left_thumb connections
    Color(0xff8000ff),
    Color(0xff8000ff),
    Color(0xff8000ff),
    Color(0xff99ffff), // left_forefinger connections
    Color(0xff99ffff),
    Color(0xff99ffff),
    Color(0xff99ffff),
    Color(0x66b2ffff), // left_middle_finger connections
    Color(0x66b2ffff),
    Color(0x66b2ffff),
    Color(0x66b2ffff),
    Color(0xff3333ff), // left_ring_finger connections
    Color(0xff3333ff),
    Color(0xff3333ff),
    Color(0xff3333ff),
    Color(0x00ff00ff), // left_pinky_finger connections
    Color(0x00ff00ff),
    Color(0x00ff00ff),
    Color(0x00ff00ff),
    // Right hand
    Color(0xff8000ff), // right_thumb connections
    Color(0xff8000ff),
    Color(0xff8000ff),
    Color(0xff8000ff),
    Color(0xff99ffff), // right_forefinger connections
    Color(0xff99ffff),
    Color(0xff99ffff),
    Color(0xff99ffff),
    Color(0x66b2ffff), // right_middle_finger connections
    Color(0x66b2ffff),
    Color(0x66b2ffff),
    Color(0x66b2ffff),
    Color(0xff3333ff), // right_ring_finger connections
    Color(0xff3333ff),
    Color(0xff3333ff),
    Color(0xff3333ff),
    Color(0x00ff00ff), // right_pinky_finger connections
    Color(0x00ff00ff),
    Color(0x00ff00ff),
    Color(0x00ff00ff),
];
