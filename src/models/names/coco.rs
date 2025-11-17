//! COCO dataset labels used by detection and pose tasks.

/// COCO 80-class object categories (common split).
pub const NAMES_COCO_80: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// COCO 91-class extended categories (keeps original index mapping with gaps).
pub const NAMES_COCO_91: [&str; 91] = [
    "background",     // 0
    "person",         // 1
    "bicycle",        // 2
    "car",            // 3
    "motorcycle",     // 4
    "airplane",       // 5
    "bus",            // 6
    "train",          // 7
    "truck",          // 8
    "boat",           // 9
    "traffic light",  // 10
    "fire hydrant",   // 11
    "unused",         // 12
    "stop sign",      // 13
    "parking meter",  // 14
    "bench",          // 15
    "bird",           // 16
    "cat",            // 17
    "dog",            // 18
    "horse",          // 19
    "sheep",          // 20
    "cow",            // 21
    "elephant",       // 22
    "bear",           // 23
    "zebra",          // 24
    "giraffe",        // 25
    "unused",         // 26
    "backpack",       // 27
    "umbrella",       // 28
    "unused",         // 29
    "unused",         // 30
    "handbag",        // 31
    "tie",            // 32
    "suitcase",       // 33
    "frisbee",        // 34
    "skis",           // 35
    "snowboard",      // 36
    "sports ball",    // 37
    "kite",           // 38
    "baseball bat",   // 39
    "baseball glove", // 40
    "skateboard",     // 41
    "surfboard",      // 42
    "tennis racket",  // 43
    "bottle",         // 44
    "unused",         // 45
    "wine glass",     // 46
    "cup",            // 47
    "fork",           // 48
    "knife",          // 49
    "spoon",          // 50
    "bowl",           // 51
    "banana",         // 52
    "apple",          // 53
    "sandwich",       // 54
    "orange",         // 55
    "broccoli",       // 56
    "carrot",         // 57
    "hot dog",        // 58
    "pizza",          // 59
    "donut",          // 60
    "cake",           // 61
    "chair",          // 62
    "couch",          // 63
    "potted plant",   // 64
    "bed",            // 65
    "unused",         // 66
    "dining table",   // 67
    "unused",         // 68
    "unused",         // 69
    "toilet",         // 70
    "unused",         // 71
    "tv",             // 72
    "laptop",         // 73
    "mouse",          // 74
    "remote",         // 75
    "keyboard",       // 76
    "cell phone",     // 77
    "microwave",      // 78
    "oven",           // 79
    "toaster",        // 80
    "sink",           // 81
    "refrigerator",   // 82
    "book",           // 83
    "unused",         // 84
    "clock",          // 85
    "vase",           // 86
    "scissors",       // 87
    "teddy bear",     // 88
    "hair drier",     // 89
    "toothbrush",     // 90
];

/// COCO 17 human keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).
pub const NAMES_COCO_KEYPOINTS_17: [&str; 17] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
];

/// COCO-WholeBody 133 keypoints (body, face, and hands combined).
pub const NAMES_COCO_KEYPOINTS_133: [&str; 133] = [
    // Body keypoints (0-22)
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    // Face landmarks (23-90)
    "face-0",
    "face-1",
    "face-2",
    "face-3",
    "face-4",
    "face-5",
    "face-6",
    "face-7",
    "face-8",
    "face-9",
    "face-10",
    "face-11",
    "face-12",
    "face-13",
    "face-14",
    "face-15",
    "face-16",
    "face-17",
    "face-18",
    "face-19",
    "face-20",
    "face-21",
    "face-22",
    "face-23",
    "face-24",
    "face-25",
    "face-26",
    "face-27",
    "face-28",
    "face-29",
    "face-30",
    "face-31",
    "face-32",
    "face-33",
    "face-34",
    "face-35",
    "face-36",
    "face-37",
    "face-38",
    "face-39",
    "face-40",
    "face-41",
    "face-42",
    "face-43",
    "face-44",
    "face-45",
    "face-46",
    "face-47",
    "face-48",
    "face-49",
    "face-50",
    "face-51",
    "face-52",
    "face-53",
    "face-54",
    "face-55",
    "face-56",
    "face-57",
    "face-58",
    "face-59",
    "face-60",
    "face-61",
    "face-62",
    "face-63",
    "face-64",
    "face-65",
    "face-66",
    "face-67",
    // Left hand keypoints (91-111)
    "left_hand_root",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "left_thumb4",
    "left_forefinger1",
    "left_forefinger2",
    "left_forefinger3",
    "left_forefinger4",
    "left_middle_finger1",
    "left_middle_finger2",
    "left_middle_finger3",
    "left_middle_finger4",
    "left_ring_finger1",
    "left_ring_finger2",
    "left_ring_finger3",
    "left_ring_finger4",
    "left_pinky_finger1",
    "left_pinky_finger2",
    "left_pinky_finger3",
    "left_pinky_finger4",
    // Right hand keypoints (112-132)
    "right_hand_root",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "right_thumb4",
    "right_forefinger1",
    "right_forefinger2",
    "right_forefinger3",
    "right_forefinger4",
    "right_middle_finger1",
    "right_middle_finger2",
    "right_middle_finger3",
    "right_middle_finger4",
    "right_ring_finger1",
    "right_ring_finger2",
    "right_ring_finger3",
    "right_ring_finger4",
    "right_pinky_finger1",
    "right_pinky_finger2",
    "right_pinky_finger3",
    "right_pinky_finger4",
];

/// HALPE 26 keypoints (body-centric extension with detailed feet).
pub const NAMES_HALPE_KEYPOINTS_26: [&str; 26] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "head",
    "neck",
    "hip",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
];
