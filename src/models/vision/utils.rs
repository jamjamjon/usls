/// Calculate the gaze projection endpoint given pitch, yaw angles and bbox coordinates.
///
/// This function is **generic for all 2D gaze estimation tasks** where:
/// - Gaze is represented as pitch (vertical) and yaw (horizontal) angles
/// - A bounding box defines the reference region (face, eye, etc.)
/// - The gaze line originates from the bbox center
///
/// # Arguments
/// * `pitch` - Pitch angle in radians (vertical rotation)
/// * `yaw` - Yaw angle in radians (horizontal rotation)
/// * `bbox` - Bounding box coordinates (x_min, y_min, x_max, y_max)
/// * `length_scale` - Optional scale factor for gaze line length (default: 1.0, uses bbox width)
///
/// # Returns
/// A tuple of (center_x, center_y, end_x, end_y) representing the gaze line
///
/// # Reference
/// Standard 2D gaze projection formula from computer vision literature:
/// ```python
/// x_center = (x_min + x_max) // 2
/// y_center = (y_min + y_max) // 2
/// length = x_max - x_min
/// dx = -length * sin(pitch) * cos(yaw)
/// dy = -length * sin(yaw)
/// ```
pub fn calculate_gaze_projection_2d(
    pitch: f32,
    yaw: f32,
    bbox: (f32, f32, f32, f32),
    length_scale: Option<f32>,
) -> (f32, f32, f32, f32) {
    let (x_min, y_min, x_max, y_max) = bbox;

    // Calculate center of the bounding box
    let x_center = (x_min + x_max) / 2.0;
    let y_center = (y_min + y_max) / 2.0;

    // Use bbox width as base gaze line length, with optional scaling
    let length = (x_max - x_min) * length_scale.unwrap_or(1.0);

    // Calculate gaze direction offset using standard 2D projection
    // Negative sign for pitch ensures correct vertical direction
    let dx = -length * pitch.sin() * yaw.cos();
    let dy = -length * yaw.sin();

    // Calculate end point
    let x_end = x_center + dx;
    let y_end = y_center + dy;

    (x_center, y_center, x_end, y_end)
}
