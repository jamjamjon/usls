use crate::Point;

#[derive(Default, PartialOrd, PartialEq, Clone, Copy)]
pub struct RotatedRect {
    center: Point,
    width: f32,
    height: f32,
    rotation: f32, // (0, 90) radians
}

impl std::fmt::Debug for RotatedRect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RotatedRectangle")
            .field("height", &self.height)
            .field("width", &self.width)
            .field("center", &self.center)
            .field("rotation", &self.rotation)
            .field("vertices", &self.vertices())
            .finish()
    }
}

impl RotatedRect {
    pub fn new(center: Point, width: f32, height: f32, rotation: f32) -> Self {
        Self {
            center,
            width,
            height,
            rotation,
        }
    }

    pub fn vertices(&self) -> [Point; 4] {
        // [cos -sin]
        // [sin cos]
        let m = [
            [
                self.rotation.cos() * 0.5 * self.width,
                -self.rotation.sin() * 0.5 * self.height,
            ],
            [
                self.rotation.sin() * 0.5 * self.width,
                self.rotation.cos() * 0.5 * self.height,
            ],
        ];
        let v1 = self.center + Point::new(m[0][0] + m[0][1], m[1][0] + m[1][1]);
        let v2 = self.center + Point::new(m[0][0] - m[0][1], m[1][0] - m[1][1]);
        let v3 = self.center * 2.0 - v1;
        let v4 = self.center * 2.0 - v2;
        [v1, v2, v3, v4]
    }

    pub fn height(&self) -> f32 {
        self.height
    }

    pub fn width(&self) -> f32 {
        self.width
    }

    pub fn center(&self) -> Point {
        self.center
    }

    pub fn area(&self) -> f32 {
        self.height * self.width
    }

    // pub fn contain_point(&self, point: Point) -> bool {
    //     // ray casting
    //     todo!()
    // }
}

#[test]
fn test1() {
    let pi = std::f32::consts::PI;
    let rt = RotatedRect::new(
        Point::new(0.0f32, 0.0f32),
        2.0f32,
        4.0f32,
        pi / 180.0 * 90.0,
    );

    assert_eq!(
        rt.vertices(),
        [
            Point {
                x: -2.0,
                y: 0.99999994,
            },
            Point {
                x: 2.0,
                y: 1.0000001,
            },
            Point {
                x: 2.0,
                y: -0.99999994,
            },
            Point {
                x: -2.0,
                y: -1.0000001,
            },
        ]
    );
}

#[test]
fn test2() {
    let pi = std::f32::consts::PI;
    let rt = RotatedRect::new(
        Point::new(0.0f32, 0.0f32),
        2.0f32.sqrt(),
        2.0f32.sqrt(),
        pi / 180.0 * 45.0,
    );

    assert_eq!(
        rt.vertices(),
        [
            Point {
                x: 0.0,
                y: 0.99999994
            },
            Point {
                x: 0.99999994,
                y: 0.0
            },
            Point {
                x: 0.0,
                y: -0.99999994
            },
            Point {
                x: -0.99999994,
                y: 0.0
            },
        ]
    );
}

// #[test]
// fn contain_point() {
//     let pi = std::f32::consts::PI;
//     let rt = RotatedRect::new(
//         Point::new(0.0f32, 0.0f32),
//         1.0f32.sqrt(),
//         1.0f32.sqrt(),
//         pi / 180.0 * 45.0,
//     );

//     assert!(rt.contain_point(Point::new(0.0, 0.0)));
//     assert!(rt.contain_point(Point::new(0.5, 0.0)));
//     assert!(rt.contain_point(Point::new(0.0, 0.5)));

// }
