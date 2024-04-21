#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Device {
    Cpu(usize),
    Cuda(usize),
    Trt(usize),
    CoreML(usize),
    // Cann(usize),
    // Acl(usize),
    // Rocm(usize),
    // Rknpu(usize),
    // Openvino(usize),
    // Onednn(usize),
}
