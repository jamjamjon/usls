#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Device {
    Auto(usize),
    Cpu(usize),
    Cuda(usize),
    TensorRT(usize),
    CoreML(usize),
    // Cann(usize),
    // Acl(usize),
    // Rocm(usize),
    // Rknpu(usize),
    // Openvino(usize),
    // Onednn(usize),
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let x = match self {
            Self::Auto(i) => format!("auto:{}", i),
            Self::Cpu(i) => format!("cpu:{}", i),
            Self::Cuda(i) => format!("cuda:{}", i),
            Self::TensorRT(i) => format!("tensorrt:{}", i),
            Self::CoreML(i) => format!("mps:{}", i),
        };
        write!(f, "{}", x)
    }
}

impl TryFrom<&str> for Device {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        // device and its id
        let d_id: Vec<&str> = s.trim().split(':').collect();
        let (d, id) = match d_id.len() {
            1 => (d_id[0].trim(), 0),
            2 => (d_id[0].trim(), d_id[1].trim().parse::<usize>().unwrap_or(0)),
            _ => anyhow::bail!(
                "Fail to parse device string: {s}. Expect: `device:device_id` or `device`. e.g. `cuda:0` or `cuda`"
            ),
        };
        // TODO: device-id checking
        match d.to_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu(id)),
            "cuda" => Ok(Self::Cuda(id)),
            "trt" | "tensorrt" => Ok(Self::TensorRT(id)),
            "coreml" | "mps" => Ok(Self::CoreML(id)),
            _ => anyhow::bail!("Unsupported device str: {s:?}."),
        }
    }
}

impl Device {
    pub fn id(&self) -> usize {
        match self {
            Device::Auto(i) => *i,
            Device::Cpu(i) => *i,
            Device::Cuda(i) => *i,
            Device::TensorRT(i) => *i,
            Device::CoreML(i) => *i,
        }
    }
}
