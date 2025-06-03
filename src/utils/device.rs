#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Device {
    Cpu(usize),
    Cuda(usize),
    TensorRt(usize),
    CoreMl(usize),
}

impl Default for Device {
    fn default() -> Self {
        Self::Cpu(0)
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let x = match self {
            Self::Cpu(i) => format!("CPU:{}", i),
            Self::Cuda(i) => format!("CUDA:{}(NVIDIA)", i),
            Self::TensorRt(i) => format!("TensorRT:{}(NVIDIA)", i),
            Self::CoreMl(i) => format!("CoreML:{}(Apple)", i),
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
            "trt" | "tensorrt" => Ok(Self::TensorRt(id)),
            "coreml" | "mps" => Ok(Self::CoreMl(id)),

            _ => anyhow::bail!("Unsupported device str: {s:?}."),
        }
    }
}

impl Device {
    pub fn id(&self) -> usize {
        match self {
            Self::Cpu(i) | Self::Cuda(i) | Self::TensorRt(i) | Self::CoreMl(i) => *i,
        }
    }
}
