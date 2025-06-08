#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// Device types for model execution.
pub enum Device {
    Cpu(usize),
    Cuda(usize),
    TensorRt(usize),
    OpenVino(&'static str),
    DirectMl(usize),
    Cann(usize),
    Rocm(usize),
    Qnn(usize),
    MiGraphX(usize),
    CoreMl,
    Xnnpack,
    RkNpu,
    OneDnn,
    Acl,
    NnApi,
    ArmNn,
    Tvm,
    Vitis,
    Azure,
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
            Self::Cann(i) => format!("CANN:{}(Huawei)", i),
            Self::OpenVino(s) => format!("OpenVINO:{}(Intel)", s),
            Self::DirectMl(i) => format!("DirectML:{}(Microsoft)", i),
            Self::Qnn(i) => format!("QNN:{}(Qualcomm)", i),
            Self::MiGraphX(i) => format!("MIGraphX:{}(AMD)", i),
            Self::Rocm(i) => format!("ROCm:{}(AMD)", i),
            Self::CoreMl => "CoreML(Apple)".to_string(),
            Self::Azure => "Azure(Microsoft)".to_string(),
            Self::Xnnpack => "XNNPACK".to_string(),
            Self::OneDnn => "oneDNN(Intel)".to_string(),
            Self::RkNpu => "RKNPU".to_string(),
            Self::Acl => "ACL(Arm)".to_string(),
            Self::NnApi => "NNAPI(Android)".to_string(),
            Self::ArmNn => "ArmNN(Arm)".to_string(),
            Self::Tvm => "TVM(Apache)".to_string(),
            Self::Vitis => "VitisAI(AMD)".to_string(),
        };
        write!(f, "{}", x)
    }
}

impl std::str::FromStr for Device {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        #[inline]
        fn parse_device_id(id_str: Option<&str>) -> usize {
            id_str
                .map(|s| s.trim().parse::<usize>().unwrap_or(0))
                .unwrap_or(0)
        }

        // Use split_once for better performance - no Vec allocation
        let (device_type, id_part) = s
            .trim()
            .split_once(':')
            .map_or_else(|| (s.trim(), None), |(device, id)| (device, Some(id)));

        match device_type.to_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu(parse_device_id(id_part))),
            "cuda" => Ok(Self::Cuda(parse_device_id(id_part))),
            "trt" | "tensorrt" => Ok(Self::TensorRt(parse_device_id(id_part))),
            "coreml" | "mps" => Ok(Self::CoreMl),
            "openvino" => {
                // For OpenVino, use the user input directly after first colon (trimmed)
                let device_spec = id_part.map(|s| s.trim()).unwrap_or("CPU"); // Default to CPU if no specification provided
                Ok(Self::OpenVino(Box::leak(
                    device_spec.to_string().into_boxed_str(),
                )))
            }
            "directml" => Ok(Self::DirectMl(parse_device_id(id_part))),
            "xnnpack" => Ok(Self::Xnnpack),
            "cann" => Ok(Self::Cann(parse_device_id(id_part))),
            "rknpu" => Ok(Self::RkNpu),
            "onednn" => Ok(Self::OneDnn),
            "acl" => Ok(Self::Acl),
            "rocm" => Ok(Self::Rocm(parse_device_id(id_part))),
            "nnapi" => Ok(Self::NnApi),
            "armnn" => Ok(Self::ArmNn),
            "tvm" => Ok(Self::Tvm),
            "qnn" => Ok(Self::Qnn(parse_device_id(id_part))),
            "migraphx" => Ok(Self::MiGraphX(parse_device_id(id_part))),
            "vitisai" => Ok(Self::Vitis),
            "azure" => Ok(Self::Azure),
            _ => anyhow::bail!("Unsupported device str: {s:?}."),
        }
    }
}

impl Device {
    pub fn id(&self) -> Option<usize> {
        match self {
            Self::Cpu(i)
            | Self::Cuda(i)
            | Self::TensorRt(i)
            | Self::Cann(i)
            | Self::Qnn(i)
            | Self::Rocm(i)
            | Self::MiGraphX(i)
            | Self::DirectMl(i) => Some(*i),
            Self::OpenVino(_)
            | Self::Xnnpack
            | Self::CoreMl
            | Self::RkNpu
            | Self::OneDnn
            | Self::NnApi
            | Self::Azure
            | Self::Vitis
            | Self::ArmNn
            | Self::Tvm
            | Self::Acl => None,
        }
    }
}
