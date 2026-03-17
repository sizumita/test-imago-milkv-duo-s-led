#![cfg_attr(test, allow(dead_code))]

#[cfg(target_arch = "wasm32")]
wit_bindgen::generate!({
    path: "wit",
    world: "plugin-imports",
    generate_all
});

#[cfg(any(target_arch = "wasm32", test))]
use font8x8::{BASIC_FONTS, UnicodeFonts};
#[cfg(any(target_arch = "wasm32", test))]
use serde::Deserialize;
#[cfg(any(target_arch = "wasm32", test))]
use std::cmp::Ordering;

#[cfg(target_arch = "wasm32")]
use self::imago::usb::{
    device::Device,
    provider,
    types::{ControlSetup, ControlType, Recipient, TransferType, UsbError},
    usb_interface::ClaimedInterface,
};
#[cfg(target_arch = "wasm32")]
use self::wasi::nn::{
    errors::Error as WasiNnError,
    graph::{self, ExecutionTarget, Graph, GraphEncoding},
    tensor::{Tensor, TensorType},
};

#[cfg(any(target_arch = "wasm32", test))]
const MODEL_PATH: &str = "/app/assets/models/yolo.cvimodel";
#[cfg(any(target_arch = "wasm32", test))]
const MODEL_CONFIG_PATH: &str = "/app/assets/models/yolo.toml";
#[cfg(any(target_arch = "wasm32", test))]
const OUTPUT_PATH: &str = "/captures/last-frame.jpg";
#[cfg(any(target_arch = "wasm32", test))]
const OUTPUT_TMP_PATH: &str = "/captures/last-frame.jpg.tmp";
#[cfg(any(target_arch = "wasm32", test))]
const LETTERBOX_FILL: u8 = 114;
#[cfg(any(target_arch = "wasm32", test))]
const JPEG_QUALITY: u8 = 85;
#[cfg(any(target_arch = "wasm32", test))]
const BOX_THICKNESS: u32 = 2;

#[cfg(target_arch = "wasm32")]
const TARGET_DEVICE_PATH: &str = "/dev/bus/usb/001/002";
#[cfg(target_arch = "wasm32")]
const TARGET_VENDOR_ID: u16 = 0x291a;
#[cfg(target_arch = "wasm32")]
const TARGET_PRODUCT_ID: u16 = 0x3361;
#[cfg(target_arch = "wasm32")]
const STREAM_INTERFACE_NUMBER: u8 = 3;
#[cfg(target_arch = "wasm32")]
const STREAM_ALT_SETTING_IDLE: u8 = 0;
#[cfg(target_arch = "wasm32")]
const STREAM_ALT_SETTING_ACTIVE: u8 = 1;
#[cfg(target_arch = "wasm32")]
const STREAM_ENDPOINT_ADDRESS: u8 = 0x83;
#[cfg(target_arch = "wasm32")]
const MJPEG_FORMAT_INDEX: u8 = 1;
#[cfg(target_arch = "wasm32")]
const MJPEG_FRAME_INDEX: u8 = 5;
#[cfg(target_arch = "wasm32")]
const MJPEG_FRAME_INTERVAL_30_FPS: u32 = 333_333;
#[cfg(target_arch = "wasm32")]
const MJPEG_MAX_FRAME_SIZE: u32 = 614_400;
#[cfg(target_arch = "wasm32")]
const ISO_PACKET_BYTES: u32 = 3_072;
#[cfg(target_arch = "wasm32")]
const ISO_TRANSFER_TIMEOUT_MS: u32 = 1_000;
#[cfg(target_arch = "wasm32")]
const CONTROL_TIMEOUT_MS: u32 = 1_000;
#[cfg(target_arch = "wasm32")]
const MAX_PACKETS_PER_CAPTURE: usize = 4_096;
#[cfg(target_arch = "wasm32")]
const UVC_SET_CUR: u8 = 0x01;
#[cfg(target_arch = "wasm32")]
const UVC_GET_CUR: u8 = 0x81;
#[cfg(target_arch = "wasm32")]
const UVC_VS_PROBE_CONTROL: u8 = 0x01;
#[cfg(target_arch = "wasm32")]
const UVC_VS_COMMIT_CONTROL: u8 = 0x02;
#[cfg(target_arch = "wasm32")]
const UVC_STREAM_CONTROL_LEN: usize = 26;
#[cfg(target_arch = "wasm32")]
const UVC_HEADER_FID: u8 = 0x01;
#[cfg(target_arch = "wasm32")]
const UVC_HEADER_EOF: u8 = 0x02;
#[cfg(target_arch = "wasm32")]
const UVC_HEADER_ERR: u8 = 0x40;
#[cfg(target_arch = "wasm32")]
const INITIAL_RETRY_INTERVAL_SECS: u64 = 5;
#[cfg(target_arch = "wasm32")]
const REFRESH_INTERVAL_SECS: u64 = 30;

#[cfg(any(target_arch = "wasm32", test))]
const PALETTE: [[u8; 3]; 12] = [
    [0xe6, 0x39, 0x46],
    [0xf4, 0xa2, 0x61],
    [0xe9, 0xc4, 0x6a],
    [0x2a, 0x9d, 0x8f],
    [0x26, 0x7d, 0xa8],
    [0x8d, 0x99, 0xae],
    [0xff, 0x00, 0x7f],
    [0x00, 0xb4, 0xd8],
    [0x90, 0xbe, 0x6d],
    [0x43, 0xaa, 0x8b],
    [0x57, 0x7d, 0x86],
    [0xff, 0x59, 0x59],
];

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Debug, Clone, Deserialize)]
struct ModelConfig {
    labels_path: String,
    input: InputConfig,
    heads: Vec<HeadConfig>,
    thresholds: ThresholdConfig,
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Debug, Clone, Deserialize)]
struct InputConfig {
    name: String,
    width: u32,
    height: u32,
    #[serde(default)]
    channel_order: ChannelOrder,
    #[serde(default)]
    dtype: TensorElementType,
    qscale: f32,
    zero_point: i32,
    #[serde(default = "default_normalize_divisor")]
    normalize_divisor: f32,
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum ChannelOrder {
    #[default]
    Rgb,
    Bgr,
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Debug, Clone, Deserialize)]
struct HeadConfig {
    stride: u32,
    anchors: Vec<[f32; 2]>,
    box_tensor: QuantizedTensorConfig,
    objectness_tensor: QuantizedTensorConfig,
    classes_tensor: QuantizedTensorConfig,
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Debug, Clone, Deserialize)]
struct QuantizedTensorConfig {
    name: String,
    dimensions: Vec<u32>,
    #[serde(default)]
    dtype: TensorElementType,
    qscale: f32,
    zero_point: i32,
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum TensorElementType {
    #[default]
    I8,
    U8,
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Debug, Clone, Deserialize)]
struct ThresholdConfig {
    score: f32,
    nms_iou: f32,
}

#[cfg(any(target_arch = "wasm32", test))]
fn default_normalize_divisor() -> f32 {
    255.0
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct RgbImage {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Debug, Clone, Copy, PartialEq)]
struct LetterboxTransform {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Debug, Clone, PartialEq)]
struct Detection {
    class_id: usize,
    score: f32,
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

#[cfg(target_arch = "wasm32")]
#[derive(Debug, Clone, Copy)]
struct StreamControl {
    hint: u16,
    format_index: u8,
    frame_index: u8,
    frame_interval: u32,
    key_frame_rate: u16,
    p_frame_rate: u16,
    compression_quality: u16,
    compression_window_size: u16,
    delay: u16,
    max_video_frame_size: u32,
    max_payload_transfer_size: u32,
}

#[cfg(target_arch = "wasm32")]
struct Detector {
    graph: Graph,
    config: ModelConfig,
    labels: Vec<String>,
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        if let Err(err) = run() {
            eprintln!("milkv-led: fatal error: {err}");
            std::process::exit(1);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn run() -> Result<(), String> {
    use std::{thread, time::Duration};

    let detector = Detector::load()?;
    println!(
        "milkv-led: starting USB camera detection service output={OUTPUT_PATH} labels={}",
        detector.labels.len()
    );
    println!(
        "milkv-led: target path={TARGET_DEVICE_PATH} vid={TARGET_VENDOR_ID:04x} pid={TARGET_PRODUCT_ID:04x} mode=640x480 mjpeg"
    );

    let mut captured_once = false;
    loop {
        match capture_once(&detector) {
            Ok((output_len, detections)) => {
                captured_once = true;
                println!(
                    "milkv-led: saved annotated frame bytes={output_len} detections={detections} output={OUTPUT_PATH}"
                );
            }
            Err(err) => {
                eprintln!("milkv-led: capture retry cause={err}");
            }
        }

        thread::sleep(Duration::from_secs(if captured_once {
            REFRESH_INTERVAL_SECS
        } else {
            INITIAL_RETRY_INTERVAL_SECS
        }));
    }
}

#[cfg(target_arch = "wasm32")]
impl Detector {
    fn load() -> Result<Self, String> {
        let config_text = std::fs::read_to_string(MODEL_CONFIG_PATH)
            .map_err(|err| format!("failed to read {MODEL_CONFIG_PATH}: {err}"))?;
        let config = parse_model_config(&config_text)?;
        let labels = load_labels(&config.labels_path)?;
        validate_model_config(&config, &labels)?;
        let model_bytes = std::fs::read(MODEL_PATH)
            .map_err(|err| format!("failed to read {MODEL_PATH}: {err}"))?;
        let graph = graph::load(
            &[model_bytes],
            GraphEncoding::Autodetect,
            ExecutionTarget::Tpu,
        )
        .map_err(format_wasi_nn_error)?;

        println!(
            "milkv-led: model loaded path={MODEL_PATH} input={}x{} heads={}",
            config.input.width,
            config.input.height,
            config.heads.len()
        );

        Ok(Self {
            graph,
            config,
            labels,
        })
    }
}

#[cfg(target_arch = "wasm32")]
impl StreamControl {
    fn default_mjpeg_640x480() -> Self {
        Self {
            hint: 0,
            format_index: MJPEG_FORMAT_INDEX,
            frame_index: MJPEG_FRAME_INDEX,
            frame_interval: MJPEG_FRAME_INTERVAL_30_FPS,
            key_frame_rate: 0,
            p_frame_rate: 0,
            compression_quality: 0,
            compression_window_size: 0,
            delay: 0,
            max_video_frame_size: 0,
            max_payload_transfer_size: 0,
        }
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != UVC_STREAM_CONTROL_LEN {
            return Err(format!(
                "unexpected UVC stream control length: expected {UVC_STREAM_CONTROL_LEN}, got {}",
                bytes.len()
            ));
        }
        Ok(Self {
            hint: le_u16(&bytes[0..2]),
            format_index: bytes[2],
            frame_index: bytes[3],
            frame_interval: le_u32(&bytes[4..8]),
            key_frame_rate: le_u16(&bytes[8..10]),
            p_frame_rate: le_u16(&bytes[10..12]),
            compression_quality: le_u16(&bytes[12..14]),
            compression_window_size: le_u16(&bytes[14..16]),
            delay: le_u16(&bytes[16..18]),
            max_video_frame_size: le_u32(&bytes[18..22]),
            max_payload_transfer_size: le_u32(&bytes[22..26]),
        })
    }

    fn to_bytes(self) -> [u8; UVC_STREAM_CONTROL_LEN] {
        let mut out = [0u8; UVC_STREAM_CONTROL_LEN];
        out[0..2].copy_from_slice(&self.hint.to_le_bytes());
        out[2] = self.format_index;
        out[3] = self.frame_index;
        out[4..8].copy_from_slice(&self.frame_interval.to_le_bytes());
        out[8..10].copy_from_slice(&self.key_frame_rate.to_le_bytes());
        out[10..12].copy_from_slice(&self.p_frame_rate.to_le_bytes());
        out[12..14].copy_from_slice(&self.compression_quality.to_le_bytes());
        out[14..16].copy_from_slice(&self.compression_window_size.to_le_bytes());
        out[16..18].copy_from_slice(&self.delay.to_le_bytes());
        out[18..22].copy_from_slice(&self.max_video_frame_size.to_le_bytes());
        out[22..26].copy_from_slice(&self.max_payload_transfer_size.to_le_bytes());
        out
    }
}

#[cfg(target_arch = "wasm32")]
fn capture_once(detector: &Detector) -> Result<(usize, usize), String> {
    let devices = provider::list_openable_devices().map_err(|err| {
        format!(
            "provider.list_openable_devices failed: {}",
            describe_usb_error(err)
        )
    })?;
    let openable = devices
        .into_iter()
        .find(|device| device.path == TARGET_DEVICE_PATH)
        .ok_or_else(|| format!("target camera not present in allowlist: {TARGET_DEVICE_PATH}"))?;
    println!(
        "milkv-led: discovered device path={} vid={:04x} pid={:04x}",
        openable.path, openable.vendor_id, openable.product_id
    );

    let device = provider::open_device(TARGET_DEVICE_PATH)
        .map_err(|err| format!("provider.open_device failed: {}", describe_usb_error(err)))?;
    validate_target_identity(&device)?;
    ensure_active_configuration(&device)?;
    ensure_stream_interface_descriptor(&device)?;

    let stream = device
        .claim_interface(STREAM_INTERFACE_NUMBER)
        .map_err(|err| format!("claim stream interface failed: {}", describe_usb_error(err)))?;

    let capture_result = (|| {
        stream
            .set_alternate_setting(STREAM_ALT_SETTING_IDLE)
            .map_err(|err| format!("set stream alt=0 failed: {}", describe_usb_error(err)))?;

        let negotiated = negotiate_stream(&stream)?;
        println!(
            "milkv-led: negotiated format={} frame={} interval={} payload={} frame_size={}",
            negotiated.format_index,
            negotiated.frame_index,
            negotiated.frame_interval,
            negotiated.max_payload_transfer_size,
            negotiated.max_video_frame_size
        );

        stream
            .set_alternate_setting(STREAM_ALT_SETTING_ACTIVE)
            .map_err(|err| format!("set stream alt=1 failed: {}", describe_usb_error(err)))?;

        let frame = capture_mjpeg_frame(&stream, negotiated.max_payload_transfer_size)?;
        let (annotated, detections) = annotate_frame(detector, &frame)?;
        persist_frame(&annotated)?;
        Ok::<(usize, usize), String>((annotated.len(), detections))
    })();

    let _ = stream.set_alternate_setting(STREAM_ALT_SETTING_IDLE);
    drop(stream);
    let _ = device.release_interface(STREAM_INTERFACE_NUMBER);
    capture_result
}

#[cfg(target_arch = "wasm32")]
fn annotate_frame(detector: &Detector, frame: &[u8]) -> Result<(Vec<u8>, usize), String> {
    let image = decode_jpeg(frame)?;
    let (letterboxed, transform) = letterbox_resize(
        &image,
        detector.config.input.width,
        detector.config.input.height,
    )?;
    let input_bytes = image_to_nchw_bytes(&letterboxed, &detector.config.input);
    let input_dimensions = [
        1,
        3,
        detector.config.input.height,
        detector.config.input.width,
    ];
    let input = Tensor::new(&input_dimensions, TensorType::U8, &input_bytes);
    let context = detector
        .graph
        .init_execution_context()
        .map_err(format_wasi_nn_error)?;
    let outputs = context
        .compute(vec![(detector.config.input.name.clone(), input)])
        .map_err(format_wasi_nn_error)?;

    let available_output_names = outputs
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let detections = decode_outputs(
        &detector.config,
        &detector.labels,
        outputs,
        &available_output_names,
        transform,
        image.width,
        image.height,
    )?;

    println!(
        "milkv-led: detections={} outputs={available_output_names:?}",
        detections.len()
    );

    let mut annotated = image.clone();
    draw_detections(&mut annotated, &detections, &detector.labels);
    let encoded = encode_jpeg(&annotated)?;
    Ok((encoded, detections.len()))
}

#[cfg(target_arch = "wasm32")]
fn decode_outputs(
    config: &ModelConfig,
    labels: &[String],
    outputs: Vec<(String, Tensor)>,
    available_output_names: &[String],
    transform: LetterboxTransform,
    original_width: u32,
    original_height: u32,
) -> Result<Vec<Detection>, String> {
    let mut detections = Vec::new();

    for head in &config.heads {
        let box_tensor =
            find_named_tensor(&outputs, &head.box_tensor.name, available_output_names)?;
        let objectness_tensor = find_named_tensor(
            &outputs,
            &head.objectness_tensor.name,
            available_output_names,
        )?;
        let classes_tensor =
            find_named_tensor(&outputs, &head.classes_tensor.name, available_output_names)?;

        validate_runtime_tensor(&head.box_tensor, box_tensor)?;
        validate_runtime_tensor(&head.objectness_tensor, objectness_tensor)?;
        validate_runtime_tensor(&head.classes_tensor, classes_tensor)?;

        detections.extend(decode_quantized_yolov5_head(
            &head.box_tensor,
            &box_tensor.data(),
            &head.objectness_tensor,
            &objectness_tensor.data(),
            &head.classes_tensor,
            &classes_tensor.data(),
            labels.len(),
            head,
            config.thresholds.score,
            transform,
            original_width,
            original_height,
        )?);
    }

    Ok(apply_classwise_nms(
        detections,
        config.thresholds.nms_iou,
        original_width,
        original_height,
    ))
}

#[cfg(target_arch = "wasm32")]
fn find_named_tensor<'a>(
    outputs: &'a [(String, Tensor)],
    expected_name: &str,
    available_output_names: &[String],
) -> Result<&'a Tensor, String> {
    let (_, tensor) = outputs
        .iter()
        .find(|(name, _)| name == expected_name)
        .ok_or_else(|| {
            format!(
                "missing `{expected_name}` output; available outputs: {available_output_names:?}"
            )
        })?;
    Ok(tensor)
}

#[cfg(target_arch = "wasm32")]
fn validate_runtime_tensor(
    expected: &QuantizedTensorConfig,
    tensor: &Tensor,
) -> Result<(), String> {
    if tensor.ty() != TensorType::U8 {
        return Err(format!(
            "unexpected output tensor type for {}: {:?}",
            expected.name,
            tensor.ty()
        ));
    }

    let actual_dimensions = tensor.dimensions();
    if actual_dimensions != expected.dimensions {
        return Err(format!(
            "output {} dimensions mismatch: expected {:?}, got {:?}",
            expected.name, expected.dimensions, actual_dimensions
        ));
    }

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn validate_target_identity(device: &Device) -> Result<(), String> {
    let descriptor = device
        .device_descriptor()
        .map_err(|err| format!("device_descriptor failed: {}", describe_usb_error(err)))?;
    if descriptor.vendor_id != TARGET_VENDOR_ID || descriptor.product_id != TARGET_PRODUCT_ID {
        return Err(format!(
            "unexpected USB device identity: expected {:04x}:{:04x}, got {:04x}:{:04x}",
            TARGET_VENDOR_ID, TARGET_PRODUCT_ID, descriptor.vendor_id, descriptor.product_id
        ));
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn ensure_active_configuration(device: &Device) -> Result<(), String> {
    let active = device
        .active_configuration()
        .map_err(|err| format!("active_configuration failed: {}", describe_usb_error(err)))?;
    if active == 1 {
        return Ok(());
    }
    device.select_configuration(1).map_err(|err| {
        format!(
            "select_configuration(1) failed: {}",
            describe_usb_error(err)
        )
    })
}

#[cfg(target_arch = "wasm32")]
fn ensure_stream_interface_descriptor(device: &Device) -> Result<(), String> {
    let configs = device
        .configurations()
        .map_err(|err| format!("configurations failed: {}", describe_usb_error(err)))?;
    let found_stream_interface = configs.iter().any(|config| {
        config.number == 1
            && config.interfaces.iter().any(|interface| {
                interface.class_code == 0x0e
                    && interface.number == STREAM_INTERFACE_NUMBER
                    && interface.alternate_setting == STREAM_ALT_SETTING_ACTIVE
                    && interface.endpoint_descriptors.iter().any(|endpoint| {
                        endpoint.address == STREAM_ENDPOINT_ADDRESS
                            && endpoint.transfer_type == TransferType::Isochronous
                    })
            })
    });

    if !found_stream_interface {
        return Err(format!(
            "camera stream interface {} alt {} endpoint 0x{STREAM_ENDPOINT_ADDRESS:02x} not found in descriptors",
            STREAM_INTERFACE_NUMBER, STREAM_ALT_SETTING_ACTIVE
        ));
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn negotiate_stream(stream: &ClaimedInterface) -> Result<StreamControl, String> {
    let mut probe = get_stream_control(stream, UVC_VS_PROBE_CONTROL)
        .unwrap_or_else(|_| StreamControl::default_mjpeg_640x480());
    probe.hint = 0;
    probe.format_index = MJPEG_FORMAT_INDEX;
    probe.frame_index = MJPEG_FRAME_INDEX;
    probe.frame_interval = MJPEG_FRAME_INTERVAL_30_FPS;
    if probe.max_video_frame_size == 0 {
        probe.max_video_frame_size = MJPEG_MAX_FRAME_SIZE;
    }

    set_stream_control(stream, UVC_VS_PROBE_CONTROL, probe)?;
    let negotiated = get_stream_control(stream, UVC_VS_PROBE_CONTROL)?;
    set_stream_control(stream, UVC_VS_COMMIT_CONTROL, negotiated)?;
    Ok(negotiated)
}

#[cfg(target_arch = "wasm32")]
fn get_stream_control(control: &ClaimedInterface, selector: u8) -> Result<StreamControl, String> {
    let setup = stream_control_setup(UVC_GET_CUR, selector);
    let payload = control
        .control_in(setup, UVC_STREAM_CONTROL_LEN as u32, CONTROL_TIMEOUT_MS)
        .map_err(|err| {
            format!(
                "control-in selector=0x{selector:02x} failed: {}",
                describe_usb_error(err)
            )
        })?;
    StreamControl::from_bytes(&payload)
}

#[cfg(target_arch = "wasm32")]
fn set_stream_control(
    control: &ClaimedInterface,
    selector: u8,
    value: StreamControl,
) -> Result<(), String> {
    let setup = stream_control_setup(UVC_SET_CUR, selector);
    control
        .control_out(setup, &value.to_bytes(), CONTROL_TIMEOUT_MS)
        .map_err(|err| {
            format!(
                "control-out selector=0x{selector:02x} failed: {}",
                describe_usb_error(err)
            )
        })
}

#[cfg(target_arch = "wasm32")]
fn stream_control_setup(request: u8, selector: u8) -> ControlSetup {
    ControlSetup {
        control_type: ControlType::Class,
        recipient: Recipient::InterfaceTarget,
        request,
        value: u16::from(selector) << 8,
        index: u16::from(STREAM_INTERFACE_NUMBER),
    }
}

#[cfg(target_arch = "wasm32")]
fn capture_mjpeg_frame(
    stream: &ClaimedInterface,
    negotiated_payload_bytes: u32,
) -> Result<Vec<u8>, String> {
    let packet_bytes = negotiated_payload_bytes.clamp(1, ISO_PACKET_BYTES);
    let mut frame = Vec::with_capacity(MJPEG_MAX_FRAME_SIZE as usize);
    let mut current_fid = None;

    for packet_index in 0..MAX_PACKETS_PER_CAPTURE {
        let packet = stream
            .isochronous_in(
                STREAM_ENDPOINT_ADDRESS,
                packet_bytes,
                1,
                ISO_TRANSFER_TIMEOUT_MS,
            )
            .map_err(|err| {
                format!(
                    "isochronous_in packet={} failed: {}",
                    packet_index,
                    describe_usb_error(err)
                )
            })?;

        if packet.len() < 2 {
            continue;
        }

        let header_len = packet[0] as usize;
        if header_len < 2 || header_len > packet.len() {
            continue;
        }

        let header = packet[1];
        if header & UVC_HEADER_ERR != 0 {
            frame.clear();
            current_fid = None;
            continue;
        }

        let fid = header & UVC_HEADER_FID;
        let eof = header & UVC_HEADER_EOF != 0;
        let payload = &packet[header_len..];

        if current_fid != Some(fid) {
            if let Some((start, end)) = jpeg_bounds(&frame) {
                return Ok(frame[start..end].to_vec());
            }
            frame.clear();
            current_fid = Some(fid);
        }

        if frame.is_empty() {
            if let Some(start) = find_marker(payload, &[0xff, 0xd8]) {
                frame.extend_from_slice(&payload[start..]);
            }
        } else {
            frame.extend_from_slice(payload);
        }

        if frame.len() > usize::try_from(MJPEG_MAX_FRAME_SIZE).unwrap_or(usize::MAX) {
            return Err(format!(
                "captured frame exceeded expected maximum size: {}",
                frame.len()
            ));
        }

        if let Some((start, end)) = jpeg_bounds(&frame) {
            return Ok(frame[start..end].to_vec());
        }

        if eof {
            frame.clear();
            current_fid = None;
        }
    }

    Err(format!(
        "timed out before a complete JPEG frame was reconstructed after {MAX_PACKETS_PER_CAPTURE} packets"
    ))
}

#[cfg(any(target_arch = "wasm32", test))]
fn parse_model_config(text: &str) -> Result<ModelConfig, String> {
    toml::from_str(text).map_err(|err| format!("failed to parse {MODEL_CONFIG_PATH}: {err}"))
}

#[cfg(any(target_arch = "wasm32", test))]
fn validate_model_config(config: &ModelConfig, labels: &[String]) -> Result<(), String> {
    if config.input.width == 0 || config.input.height == 0 {
        return Err("input width/height must be positive".to_string());
    }
    if config.input.qscale <= 0.0 {
        return Err("input qscale must be positive".to_string());
    }
    if config.input.normalize_divisor <= 0.0 {
        return Err("input normalize_divisor must be positive".to_string());
    }
    if labels.is_empty() {
        return Err("labels file must contain at least one class".to_string());
    }
    if !(0.0..=1.0).contains(&config.thresholds.score) {
        return Err(format!(
            "thresholds.score must be within [0, 1], got {}",
            config.thresholds.score
        ));
    }
    if !(0.0..=1.0).contains(&config.thresholds.nms_iou) {
        return Err(format!(
            "thresholds.nms_iou must be within [0, 1], got {}",
            config.thresholds.nms_iou
        ));
    }
    if config.heads.is_empty() {
        return Err("yolo.toml must contain at least one [[heads]] entry".to_string());
    }

    for head in &config.heads {
        if head.stride == 0 {
            return Err("head stride must be positive".to_string());
        }
        if head.anchors.is_empty() {
            return Err(format!(
                "head stride={} must list at least one anchor",
                head.stride
            ));
        }
        validate_quantized_tensor(&head.box_tensor)?;
        validate_quantized_tensor(&head.objectness_tensor)?;
        validate_quantized_tensor(&head.classes_tensor)?;
        validate_head_shape(head, labels.len())?;
    }

    Ok(())
}

#[cfg(any(target_arch = "wasm32", test))]
fn validate_quantized_tensor(tensor: &QuantizedTensorConfig) -> Result<(), String> {
    if tensor.qscale <= 0.0 {
        return Err(format!(
            "tensor {} qscale must be positive; refresh assets/models/yolo.toml from runtime metadata",
            tensor.name
        ));
    }
    Ok(())
}

#[cfg(any(target_arch = "wasm32", test))]
fn validate_head_shape(head: &HeadConfig, class_count: usize) -> Result<(), String> {
    let box_shape = split_tensor_shape(&head.box_tensor)?;
    let objectness_shape = split_tensor_shape(&head.objectness_tensor)?;
    let classes_shape = split_tensor_shape(&head.classes_tensor)?;

    if box_shape.anchor_count != head.anchors.len() {
        return Err(format!(
            "head stride={} box tensor anchor mismatch: expected {}, got {}",
            head.stride,
            head.anchors.len(),
            box_shape.anchor_count
        ));
    }
    if box_shape.values != 4 {
        return Err(format!(
            "head stride={} box tensor last axis must be 4, got {}",
            head.stride, box_shape.values
        ));
    }
    if objectness_shape.anchor_count != box_shape.anchor_count
        || objectness_shape.grid_height != box_shape.grid_height
        || objectness_shape.grid_width != box_shape.grid_width
    {
        return Err(format!(
            "head stride={} objectness tensor shape {:?} does not match box tensor {:?}",
            head.stride, head.objectness_tensor.dimensions, head.box_tensor.dimensions
        ));
    }
    if objectness_shape.values != 1 {
        return Err(format!(
            "head stride={} objectness tensor last axis must be 1, got {}",
            head.stride, objectness_shape.values
        ));
    }
    if classes_shape.anchor_count != box_shape.anchor_count
        || classes_shape.grid_height != box_shape.grid_height
        || classes_shape.grid_width != box_shape.grid_width
    {
        return Err(format!(
            "head stride={} classes tensor shape {:?} does not match box tensor {:?}",
            head.stride, head.classes_tensor.dimensions, head.box_tensor.dimensions
        ));
    }
    if classes_shape.values != class_count {
        return Err(format!(
            "head stride={} classes tensor last axis mismatch: expected {}, got {}",
            head.stride, class_count, classes_shape.values
        ));
    }

    Ok(())
}

#[cfg(any(target_arch = "wasm32", test))]
fn load_labels(path: &str) -> Result<Vec<String>, String> {
    let labels = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read labels file {path}: {err}"))?
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if labels.is_empty() {
        return Err(format!("labels file {path} contained no labels"));
    }
    Ok(labels)
}

#[cfg(any(target_arch = "wasm32", test))]
fn decode_jpeg(bytes: &[u8]) -> Result<RgbImage, String> {
    use jpeg_decoder::PixelFormat;

    let mut decoder = jpeg_decoder::Decoder::new(bytes);
    let pixels = decoder
        .decode()
        .map_err(|err| format!("failed to decode jpeg: {err}"))?;
    let info = decoder
        .info()
        .ok_or_else(|| "jpeg decoder did not return image info".to_string())?;

    let data = match info.pixel_format {
        PixelFormat::RGB24 => pixels,
        PixelFormat::L8 => pixels
            .into_iter()
            .flat_map(|value| [value, value, value])
            .collect(),
        PixelFormat::CMYK32 => pixels
            .chunks_exact(4)
            .flat_map(|chunk| {
                let c = chunk[0] as f32 / 255.0;
                let m = chunk[1] as f32 / 255.0;
                let y = chunk[2] as f32 / 255.0;
                let k = chunk[3] as f32 / 255.0;
                let r = ((1.0 - c) * (1.0 - k) * 255.0).round() as u8;
                let g = ((1.0 - m) * (1.0 - k) * 255.0).round() as u8;
                let b = ((1.0 - y) * (1.0 - k) * 255.0).round() as u8;
                [r, g, b]
            })
            .collect(),
        other => {
            return Err(format!("unsupported jpeg pixel format: {other:?}"));
        }
    };

    RgbImage::new(u32::from(info.width), u32::from(info.height), data)
}

#[cfg(any(target_arch = "wasm32", test))]
fn encode_jpeg(image: &RgbImage) -> Result<Vec<u8>, String> {
    use jpeg_encoder::{ColorType, Encoder};

    let mut output = Vec::new();
    let encoder = Encoder::new(&mut output, JPEG_QUALITY);
    encoder
        .encode(
            &image.data,
            image.width as u16,
            image.height as u16,
            ColorType::Rgb,
        )
        .map_err(|err| format!("failed to encode jpeg: {err}"))?;
    Ok(output)
}

#[cfg(any(target_arch = "wasm32", test))]
fn letterbox_resize(
    image: &RgbImage,
    target_width: u32,
    target_height: u32,
) -> Result<(RgbImage, LetterboxTransform), String> {
    if image.width == 0 || image.height == 0 {
        return Err("input image must be non-empty".to_string());
    }
    if target_width == 0 || target_height == 0 {
        return Err("target size must be non-empty".to_string());
    }

    let scale = f32::min(
        target_width as f32 / image.width as f32,
        target_height as f32 / image.height as f32,
    );
    let resized_width = (image.width as f32 * scale).round().max(1.0) as u32;
    let resized_height = (image.height as f32 * scale).round().max(1.0) as u32;
    let pad_x = (target_width - resized_width) / 2;
    let pad_y = (target_height - resized_height) / 2;

    let mut output = RgbImage::filled(target_width, target_height, LETTERBOX_FILL);
    let x_ratio = image.width as f32 / resized_width as f32;
    let y_ratio = image.height as f32 / resized_height as f32;

    for dest_y in 0..resized_height {
        let src_y = ((dest_y as f32 + 0.5) * y_ratio - 0.5).clamp(0.0, image.height as f32 - 1.0);
        for dest_x in 0..resized_width {
            let src_x =
                ((dest_x as f32 + 0.5) * x_ratio - 0.5).clamp(0.0, image.width as f32 - 1.0);
            let pixel = sample_bilinear(image, src_x, src_y);
            output.set_pixel(dest_x + pad_x, dest_y + pad_y, pixel);
        }
    }

    Ok((
        output,
        LetterboxTransform {
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
        },
    ))
}

#[cfg(any(target_arch = "wasm32", test))]
fn sample_bilinear(image: &RgbImage, x: f32, y: f32) -> [u8; 3] {
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width.saturating_sub(1));
    let y1 = (y0 + 1).min(image.height.saturating_sub(1));
    let wx = x - x0 as f32;
    let wy = y - y0 as f32;

    let top_left = image.pixel(x0, y0);
    let top_right = image.pixel(x1, y0);
    let bottom_left = image.pixel(x0, y1);
    let bottom_right = image.pixel(x1, y1);

    let mut out = [0u8; 3];
    for index in 0..3 {
        let top = top_left[index] as f32 * (1.0 - wx) + top_right[index] as f32 * wx;
        let bottom = bottom_left[index] as f32 * (1.0 - wx) + bottom_right[index] as f32 * wx;
        out[index] = (top * (1.0 - wy) + bottom * wy).round() as u8;
    }
    out
}

#[cfg(any(target_arch = "wasm32", test))]
fn image_to_nchw_bytes(image: &RgbImage, input: &InputConfig) -> Vec<u8> {
    let pixel_count = (image.width * image.height) as usize;
    let mut output = Vec::with_capacity(pixel_count * 3);
    let channel_indices = match input.channel_order {
        ChannelOrder::Rgb => [0usize, 1, 2],
        ChannelOrder::Bgr => [2usize, 1, 0],
    };

    for channel in channel_indices {
        for pixel_index in 0..pixel_count {
            output.push(quantize_input_value(
                image.data[pixel_index * 3 + channel],
                input,
            ));
        }
    }

    output
}

#[cfg(any(target_arch = "wasm32", test))]
fn quantize_input_value(value: u8, input: &InputConfig) -> u8 {
    let normalized = value as f32 / input.normalize_divisor;
    let quantized = (normalized * input.qscale).round() as i32 + input.zero_point;
    match input.dtype {
        TensorElementType::I8 => {
            (quantized.clamp(i8::MIN as i32, i8::MAX as i32) as i8).to_ne_bytes()[0]
        }
        TensorElementType::U8 => quantized.clamp(u8::MIN as i32, u8::MAX as i32) as u8,
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn decode_quantized_yolov5_head(
    box_config: &QuantizedTensorConfig,
    box_bytes: &[u8],
    objectness_config: &QuantizedTensorConfig,
    objectness_bytes: &[u8],
    classes_config: &QuantizedTensorConfig,
    classes_bytes: &[u8],
    class_count: usize,
    head: &HeadConfig,
    score_threshold: f32,
    transform: LetterboxTransform,
    original_width: u32,
    original_height: u32,
) -> Result<Vec<Detection>, String> {
    validate_head_shape(head, class_count)?;
    let box_shape = split_tensor_shape(box_config)?;
    let objectness_shape = split_tensor_shape(objectness_config)?;
    let classes_shape = split_tensor_shape(classes_config)?;

    let expected_box_len = box_config
        .dimensions
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim as usize))
        .ok_or_else(|| format!("tensor {} dimensions overflowed", box_config.name))?;
    if box_bytes.len() != expected_box_len {
        return Err(format!(
            "tensor {} data length mismatch: expected {}, got {}",
            box_config.name,
            expected_box_len,
            box_bytes.len()
        ));
    }
    let expected_objectness_len = objectness_config
        .dimensions
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim as usize))
        .ok_or_else(|| format!("tensor {} dimensions overflowed", objectness_config.name))?;
    if objectness_bytes.len() != expected_objectness_len {
        return Err(format!(
            "tensor {} data length mismatch: expected {}, got {}",
            objectness_config.name,
            expected_objectness_len,
            objectness_bytes.len()
        ));
    }
    let expected_classes_len = classes_config
        .dimensions
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim as usize))
        .ok_or_else(|| format!("tensor {} dimensions overflowed", classes_config.name))?;
    if classes_bytes.len() != expected_classes_len {
        return Err(format!(
            "tensor {} data length mismatch: expected {}, got {}",
            classes_config.name,
            expected_classes_len,
            classes_bytes.len()
        ));
    }

    let mut detections = Vec::new();
    for anchor_index in 0..box_shape.anchor_count {
        let anchor = head.anchors[anchor_index];
        for grid_y in 0..box_shape.grid_height {
            for grid_x in 0..box_shape.grid_width {
                let tx = dequantize_output(
                    box_bytes[box_shape.index(anchor_index, grid_y, grid_x, 0)],
                    box_config,
                );
                let ty = dequantize_output(
                    box_bytes[box_shape.index(anchor_index, grid_y, grid_x, 1)],
                    box_config,
                );
                let tw = dequantize_output(
                    box_bytes[box_shape.index(anchor_index, grid_y, grid_x, 2)],
                    box_config,
                );
                let th = dequantize_output(
                    box_bytes[box_shape.index(anchor_index, grid_y, grid_x, 3)],
                    box_config,
                );
                let objectness = sigmoid(dequantize_output(
                    objectness_bytes[objectness_shape.index(anchor_index, grid_y, grid_x, 0)],
                    objectness_config,
                ));

                let mut best_class_id = 0usize;
                let mut best_class_prob = 0.0f32;
                for class_id in 0..class_count {
                    let class_prob = sigmoid(dequantize_output(
                        classes_bytes[classes_shape.index(anchor_index, grid_y, grid_x, class_id)],
                        classes_config,
                    ));
                    if class_prob > best_class_prob {
                        best_class_prob = class_prob;
                        best_class_id = class_id;
                    }
                }

                let score = objectness * best_class_prob;
                if score < score_threshold {
                    continue;
                }

                let center_x = (sigmoid(tx) * 2.0 - 0.5 + grid_x as f32) * head.stride as f32;
                let center_y = (sigmoid(ty) * 2.0 - 0.5 + grid_y as f32) * head.stride as f32;
                let width = (sigmoid(tw) * 2.0).powi(2) * anchor[0];
                let height = (sigmoid(th) * 2.0).powi(2) * anchor[1];

                let left = undo_letterbox(
                    center_x - width / 2.0,
                    transform.pad_x,
                    transform.scale,
                    original_width,
                );
                let top = undo_letterbox(
                    center_y - height / 2.0,
                    transform.pad_y,
                    transform.scale,
                    original_height,
                );
                let right = undo_letterbox(
                    center_x + width / 2.0,
                    transform.pad_x,
                    transform.scale,
                    original_width,
                );
                let bottom = undo_letterbox(
                    center_y + height / 2.0,
                    transform.pad_y,
                    transform.scale,
                    original_height,
                );

                if right <= left || bottom <= top {
                    continue;
                }

                detections.push(Detection {
                    class_id: best_class_id,
                    score,
                    left,
                    top,
                    right,
                    bottom,
                });
            }
        }
    }

    Ok(detections)
}

#[cfg(any(target_arch = "wasm32", test))]
struct SplitTensorShape {
    grid_width: usize,
    grid_height: usize,
    anchor_count: usize,
    values: usize,
}

#[cfg(any(target_arch = "wasm32", test))]
impl SplitTensorShape {
    fn index(&self, anchor: usize, grid_y: usize, grid_x: usize, value: usize) -> usize {
        (((anchor * self.grid_height + grid_y) * self.grid_width + grid_x) * self.values) + value
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn split_tensor_shape(tensor: &QuantizedTensorConfig) -> Result<SplitTensorShape, String> {
    match tensor.dimensions.as_slice() {
        [anchors, grid_height, grid_width, values] => Ok(SplitTensorShape {
            grid_width: *grid_width as usize,
            grid_height: *grid_height as usize,
            anchor_count: *anchors as usize,
            values: *values as usize,
        }),
        [1, anchors, grid_height, grid_width, values] => Ok(SplitTensorShape {
            grid_width: *grid_width as usize,
            grid_height: *grid_height as usize,
            anchor_count: *anchors as usize,
            values: *values as usize,
        }),
        _ => Err(format!(
            "tensor {} dimensions {:?} are unsupported; expected [anchors, H, W, C] or [1, anchors, H, W, C]",
            tensor.name, tensor.dimensions
        )),
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn dequantize_output(value: u8, tensor: &QuantizedTensorConfig) -> f32 {
    let quantized = match tensor.dtype {
        TensorElementType::I8 => i8::from_ne_bytes([value]) as i32,
        TensorElementType::U8 => value as i32,
    };
    (quantized - tensor.zero_point) as f32 * tensor.qscale
}

#[cfg(any(target_arch = "wasm32", test))]
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(any(target_arch = "wasm32", test))]
fn undo_letterbox(value: f32, pad: f32, scale: f32, max: u32) -> f32 {
    ((value - pad) / scale).clamp(0.0, max as f32)
}

#[cfg(any(target_arch = "wasm32", test))]
fn apply_classwise_nms(
    mut detections: Vec<Detection>,
    iou_threshold: f32,
    image_width: u32,
    image_height: u32,
) -> Vec<Detection> {
    detections.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });

    let mut selected: Vec<Detection> = Vec::new();
    'candidate: for detection in detections {
        for existing in &selected {
            if existing.class_id == detection.class_id
                && intersection_over_union(existing, &detection) > iou_threshold
            {
                continue 'candidate;
            }
        }

        let mut clipped = detection.clone();
        clipped.left = clipped.left.clamp(0.0, image_width as f32);
        clipped.right = clipped.right.clamp(0.0, image_width as f32);
        clipped.top = clipped.top.clamp(0.0, image_height as f32);
        clipped.bottom = clipped.bottom.clamp(0.0, image_height as f32);
        if clipped.right > clipped.left && clipped.bottom > clipped.top {
            selected.push(clipped);
        }
    }

    selected
}

#[cfg(any(target_arch = "wasm32", test))]
fn intersection_over_union(left: &Detection, right: &Detection) -> f32 {
    let x1 = left.left.max(right.left);
    let y1 = left.top.max(right.top);
    let x2 = left.right.min(right.right);
    let y2 = left.bottom.min(right.bottom);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = (x2 - x1) * (y2 - y1);
    let left_area = (left.right - left.left) * (left.bottom - left.top);
    let right_area = (right.right - right.left) * (right.bottom - right.top);
    intersection / (left_area + right_area - intersection)
}

#[cfg(any(target_arch = "wasm32", test))]
fn draw_detections(image: &mut RgbImage, detections: &[Detection], labels: &[String]) {
    for detection in detections {
        let color = PALETTE[detection.class_id % PALETTE.len()];
        let left = detection.left.floor().max(0.0) as u32;
        let top = detection.top.floor().max(0.0) as u32;
        let right = detection.right.ceil().max(0.0) as u32;
        let bottom = detection.bottom.ceil().max(0.0) as u32;
        image.draw_rect(left, top, right, bottom, color, BOX_THICKNESS);
        let label = labels
            .get(detection.class_id)
            .map(String::as_str)
            .unwrap_or("unknown");
        let text = format!("{label} {:.2}", detection.score);
        let text_top = top.saturating_sub(10);
        image.draw_label(left, text_top, &text, color);
    }
}

#[cfg(any(target_arch = "wasm32", test))]
impl RgbImage {
    fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self, String> {
        let expected = width as usize * height as usize * 3;
        if data.len() != expected {
            return Err(format!(
                "rgb image data length mismatch: expected {expected}, got {}",
                data.len()
            ));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    fn filled(width: u32, height: u32, value: u8) -> Self {
        Self {
            width,
            height,
            data: vec![value; width as usize * height as usize * 3],
        }
    }

    fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        let index = ((y * self.width + x) * 3) as usize;
        [self.data[index], self.data[index + 1], self.data[index + 2]]
    }

    fn set_pixel(&mut self, x: u32, y: u32, color: [u8; 3]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let index = ((y * self.width + x) * 3) as usize;
        self.data[index] = color[0];
        self.data[index + 1] = color[1];
        self.data[index + 2] = color[2];
    }

    fn fill_rect(&mut self, left: u32, top: u32, width: u32, height: u32, color: [u8; 3]) {
        let right = left.saturating_add(width).min(self.width);
        let bottom = top.saturating_add(height).min(self.height);
        for y in top..bottom {
            for x in left..right {
                self.set_pixel(x, y, color);
            }
        }
    }

    fn draw_rect(
        &mut self,
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
        color: [u8; 3],
        thickness: u32,
    ) {
        if right <= left || bottom <= top {
            return;
        }

        for offset in 0..thickness {
            let x0 = left.saturating_add(offset);
            let y0 = top.saturating_add(offset);
            let x1 = right.saturating_sub(offset + 1);
            let y1 = bottom.saturating_sub(offset + 1);
            if x1 <= x0 || y1 <= y0 {
                break;
            }
            for x in x0..=x1 {
                self.set_pixel(x, y0, color);
                self.set_pixel(x, y1, color);
            }
            for y in y0..=y1 {
                self.set_pixel(x0, y, color);
                self.set_pixel(x1, y, color);
            }
        }
    }

    fn draw_label(&mut self, left: u32, top: u32, text: &str, color: [u8; 3]) {
        let width = (text.len() as u32 * 8).min(self.width.saturating_sub(left));
        let height = 9;
        self.fill_rect(left, top, width, height, [0, 0, 0]);
        for (index, ch) in text.chars().enumerate() {
            let x = left + (index as u32 * 8);
            if x + 8 > self.width {
                break;
            }
            draw_glyph(self, x, top, ch, color);
        }
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn draw_glyph(image: &mut RgbImage, left: u32, top: u32, ch: char, color: [u8; 3]) {
    let glyph = BASIC_FONTS
        .get(ch)
        .or_else(|| BASIC_FONTS.get('?'))
        .expect("font8x8 should contain ASCII glyphs");
    for (row_index, row_bits) in glyph.iter().copied().enumerate() {
        for column_index in 0..8 {
            if row_bits & (1 << column_index) != 0 {
                image.set_pixel(left + column_index as u32, top + row_index as u32, color);
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn persist_frame(frame: &[u8]) -> Result<(), String> {
    std::fs::write(OUTPUT_TMP_PATH, frame)
        .map_err(|err| format!("failed to write {OUTPUT_TMP_PATH}: {err}"))?;
    std::fs::rename(OUTPUT_TMP_PATH, OUTPUT_PATH)
        .map_err(|err| format!("failed to rename {OUTPUT_TMP_PATH} -> {OUTPUT_PATH}: {err}"))?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn jpeg_bounds(payload: &[u8]) -> Option<(usize, usize)> {
    let start = find_marker(payload, &[0xff, 0xd8])?;
    let end = find_marker(&payload[start..], &[0xff, 0xd9])?;
    Some((start, start + end + 2))
}

#[cfg(target_arch = "wasm32")]
fn find_marker(haystack: &[u8], needle: &[u8; 2]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(target_arch = "wasm32")]
fn le_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

#[cfg(target_arch = "wasm32")]
fn le_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(target_arch = "wasm32")]
fn describe_usb_error(err: UsbError) -> String {
    match err {
        UsbError::NotAllowed => "not-allowed".to_string(),
        UsbError::Timeout => "timeout".to_string(),
        UsbError::Disconnected => "disconnected".to_string(),
        UsbError::Busy => "busy".to_string(),
        UsbError::InvalidArgument => "invalid-argument".to_string(),
        UsbError::TransferFault => "transfer-fault".to_string(),
        UsbError::OperationNotSupported => "operation-not-supported".to_string(),
        UsbError::Other(detail) => format!("other({detail})"),
    }
}

#[cfg(target_arch = "wasm32")]
fn format_wasi_nn_error(error: WasiNnError) -> String {
    format!("{:?}: {}", error.code(), error.data())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tensor_config(name: &str, dims: Vec<u32>, qscale: f32) -> QuantizedTensorConfig {
        QuantizedTensorConfig {
            name: name.to_string(),
            dimensions: dims,
            dtype: TensorElementType::I8,
            qscale,
            zero_point: 0,
        }
    }

    fn sample_head_config() -> HeadConfig {
        HeadConfig {
            stride: 8,
            anchors: vec![[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]],
            box_tensor: sample_tensor_config("boxes", vec![3, 1, 1, 4], 1.0),
            objectness_tensor: sample_tensor_config("objectness", vec![3, 1, 1, 1], 1.0),
            classes_tensor: sample_tensor_config("classes", vec![3, 1, 1, 1], 1.0),
        }
    }

    #[test]
    fn parse_model_config_matches_downloaded_model() {
        let config = parse_model_config(include_str!("../assets/models/yolo.toml"))
            .expect("yolo.toml should parse");
        let labels = load_labels("assets/models/coco.names").expect("coco names should load");
        validate_model_config(&config, &labels).expect("downloaded model config should validate");
        assert_eq!(config.input.name, "images");
        assert_eq!(config.heads.len(), 3);
        assert_eq!(config.heads[0].box_tensor.name, "output0_Gather__reshape");
    }

    #[test]
    fn letterbox_resize_tracks_padding_and_inverse_mapping() {
        let image = RgbImage::filled(640, 480, 0);
        let (_, transform) = letterbox_resize(&image, 640, 640).expect("resize should succeed");
        assert_eq!(
            transform,
            LetterboxTransform {
                scale: 1.0,
                pad_x: 0.0,
                pad_y: 80.0,
            }
        );
        assert_eq!(
            undo_letterbox(96.0, transform.pad_y, transform.scale, 480),
            16.0
        );
        assert_eq!(
            undo_letterbox(176.0, transform.pad_y, transform.scale, 480),
            96.0
        );
    }

    #[test]
    fn dequantize_output_uses_qscale_and_zero_point() {
        let tensor = QuantizedTensorConfig {
            name: "tensor".to_string(),
            dimensions: vec![1],
            dtype: TensorElementType::I8,
            qscale: 0.5,
            zero_point: 0,
        };
        let value = dequantize_output((12i8).to_ne_bytes()[0], &tensor);
        assert_eq!(value, 6.0);
    }

    #[test]
    fn decode_quantized_yolov5_head_decodes_single_detection() {
        let head = sample_head_config();
        let mut box_bytes = vec![(-10i8).to_ne_bytes()[0]; 12];
        let mut objectness_bytes = vec![(-10i8).to_ne_bytes()[0]; 3];
        let mut class_bytes = vec![(-10i8).to_ne_bytes()[0]; 3];
        box_bytes[0] = 0;
        box_bytes[1] = 0;
        box_bytes[2] = 0;
        box_bytes[3] = 0;
        objectness_bytes[0] = 8;
        class_bytes[0] = 8;
        let detections = decode_quantized_yolov5_head(
            &head.box_tensor,
            &box_bytes,
            &head.objectness_tensor,
            &objectness_bytes,
            &head.classes_tensor,
            &class_bytes,
            1,
            &head,
            0.25,
            LetterboxTransform {
                scale: 1.0,
                pad_x: 0.0,
                pad_y: 0.0,
            },
            640,
            640,
        )
        .expect("decode should succeed");
        assert_eq!(detections.len(), 1);
        let detection = &detections[0];
        assert_eq!(detection.class_id, 0);
        assert!(detection.score > 0.99);
        assert!(detection.right > detection.left);
        assert!(detection.bottom > detection.top);
    }

    #[test]
    fn apply_classwise_nms_keeps_highest_score_per_overlap() {
        let detections = vec![
            Detection {
                class_id: 0,
                score: 0.95,
                left: 10.0,
                top: 10.0,
                right: 110.0,
                bottom: 110.0,
            },
            Detection {
                class_id: 0,
                score: 0.80,
                left: 15.0,
                top: 12.0,
                right: 108.0,
                bottom: 109.0,
            },
            Detection {
                class_id: 1,
                score: 0.70,
                left: 15.0,
                top: 12.0,
                right: 108.0,
                bottom: 109.0,
            },
        ];

        let detections = apply_classwise_nms(detections, 0.45, 640, 640);
        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].class_id, 0);
        assert_eq!(detections[1].class_id, 1);
    }

    #[test]
    fn draw_rect_and_label_mutate_pixels() {
        let mut image = RgbImage::filled(32, 32, 0);
        image.draw_rect(4, 4, 20, 20, [255, 0, 0], 2);
        image.draw_label(4, 0, "dog 0.99", [0, 255, 0]);
        assert_eq!(image.pixel(4, 10), [255, 0, 0]);
        let mut any_label_pixel = false;
        for y in 0..9 {
            for x in 4..32 {
                if image.pixel(x, y) != [0, 0, 0] {
                    any_label_pixel = true;
                    break;
                }
            }
            if any_label_pixel {
                break;
            }
        }
        assert!(
            any_label_pixel,
            "label drawing should change at least one pixel"
        );
    }
}
