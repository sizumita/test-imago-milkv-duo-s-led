#[cfg(target_arch = "wasm32")]
wit_bindgen::generate!({
    path: "wit",
    generate_all
});

#[cfg(target_arch = "wasm32")]
mod app {
    use super::imago::usb::{
        device::Device,
        provider,
        types::{ControlSetup, ControlType, Recipient, TransferType, UsbError},
        usb_interface::ClaimedInterface,
    };
    use std::{fs, thread, time::Duration};

    const TARGET_DEVICE_PATH: &str = "/dev/bus/usb/001/002";
    const TARGET_VENDOR_ID: u16 = 0x291a;
    const TARGET_PRODUCT_ID: u16 = 0x3361;
    const STREAM_INTERFACE_NUMBER: u8 = 3;
    const STREAM_ALT_SETTING_IDLE: u8 = 0;
    const STREAM_ALT_SETTING_ACTIVE: u8 = 1;
    const STREAM_ENDPOINT_ADDRESS: u8 = 0x83;
    const MJPEG_FORMAT_INDEX: u8 = 1;
    const MJPEG_FRAME_INDEX: u8 = 5;
    const MJPEG_FRAME_INTERVAL_30_FPS: u32 = 333_333;
    const MJPEG_MAX_FRAME_SIZE: u32 = 614_400;
    const ISO_PACKET_BYTES: u32 = 3_072;
    const ISO_TRANSFER_TIMEOUT_MS: u32 = 1_000;
    const CONTROL_TIMEOUT_MS: u32 = 1_000;
    const MAX_PACKETS_PER_CAPTURE: usize = 4_096;
    const INITIAL_RETRY_INTERVAL: Duration = Duration::from_secs(5);
    const REFRESH_INTERVAL: Duration = Duration::from_secs(30);
    const OUTPUT_PATH: &str = "/captures/last-frame.jpg";
    const OUTPUT_TMP_PATH: &str = "/captures/last-frame.jpg.tmp";
    const UVC_SET_CUR: u8 = 0x01;
    const UVC_GET_CUR: u8 = 0x81;
    const UVC_VS_PROBE_CONTROL: u8 = 0x01;
    const UVC_VS_COMMIT_CONTROL: u8 = 0x02;
    const UVC_STREAM_CONTROL_LEN: usize = 26;
    const UVC_HEADER_FID: u8 = 0x01;
    const UVC_HEADER_EOF: u8 = 0x02;
    const UVC_HEADER_ERR: u8 = 0x40;

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

    pub fn main() {
        run();
    }

    fn run() {
        println!("milkv-led: starting USB camera capture service");
        println!(
            "milkv-led: target path={TARGET_DEVICE_PATH} vid={TARGET_VENDOR_ID:04x} pid={TARGET_PRODUCT_ID:04x} mode=640x480 mjpeg"
        );

        let mut captured_once = false;
        loop {
            match capture_once() {
                Ok(frame_len) => {
                    captured_once = true;
                    println!("milkv-led: capture success bytes={frame_len} output={OUTPUT_PATH}");
                }
                Err(err) => {
                    eprintln!("milkv-led: capture retry cause={err}");
                }
            }

            thread::sleep(if captured_once {
                REFRESH_INTERVAL
            } else {
                INITIAL_RETRY_INTERVAL
            });
        }
    }

    fn capture_once() -> Result<usize, String> {
        let devices = provider::list_openable_devices().map_err(|err| {
            format!(
                "provider.list_openable_devices failed: {}",
                describe_usb_error(err)
            )
        })?;
        let openable = devices
            .into_iter()
            .find(|device| device.path == TARGET_DEVICE_PATH)
            .ok_or_else(|| {
                format!("target camera not present in allowlist: {TARGET_DEVICE_PATH}")
            })?;
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
            persist_frame(&frame)?;
            Ok::<usize, String>(frame.len())
        })();

        let _ = stream.set_alternate_setting(STREAM_ALT_SETTING_IDLE);
        drop(stream);
        let _ = device.release_interface(STREAM_INTERFACE_NUMBER);
        capture_result
    }

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

    fn get_stream_control(
        control: &ClaimedInterface,
        selector: u8,
    ) -> Result<StreamControl, String> {
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

    fn stream_control_setup(request: u8, selector: u8) -> ControlSetup {
        ControlSetup {
            control_type: ControlType::Class,
            recipient: Recipient::InterfaceTarget,
            request,
            value: u16::from(selector) << 8,
            index: u16::from(STREAM_INTERFACE_NUMBER),
        }
    }

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

    fn persist_frame(frame: &[u8]) -> Result<(), String> {
        fs::write(OUTPUT_TMP_PATH, frame)
            .map_err(|err| format!("failed to write {OUTPUT_TMP_PATH}: {err}"))?;
        fs::rename(OUTPUT_TMP_PATH, OUTPUT_PATH)
            .map_err(|err| format!("failed to rename {OUTPUT_TMP_PATH} -> {OUTPUT_PATH}: {err}"))?;
        Ok(())
    }

    fn jpeg_bounds(payload: &[u8]) -> Option<(usize, usize)> {
        let start = find_marker(payload, &[0xff, 0xd8])?;
        let end = find_marker(&payload[start..], &[0xff, 0xd9])?;
        Some((start, start + end + 2))
    }

    fn find_marker(haystack: &[u8], needle: &[u8; 2]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }

    fn le_u16(bytes: &[u8]) -> u16 {
        u16::from_le_bytes([bytes[0], bytes[1]])
    }

    fn le_u32(bytes: &[u8]) -> u32 {
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

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
}

#[cfg(target_arch = "wasm32")]
fn main() {
    app::main();
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {}
