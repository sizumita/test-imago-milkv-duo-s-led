wit_bindgen::generate!({
    path: "wit",
    generate_all
});

use imago::experimental_gpio::{delay, digital};

const PIN_LABEL: &str = "GPI459";
const BLINK_INTERVAL_MS: u32 = 3_000;

fn main() {
    let pin = digital::get_digital_out(PIN_LABEL, &[])
        .unwrap_or_else(|err| panic!("failed to acquire {PIN_LABEL}: {err:?}"));

    pin.set_inactive()
        .unwrap_or_else(|err| panic!("failed to initialize {PIN_LABEL}: {err:?}"));

    loop {
        pin.set_active()
            .unwrap_or_else(|err| panic!("failed to turn on {PIN_LABEL}: {err:?}"));
        delay::delay_ms(BLINK_INTERVAL_MS);

        pin.set_inactive()
            .unwrap_or_else(|err| panic!("failed to turn off {PIN_LABEL}: {err:?}"));
        delay::delay_ms(BLINK_INTERVAL_MS);
    }
}
