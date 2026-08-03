fn main() {
    if std::env::var("CARGO_FEATURE_PY").is_ok() {
        pyo3_build_config::add_extension_module_link_args();
    }
}
