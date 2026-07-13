use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const PROCESSOR_SOURCES: &[&str] = &[
    "src/builtin_pb/sentencepiece.pb.cc",
    "src/builtin_pb/sentencepiece_model.pb.cc",
    "src/bpe_model.cc",
    "src/char_model.cc",
    "src/error.cc",
    "src/filesystem.cc",
    "src/model_factory.cc",
    "src/model_interface.cc",
    "src/normalizer.cc",
    "src/sentencepiece_processor.cc",
    "src/unigram_model.cc",
    "src/util.cc",
    "src/word_model.cc",
    "third_party/absl/flags/flag.cc",
];

const PROTOBUF_LITE_SOURCES: &[&str] = &[
    "arena.cc",
    "arenastring.cc",
    "bytestream.cc",
    "coded_stream.cc",
    "common.cc",
    "extension_set.cc",
    "generated_enum_util.cc",
    "generated_message_table_driven_lite.cc",
    "generated_message_util.cc",
    "implicit_weak_message.cc",
    "int128.cc",
    "io_win32.cc",
    "message_lite.cc",
    "parse_context.cc",
    "repeated_field.cc",
    "status.cc",
    "statusor.cc",
    "stringpiece.cc",
    "stringprintf.cc",
    "structurally_valid.cc",
    "strutil.cc",
    "time.cc",
    "wire_format_lite.cc",
    "zero_copy_stream.cc",
    "zero_copy_stream_impl.cc",
    "zero_copy_stream_impl_lite.cc",
];

fn main() {
    pyo3_build_config::add_extension_module_link_args();
    let crate_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let vendor = crate_dir.join("../vendor/sentencepiece");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    write_config_header(&out_dir);

    let mut wrapper = configured_build(&out_dir, &vendor);
    wrapper
        .warnings(true)
        .file(crate_dir.join("cpp/ps_sentencepiece.cc"))
        .compile("process_sanskrit_sentencepiece_wrapper");

    let mut vendor_build = configured_build(&out_dir, &vendor);
    vendor_build.warnings(false);
    for source in PROCESSOR_SOURCES {
        vendor_build.file(vendor.join(source));
    }
    for source in PROTOBUF_LITE_SOURCES {
        vendor_build.file(vendor.join("third_party/protobuf-lite").join(source));
    }
    if !cfg!(target_env = "msvc") {
        vendor_build.flag_if_supported("-Wno-deprecated-declarations");
        vendor_build.flag_if_supported("-Wno-sign-compare");
    }
    vendor_build.compile("process_sanskrit_sentencepiece_vendor");

    println!("cargo:rerun-if-changed=cpp/ps_sentencepiece.cc");
    println!("cargo:rerun-if-changed=cpp/ps_sentencepiece.h");
    println!("cargo:rerun-if-changed={}", vendor.display());
}

fn configured_build(out_dir: &Path, vendor: &Path) -> cc::Build {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include(out_dir)
        .include(vendor)
        .include(vendor.join("src"))
        .include(vendor.join("src/builtin_pb"))
        .include(vendor.join("third_party"))
        .include(vendor.join("third_party/protobuf-lite"))
        .define("_USE_INTERNAL_STRING_VIEW", None)
        .define("HAVE_PTHREAD", "1");
    build
}

fn write_config_header(out_dir: &Path) {
    let config = concat!(
        "#ifndef CONFIG_H_\n",
        "#define CONFIG_H_\n",
        "#define VERSION \"0.2.1\"\n",
        "#define PACKAGE \"sentencepiece\"\n",
        "#define PACKAGE_STRING \"sentencepiece 0.2.1\"\n",
        "#define INSTALL_DATADIR \"\"\n",
        "#endif\n",
    );
    fs::write(out_dir.join("config.h"), config).expect("write SentencePiece config.h");
}
