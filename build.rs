fn main() {
    cxx_build::bridge("src/bridge.rs")
        .file("cpp/matrix.cpp")
        .std("c++23")
        .flag("-O3")
        .compile("cxx-matrix");

    println!("cargo:rerun-if-changed=cpp/matrix.hpp");
    println!("cargo:rerun-if-changed=cpp/matrix.cpp");
}
