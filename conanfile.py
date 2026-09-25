from conan import ConanFile
from conan.tools.files import copy
import os


class TheFBlasConan(ConanFile):
    name = "thefblas"
    version = "0.1.0"
    description = (
        "A minimal, templated, header-only C++17 fixed-point BLAS-style library"
    )
    license = "MIT"
    url = "https://github.com/thekyria/thefblas"
    homepage = "https://github.com/thekyria/thefblas"
    topics = ("blas", "linear-algebra", "fixed-point", "header-only", "cpp17")

    # Header-only: no build settings influence the produced package.
    package_type = "header-library"
    no_copy_source = True

    # Sources are kept in the same place as the recipe
    exports_sources = (
        "include/*",
        "LICENSE",
    )

    def package_id(self):
        # Header-only package: the binary is identical for every configuration.
        self.info.clear()

    def package(self):
        copy(
            self,
            "LICENSE",
            self.source_folder,
            os.path.join(self.package_folder, "licenses"),
        )
        copy(
            self,
            "*",
            os.path.join(self.source_folder, "include"),
            os.path.join(self.package_folder, "include"),
        )

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "thefblas")
        self.cpp_info.set_property("cmake_target_name", "thefblas::thefblas")
        # Header-only: nothing to link, no binaries.
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
