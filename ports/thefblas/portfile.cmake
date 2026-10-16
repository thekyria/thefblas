vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  thekyria/thefblas
  REF
  "v${VERSION}"
  SHA512
  b12ba8f4933191f30cf308dff122129ce7169ef3d0a259d48383c205a358e4b1d09253221aa9cf30aed494145dc06a1803d0d90634ee9373cb1be1c169fd76cf
  HEAD_REF
  main)

vcpkg_cmake_configure(
  SOURCE_PATH
  "${SOURCE_PATH}"
  OPTIONS
  -DTHEFBLAS_BUILD_TESTS=OFF
  -DTHEFBLAS_BUILD_EXAMPLES=OFF
  -DTHEFBLAS_ENABLE_STRICT_WARNINGS=OFF
  -DTHEFBLAS_WARNINGS_AS_ERRORS=OFF
  -DTHEFBLAS_ENABLE_SANITIZERS=OFF
  -DTHEFBLAS_ENABLE_COVERAGE=OFF
  -DTHEFBLAS_ENABLE_IPO=OFF
  -DTHEFBLAS_ENABLE_RELEASE_HARDENING=OFF)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME thefblas CONFIG_PATH lib/cmake/thefblas)

# Header-only library: drop the (empty) debug tree and any lib directories.
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib")

# Install license
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")

# Provide a usage message shown after installation
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
