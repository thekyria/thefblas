# Top-level Makefile for common development workflows.

BUILD_DIR ?= build
CONFIG ?= Release
PREFIX ?= $(CURDIR)/install
COVERAGE_DIR ?= build/gcc-debug-coverage
COVERAGE_HTML ?= coverage_html

.PHONY: help configure build test install docs clean rebuild coverage

help:
	@echo "Available targets:"
	@echo "  configure  - Configure CMake in $(BUILD_DIR)"
	@echo "  build      - Build project"
	@echo "  test       - Run tests with CTest"
	@echo "  install    - Install to $(PREFIX)"
	@echo "  docs       - Generate Doxygen docs (requires Doxyfile)"
	@echo "  clean      - Remove $(BUILD_DIR)"
	@echo "  rebuild    - Clean, configure, and build"
	@echo "  coverage   - Build with instrumentation, run tests, generate HTML report in $(COVERAGE_HTML)/"

configure:
	@if [ -f "$(BUILD_DIR)/CMakeCache.txt" ] && ! grep -q "CMAKE_HOME_DIRECTORY:INTERNAL=$(CURDIR)" "$(BUILD_DIR)/CMakeCache.txt"; then \
		echo "Detected stale CMake cache in $(BUILD_DIR); removing it."; \
		rm -rf "$(BUILD_DIR)"; \
	fi
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG)

build: configure
	cmake --build $(BUILD_DIR) --config $(CONFIG) --parallel

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure -C $(CONFIG)

install: build
	cmake --install $(BUILD_DIR) --prefix $(PREFIX) --config $(CONFIG)

docs:
	doxygen Doxyfile

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build

coverage:
	cmake --preset gcc-debug-coverage
	cmake --build --preset gcc-debug-coverage
	lcov --directory $(COVERAGE_DIR) --zerocounters
	ctest --preset gcc-debug-coverage
	lcov --capture --directory $(COVERAGE_DIR) --output-file $(COVERAGE_DIR)/coverage.info \
	     --exclude '/usr/*' --exclude '*/tests/*'
	genhtml $(COVERAGE_DIR)/coverage.info --output-directory $(COVERAGE_HTML)
	@echo ""
	@echo "Coverage report: $(COVERAGE_HTML)/index.html"
