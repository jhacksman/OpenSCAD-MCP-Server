#!/bin/bash
# Test OpenSCAD primitives with different export formats

PYTHON="python"
OUTPUT_DIR="test_output"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the tests
$PYTHON -m src.testing.test_primitives --output-dir $OUTPUT_DIR --validate

echo "Tests completed. Results are in $OUTPUT_DIR"
