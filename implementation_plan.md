# Implementation Plan: OpenSCAD-MCP-Server with AI-Driven 3D Modeling

## 1. Project Structure Updates

### 1.1 New Modules
```
src/
├── ai/
│   ├── venice_api.py         # Venice.ai API client
│   └── sam_segmentation.py   # SAM2 integration
├── models/
│   └── threestudio_generator.py  # threestudio integration
└── workflow/
    └── image_to_model_pipeline.py  # Workflow orchestration
```

### 1.2 Dependencies
Add to requirements.txt:
```
# Image Generation - Venice.ai API
# (using existing requests and python-dotenv)

# Object Segmentation - SAM2
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
segment-anything>=1.0

# 3D Model Creation - threestudio
ninja>=1.11.0
pytorch3d>=0.7.4
trimesh>=3.21.0
```

## 2. Component Implementation

### 2.1 Venice.ai API Integration
- Create `VeniceImageGenerator` class in `venice_api.py`
- Implement authentication with API key
- Add image generation with Flux model
- Support image downloading and storage

### 2.2 SAM2 Integration
- Create `SAMSegmenter` class in `sam_segmentation.py`
- Implement model loading with PyTorch
- Add object segmentation from images
- Support mask generation and visualization

### 2.3 threestudio Integration
- Create `ThreeStudioGenerator` class in `threestudio_generator.py`
- Implement 3D model generation from masked images
- Support model export in formats compatible with OpenSCAD
- Add preview image generation

### 2.4 OpenSCAD Integration
- Extend `OpenSCADWrapper` with methods to:
  - Import 3D models from threestudio
  - Generate parametric modifications
  - Create multi-angle previews
  - Export in various formats

### 2.5 Workflow Orchestration
- Create `ImageToModelPipeline` class to coordinate the workflow:
  1. Generate image with Venice.ai API
  2. Segment object with SAM2
  3. Create 3D model with threestudio
  4. Import into OpenSCAD for parametric editing

## 3. MCP Tool Integration

Add new MCP tools to main.py:
- `generate_image_from_text`: Generate images using Venice.ai
- `segment_object_from_image`: Segment objects using SAM2
- `generate_3d_model_from_image`: Create 3D models using threestudio
- `generate_model_from_text`: End-to-end pipeline from text to 3D model

## 4. Hardware Requirements

- SAM2: NVIDIA GPU with 6GB+ VRAM
- threestudio: NVIDIA GPU with 6GB+ VRAM
- Consider implementing fallback options for environments with limited GPU resources

## 5. Implementation Phases

### Phase 1: Basic Integration
- Implement Venice.ai API client
- Set up SAM2 with basic segmentation
- Create threestudio wrapper with minimal functionality
- Extend OpenSCAD wrapper for model import

### Phase 2: Workflow Orchestration
- Implement the full pipeline
- Add MCP tools for each component
- Create end-to-end workflow tool

### Phase 3: Optimization and Refinement
- Optimize for performance
- Add error handling and recovery
- Implement corrective cycle for mesh modification
- Add user interface improvements
