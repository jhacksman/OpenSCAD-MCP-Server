import os
import base64
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

class WebInterface:
    """
    Web interface for displaying model previews and managing 3D models.
    """
    
    def __init__(self, app, static_dir: str, templates_dir: str, output_dir: str):
        """
        Initialize the web interface.
        
        Args:
            app: FastAPI application
            static_dir: Directory for static files
            templates_dir: Directory for templates
            output_dir: Directory containing output files (STL, PNG)
        """
        self.app = app
        self.static_dir = static_dir
        self.templates_dir = templates_dir
        self.output_dir = output_dir
        self.preview_dir = os.path.join(output_dir, "preview")
        self.stl_dir = os.path.join(output_dir, "stl")
        
        # Create directories if they don't exist
        os.makedirs(self.static_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Create router
        self.router = APIRouter(prefix="/ui", tags=["UI"])
        
        # Set up static files
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        
        # Set up templates
        self.templates = Jinja2Templates(directory=templates_dir)
        
        # Register routes
        self._register_routes()
        
        # Create template files
        self._create_template_files()
        
        # Create static files
        self._create_static_files()
    
    def _register_routes(self):
        """Register routes for the web interface."""
        # Home page
        @self.router.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})
        
        # Model preview page
        @self.router.get("/preview/{model_id}", response_class=HTMLResponse)
        async def preview(request: Request, model_id: str):
            # Check if preview exists
            preview_file = os.path.join(self.preview_dir, f"{model_id}.png")
            if not os.path.exists(preview_file):
                raise HTTPException(status_code=404, detail="Preview not found")
            
            # Get multi-angle previews if they exist
            angles = ["front", "top", "right", "perspective"]
            previews = {}
            
            for angle in angles:
                angle_file = os.path.join(self.preview_dir, f"{model_id}_{angle}.png")
                if os.path.exists(angle_file):
                    previews[angle] = f"/api/preview/{model_id}_{angle}"
            
            # If no multi-angle previews, use the main preview
            if not previews:
                previews["main"] = f"/api/preview/{model_id}"
            
            # Check if STL exists
            stl_file = os.path.join(self.stl_dir, f"{model_id}.stl")
            stl_url = f"/api/stl/{model_id}" if os.path.exists(stl_file) else None
            
            return self.templates.TemplateResponse(
                "preview.html", 
                {
                    "request": request, 
                    "model_id": model_id,
                    "previews": previews,
                    "stl_url": stl_url
                }
            )
        
        # List all models
        @self.router.get("/models", response_class=HTMLResponse)
        async def list_models(request: Request):
            # Get all STL files
            stl_files = []
            if os.path.exists(self.stl_dir):
                stl_files = [f for f in os.listdir(self.stl_dir) if f.endswith(".stl")]
            
            # Extract model IDs
            model_ids = [os.path.splitext(f)[0] for f in stl_files]
            
            # Get preview URLs
            models = []
            for model_id in model_ids:
                preview_file = os.path.join(self.preview_dir, f"{model_id}.png")
                preview_url = f"/api/preview/{model_id}" if os.path.exists(preview_file) else None
                stl_url = f"/api/stl/{model_id}"
                
                models.append({
                    "id": model_id,
                    "preview_url": preview_url,
                    "stl_url": stl_url
                })
            
            return self.templates.TemplateResponse(
                "models.html", 
                {"request": request, "models": models}
            )
        
        # API endpoints for serving files
        
        # Serve preview image
        @self.app.get("/api/preview/{preview_id}")
        async def get_preview(preview_id: str):
            preview_file = os.path.join(self.preview_dir, f"{preview_id}.png")
            if not os.path.exists(preview_file):
                raise HTTPException(status_code=404, detail="Preview not found")
            
            # Return the file
            with open(preview_file, "rb") as f:
                content = f.read()
            
            return {
                "content": base64.b64encode(content).decode("utf-8"),
                "content_type": "image/png"
            }
        
        # Serve STL file
        @self.app.get("/api/stl/{model_id}")
        async def get_stl(model_id: str):
            stl_file = os.path.join(self.stl_dir, f"{model_id}.stl")
            if not os.path.exists(stl_file):
                raise HTTPException(status_code=404, detail="STL file not found")
            
            # Return the file
            with open(stl_file, "rb") as f:
                content = f.read()
            
            return {
                "content": base64.b64encode(content).decode("utf-8"),
                "content_type": "application/octet-stream",
                "filename": f"{model_id}.stl"
            }
        
        # Register the router with the app
        self.app.include_router(self.router)
    
    def _create_template_files(self):
        """Create template files for the web interface."""
        # Create base template
        base_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{% block title %}OpenSCAD MCP Server{% endblock %}</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <header>
        <h1>OpenSCAD MCP Server</h1>
        <nav>
            <ul>
                <li><a href="/ui/">Home</a></li>
                <li><a href="/ui/models">Models</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        {% block content %}{% endblock %}
    </main>
    
    <footer>
        <p>OpenSCAD MCP Server - Model Context Protocol Implementation</p>
    </footer>
    
    <script src="/static/script.js"></script>
</body>
</html>
"""
        
        # Create index template
        index_template = """{% extends "base.html" %}

{% block title %}OpenSCAD MCP Server - Home{% endblock %}

{% block content %}
<section class="hero">
    <h2>Welcome to OpenSCAD MCP Server</h2>
    <p>A Model Context Protocol server for generating 3D models with OpenSCAD</p>
</section>

<section class="features">
    <div class="feature">
        <h3>Natural Language Processing</h3>
        <p>Describe 3D objects in natural language and get parametric models</p>
    </div>
    
    <div class="feature">
        <h3>Preview Generation</h3>
        <p>See your models from multiple angles before exporting</p>
    </div>
    
    <div class="feature">
        <h3>STL Export</h3>
        <p>Generate STL files ready for 3D printing</p>
    </div>
</section>
{% endblock %}
"""
        
        # Create preview template
        preview_template = """{% extends "base.html" %}

{% block title %}Model Preview - {{ model_id }}{% endblock %}

{% block content %}
<section class="model-preview">
    <h2>Model Preview: {{ model_id }}</h2>
    
    <div class="preview-container">
        {% for angle, url in previews.items() %}
        <div class="preview-angle">
            <h3>{{ angle|title }} View</h3>
            <img src="{{ url }}" alt="{{ angle }} view of {{ model_id }}" class="preview-image" data-angle="{{ angle }}">
        </div>
        {% endfor %}
    </div>
    
    {% if stl_url %}
    <div class="download-container">
        <a href="{{ stl_url }}" class="download-button" download="{{ model_id }}.stl">Download STL</a>
    </div>
    {% endif %}
</section>
{% endblock %}
"""
        
        # Create models template
        models_template = """{% extends "base.html" %}

{% block title %}All Models{% endblock %}

{% block content %}
<section class="models-list">
    <h2>All Models</h2>
    
    {% if models %}
    <div class="models-grid">
        {% for model in models %}
        <div class="model-card">
            <h3>{{ model.id }}</h3>
            {% if model.preview_url %}
            <img src="{{ model.preview_url }}" alt="Preview of {{ model.id }}" class="model-thumbnail">
            {% else %}
            <div class="no-preview">No preview available</div>
            {% endif %}
            <div class="model-actions">
                <a href="/ui/preview/{{ model.id }}" class="view-button">View</a>
                <a href="{{ model.stl_url }}" class="download-button" download="{{ model.id }}.stl">Download</a>
            </div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <p>No models found.</p>
    {% endif %}
</section>
{% endblock %}
"""
        
        # Write templates to files
        os.makedirs(self.templates_dir, exist_ok=True)
        
        with open(os.path.join(self.templates_dir, "base.html"), "w") as f:
            f.write(base_template)
        
        with open(os.path.join(self.templates_dir, "index.html"), "w") as f:
            f.write(index_template)
        
        with open(os.path.join(self.templates_dir, "preview.html"), "w") as f:
            f.write(preview_template)
        
        with open(os.path.join(self.templates_dir, "models.html"), "w") as f:
            f.write(models_template)
    
    def _create_static_files(self):
        """Create static files for the web interface."""
        # Create CSS file
        css = """/* Base styles */
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f4f4f4;
}

header {
    background-color: #333;
    color: #fff;
    padding: 1rem;
}

header h1 {
    margin-bottom: 0.5rem;
}

nav ul {
    display: flex;
    list-style: none;
}

nav ul li {
    margin-right: 1rem;
}

nav ul li a {
    color: #fff;
    text-decoration: none;
}

nav ul li a:hover {
    text-decoration: underline;
}

main {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

footer {
    background-color: #333;
    color: #fff;
    text-align: center;
    padding: 1rem;
    margin-top: 2rem;
}

/* Home page */
.hero {
    text-align: center;
    margin-bottom: 2rem;
}

.hero h2 {
    font-size: 2rem;
    margin-bottom: 1rem;
}

.features {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
}

.feature {
    flex: 1;
    background-color: #fff;
    padding: 1.5rem;
    margin: 0.5rem;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.feature h3 {
    margin-bottom: 1rem;
}

/* Preview page */
.model-preview h2 {
    margin-bottom: 1.5rem;
}

.preview-container {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.preview-angle {
    flex: 1;
    min-width: 300px;
    background-color: #fff;
    padding: 1rem;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.preview-angle h3 {
    margin-bottom: 0.5rem;
}

.preview-image {
    width: 100%;
    height: auto;
    border: 1px solid #ddd;
}

.download-container {
    text-align: center;
    margin-top: 1rem;
}

.download-button {
    display: inline-block;
    background-color: #4CAF50;
    color: white;
    padding: 0.5rem 1rem;
    text-decoration: none;
    border-radius: 4px;
}

.download-button:hover {
    background-color: #45a049;
}

/* Models page */
.models-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 1.5rem;
}

.model-card {
    background-color: #fff;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    overflow: hidden;
}

.model-card h3 {
    padding: 1rem;
    background-color: #f8f8f8;
    border-bottom: 1px solid #eee;
}

.model-thumbnail {
    width: 100%;
    height: 200px;
    object-fit: contain;
    background-color: #f4f4f4;
}

.no-preview {
    width: 100%;
    height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #f4f4f4;
    color: #999;
}

.model-actions {
    display: flex;
    padding: 1rem;
}

.view-button, .download-button {
    flex: 1;
    text-align: center;
    padding: 0.5rem;
    text-decoration: none;
    border-radius: 4px;
    margin: 0 0.25rem;
}

.view-button {
    background-color: #2196F3;
    color: white;
}

.view-button:hover {
    background-color: #0b7dda;
}
"""
        
        # Create JavaScript file
        js = """// JavaScript for OpenSCAD MCP Server web interface

document.addEventListener('DOMContentLoaded', function() {
    // Handle image loading errors
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.onerror = function() {
            this.src = '/static/placeholder.png';
        };
    });
    
    // Handle STL download
    const downloadButtons = document.querySelectorAll('.download-button');
    downloadButtons.forEach(button => {
        button.addEventListener('click', async function(e) {
            e.preventDefault();
            
            const url = this.getAttribute('href');
            const filename = this.hasAttribute('download') ? this.getAttribute('download') : 'model.stl';
            
            try {
                const response = await fetch(url);
                const data = await response.json();
                
                // Decode base64 content
                const content = atob(data.content);
                
                // Convert to Blob
                const bytes = new Uint8Array(content.length);
                for (let i = 0; i < content.length; i++) {
                    bytes[i] = content.charCodeAt(i);
                }
                const blob = new Blob([bytes], { type: data.content_type });
                
                // Create download link
                const downloadLink = document.createElement('a');
                downloadLink.href = URL.createObjectURL(blob);
                downloadLink.download = data.filename || filename;
                
                // Trigger download
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
            } catch (error) {
                console.error('Error downloading file:', error);
                alert('Error downloading file. Please try again.');
            }
        });
    });
    
    // Handle preview images
    const previewImages = document.querySelectorAll('.preview-image');
    previewImages.forEach(img => {
        img.addEventListener('click', function() {
            const url = this.getAttribute('src');
            const angle = this.getAttribute('data-angle');
            
            // Create modal for larger view
            const modal = document.createElement('div');
            modal.className = 'preview-modal';
            modal.innerHTML = `
                <div class="modal-content">
                    <span class="close-button">&times;</span>
                    <h3>${angle ? angle.charAt(0).toUpperCase() + angle.slice(1) + ' View' : 'Preview'}</h3>
                    <img src="${url}" alt="Preview">
                </div>
            `;
            
            // Add modal styles
            modal.style.position = 'fixed';
            modal.style.top = '0';
            modal.style.left = '0';
            modal.style.width = '100%';
            modal.style.height = '100%';
            modal.style.backgroundColor = 'rgba(0,0,0,0.7)';
            modal.style.display = 'flex';
            modal.style.alignItems = 'center';
            modal.style.justifyContent = 'center';
            modal.style.zIndex = '1000';
            
            const modalContent = modal.querySelector('.modal-content');
            modalContent.style.backgroundColor = '#fff';
            modalContent.style.padding = '20px';
            modalContent.style.borderRadius = '5px';
            modalContent.style.maxWidth = '90%';
            modalContent.style.maxHeight = '90%';
            modalContent.style.overflow = 'auto';
            
            const closeButton = modal.querySelector('.close-button');
            closeButton.style.float = 'right';
            closeButton.style.fontSize = '1.5rem';
            closeButton.style.fontWeight = 'bold';
            closeButton.style.cursor = 'pointer';
            
            const modalImg = modal.querySelector('img');
            modalImg.style.maxWidth = '100%';
            modalImg.style.maxHeight = '70vh';
            modalImg.style.display = 'block';
            modalImg.style.margin = '0 auto';
            
            // Add modal to body
            document.body.appendChild(modal);
            
            // Close modal when clicking close button or outside the modal
            closeButton.addEventListener('click', function() {
                document.body.removeChild(modal);
            });
            
            modal.addEventListener('click', function(e) {
                if (e.target === modal) {
                    document.body.removeChild(modal);
                }
            });
        });
    });
});
"""
        
        # Create placeholder image
        placeholder_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">
    <rect width="800" height="600" fill="#f0f0f0"/>
    <text x="400" y="300" font-family="Arial" font-size="24" text-anchor="middle" fill="#999">Preview not available</text>
</svg>"""
        
        # Write static files
        os.makedirs(self.static_dir, exist_ok=True)
        
        with open(os.path.join(self.static_dir, "styles.css"), "w") as f:
            f.write(css)
        
        with open(os.path.join(self.static_dir, "script.js"), "w") as f:
            f.write(js)
        
        with open(os.path.join(self.static_dir, "placeholder.svg"), "w") as f:
            f.write(placeholder_svg)
