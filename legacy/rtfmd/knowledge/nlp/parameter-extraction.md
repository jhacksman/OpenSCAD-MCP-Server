# Parameter Extraction for 3D Modeling

<metadata>
  author: devin-ai-integration
  timestamp: 2025-03-21T01:30:00Z
  version: 1.0.0
  tags: [parameter-extraction, nlp, 3d-modeling, regex]
</metadata>

## Overview

Parameter extraction is the process of identifying and extracting structured data from natural language descriptions of 3D models. This is a critical component in translating user intentions into actionable modeling parameters.

## Extraction Techniques

### Regular Expression Patterns

Regular expressions provide a powerful way to extract parameters:

```python
# Extract dimensions with units
dimension_pattern = r'(\d+(?:\.\d+)?)\s*(mm|cm|m|inch|in)'

# Extract color information
color_pattern = r'\b(red|green|blue|yellow|black|white|purple|orange|brown)\b'

# Extract shape type
shape_pattern = r'\b(cube|box|sphere|ball|cylinder|tube|cone|pyramid)\b'
```

### Contextual Parameter Association

After extracting raw values, they must be associated with the correct parameter:

```python
def associate_dimension(value, description):
    """Associate a dimension value with the correct parameter based on context."""
    if "width" in description or "wide" in description:
        return ("width", value)
    elif "height" in description or "tall" in description:
        return ("height", value)
    elif "depth" in description or "deep" in description:
        return ("depth", value)
    elif "radius" in description:
        return ("radius", value)
    elif "diameter" in description:
        return ("radius", value / 2)  # Convert diameter to radius
    else:
        return ("unknown", value)
```

### Default Parameters

Provide sensible defaults for unspecified parameters:

```python
default_parameters = {
    "cube": {
        "width": 10,
        "height": 10,
        "depth": 10,
        "center": True
    },
    "sphere": {
        "radius": 10,
        "segments": 32
    },
    "cylinder": {
        "radius": 5,
        "height": 10,
        "center": True,
        "segments": 32
    }
}
```

## Parameter Types

Common parameter types to extract include:

- **Dimensions**: Width, height, depth, radius, diameter
- **Positions**: X, Y, Z coordinates
- **Angles**: Rotation angles
- **Counts**: Number of sides, segments, iterations
- **Booleans**: Center, solid/hollow
- **Colors**: RGB values or named colors
- **Operations**: Union, difference, intersection
- **Transformations**: Translate, rotate, scale, mirror

## Challenges and Solutions

### Ambiguity

When parameters are ambiguous, use contextual clues or ask clarifying questions:

```python
def resolve_ambiguity(value, possible_parameters, description):
    """Resolve ambiguity between possible parameters."""
    # Try to resolve using context
    for param in possible_parameters:
        if param in description:
            return param
    
    # If still ambiguous, return a question to ask
    return f"Is {value} the {' or '.join(possible_parameters)}?"
```

### Unit Conversion

Convert all measurements to a standard unit (millimeters):

```python
def convert_to_mm(value, unit):
    """Convert a value from the given unit to millimeters."""
    if unit in ["mm", "millimeter", "millimeters"]:
        return value
    elif unit in ["cm", "centimeter", "centimeters"]:
        return value * 10
    elif unit in ["m", "meter", "meters"]:
        return value * 1000
    elif unit in ["in", "inch", "inches"]:
        return value * 25.4
    else:
        return value  # Assume mm if unit is unknown
```

### Dialog State Management

Maintain state across multiple interactions:

```python
class DialogState:
    def __init__(self):
        self.shape_type = None
        self.parameters = {}
        self.questions = []
        self.confirmed = False
    
    def add_parameter(self, name, value):
        self.parameters[name] = value
    
    def add_question(self, question):
        self.questions.append(question)
    
    def is_complete(self):
        """Check if all required parameters are present."""
        if not self.shape_type:
            return False
        
        required_params = self.get_required_parameters()
        return all(param in self.parameters for param in required_params)
    
    def get_required_parameters(self):
        """Get the required parameters for the current shape type."""
        if self.shape_type == "cube":
            return ["width", "height", "depth"]
        elif self.shape_type == "sphere":
            return ["radius"]
        elif self.shape_type == "cylinder":
            return ["radius", "height"]
        else:
            return []
```

## Best Practices

- Start with simple pattern matching and add complexity as needed
- Provide sensible defaults for all parameters
- Use contextual clues to resolve ambiguity
- Maintain dialog state for multi-turn interactions
- Convert all measurements to a standard unit
- Validate extracted parameters for reasonableness
- Handle errors gracefully with helpful messages
