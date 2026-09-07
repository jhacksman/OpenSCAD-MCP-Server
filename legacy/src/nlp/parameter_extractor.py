import re
import logging
from typing import Dict, Any, Tuple, List, Optional
import json

logger = logging.getLogger(__name__)

class ParameterExtractor:
    """
    Extract parameters from natural language descriptions.
    Implements dialog flow for collecting specifications and translating them to OpenSCAD parameters.
    """
    
    def __init__(self):
        """Initialize the parameter extractor."""
        # Using only millimeters as per project requirements
        self.unit_conversions = {
            'mm': 1.0
        }
        
        # Shape recognition patterns with expanded vocabulary
        self.shape_patterns = {
            'cube': r'\b(cube|box|square|rectangular|block|cuboid|brick)\b',
            'sphere': r'\b(sphere|ball|round|circular|globe|orb)\b',
            'cylinder': r'\b(cylinder|tube|pipe|rod|circular column|pillar|column)\b',
            'box': r'\b(hollow box|container|case|enclosure|bin|chest|tray)\b',
            'rounded_box': r'\b(rounded box|rounded container|rounded case|rounded enclosure|smooth box|rounded corners|chamfered box)\b',
            'cone': r'\b(cone|pyramid|tapered cylinder|funnel)\b',
            'torus': r'\b(torus|donut|ring|loop|circular ring)\b',
            'prism': r'\b(prism|triangular prism|wedge|triangular shape)\b',
            'custom': r'\b(custom|complex|special|unique|combined|composite)\b'
        }
        
        # Parameter recognition patterns with enhanced unit detection
        self.parameter_patterns = {
            'width': r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in|inch|inches|ft|foot|feet)?\s*(?:wide|width|across|w)',
            'height': r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in|inch|inches|ft|foot|feet)?\s*(?:high|height|tall|h)',
            'depth': r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in|inch|inches|ft|foot|feet)?\s*(?:deep|depth|long|d|length)',
            'radius': r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in|inch|inches|ft|foot|feet)?\s*(?:radius|r)',
            'diameter': r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in|inch|inches|ft|foot|feet)?\s*(?:diameter|dia)',
            'thickness': r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in|inch|inches|ft|foot|feet)?\s*(?:thick|thickness|t)',
            'segments': r'(\d+)\s*(?:segments|sides|faces|facets|smoothness)',
            'center': r'\b(centered|center|middle|origin)\b',
            'angle': r'(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|°)?\s*(?:angle|rotation|rotate|tilt)',
            'scale': r'(\d+(?:\.\d+)?)\s*(?:x|times|scale|scaling|factor)',
            'resolution': r'(\d+(?:\.\d+)?)\s*(?:resolution|quality|detail)'
        }
        
        # Dialog state for multi-turn conversations
        self.dialog_state = {}
    
    def extract_parameters(self, description: str, model_type: Optional[str] = None, 
                              existing_parameters: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Extract model type and parameters from a natural language description.
        
        Args:
            description: Natural language description of the 3D object
            model_type: Optional model type for context (if already known)
            existing_parameters: Optional existing parameters for context (for modifications)
            
        Returns:
            Tuple of (model_type, parameters)
        """
        # Use provided model_type or determine from description
        if model_type is None:
            model_type = self._determine_shape_type(description)
        
        # Start with existing parameters if provided
        parameters = existing_parameters.copy() if existing_parameters else {}
        
        # Extract parameters based on the shape type
        new_parameters = self._extract_shape_parameters(description, model_type)
        
        # Update parameters with newly extracted ones
        parameters.update(new_parameters)
        
        # Apply default parameters if needed
        parameters = self._apply_default_parameters(model_type, parameters)
        
        logger.info(f"Extracted model type: {model_type}, parameters: {parameters}")
        return model_type, parameters
    
    def extract_parameters_from_modifications(self, modifications: str, model_type: Optional[str] = None, 
                                               existing_parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract parameters from modification description with contextual understanding.
        
        Args:
            modifications: Description of modifications to make
            model_type: Optional model type for context
            existing_parameters: Optional existing parameters for context
            
        Returns:
            Dictionary of parameters to update
        """
        # Start with existing parameters if provided
        parameters = existing_parameters.copy() if existing_parameters else {}
        
        # Extract all possible parameters from the modifications
        new_parameters = {}
        for param_name, pattern in self.parameter_patterns.items():
            matches = re.findall(pattern, modifications, re.IGNORECASE)
            if matches:
                # Take the last match if multiple are found
                value = matches[-1]
                if isinstance(value, tuple):
                    value = value[0]  # Extract from capture group
                new_parameters[param_name] = self._convert_to_mm(value, modifications)
        
        # Update parameters with newly extracted ones
        parameters.update(new_parameters)
        
        # Apply contextual understanding based on model type
        if model_type and not new_parameters:
            # If no explicit parameters were found, try to infer from context
            # For now, we'll just log this case since inference is complex
            logger.info(f"No explicit parameters found in '{modifications}', using existing parameters")
        
        logger.info(f"Extracted modification parameters: {parameters}")
        return parameters
    
    def get_missing_parameters(self, model_type: str, parameters: Dict[str, Any]) -> List[str]:
        """
        Determine which required parameters are missing for a given model type.
        
        Args:
            model_type: Type of model
            parameters: Currently extracted parameters
            
        Returns:
            List of missing parameter names
        """
        required_params = self._get_required_parameters(model_type)
        return [param for param in required_params if param not in parameters]
    
    def update_dialog_state(self, user_id: str, model_type: Optional[str] = None, 
                           parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the dialog state for a user.
        
        Args:
            user_id: Unique identifier for the user
            model_type: Optional model type to update
            parameters: Optional parameters to update
        """
        if user_id not in self.dialog_state:
            self.dialog_state[user_id] = {
                'model_type': None,
                'parameters': {},
                'missing_parameters': [],
                'current_question': None
            }
        
        if model_type:
            self.dialog_state[user_id]['model_type'] = model_type
        
        if parameters:
            self.dialog_state[user_id]['parameters'].update(parameters)
            
        # Update missing parameters
        if self.dialog_state[user_id]['model_type']:
            missing = self.get_missing_parameters(
                self.dialog_state[user_id]['model_type'],
                self.dialog_state[user_id]['parameters']
            )
            self.dialog_state[user_id]['missing_parameters'] = missing
    
    def get_next_question(self, user_id: str) -> Optional[str]:
        """
        Get the next question to ask the user based on missing parameters.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Question string or None if all parameters are collected
        """
        if user_id not in self.dialog_state:
            return "What kind of 3D object would you like to create?"
        
        state = self.dialog_state[user_id]
        
        # If we don't have a model type yet, ask for it
        if not state['model_type']:
            state['current_question'] = "What kind of 3D object would you like to create?"
            return state['current_question']
        
        # If we have missing parameters, ask for the first one
        if state['missing_parameters']:
            param = state['missing_parameters'][0]
            question = self._get_parameter_question(param, state['model_type'])
            state['current_question'] = question
            return question
        
        # All parameters collected
        state['current_question'] = None
        return None
    
    def process_answer(self, user_id: str, answer: str) -> Dict[str, Any]:
        """
        Process a user's answer to a question.
        
        Args:
            user_id: Unique identifier for the user
            answer: User's answer to the current question
            
        Returns:
            Updated dialog state
        """
        if user_id not in self.dialog_state:
            # Initialize with default state
            self.update_dialog_state(user_id)
        
        state = self.dialog_state[user_id]
        current_question = state['current_question']
        
        # Process based on current question
        if not state['model_type']:
            # Trying to determine the model type
            model_type = self._determine_shape_type(answer)
            self.update_dialog_state(user_id, model_type=model_type)
        elif state['missing_parameters']:
            # Trying to collect a specific parameter
            param = state['missing_parameters'][0]
            value = self._extract_parameter_value(param, answer)
            if value is not None:
                self.update_dialog_state(user_id, parameters={param: value})
        
        # Return the updated state
        return self.dialog_state[user_id]
    
    def _determine_shape_type(self, description: str) -> str:
        """
        Determine the shape type from the description.
        Enhanced to support more shape types and better pattern matching.
        """
        # Check for explicit shape mentions
        for shape, pattern in self.shape_patterns.items():
            if re.search(pattern, description, re.IGNORECASE):
                logger.info(f"Detected shape type: {shape} from pattern: {pattern}")
                return shape
        
        # Try to infer shape from context if no explicit mention
        if re.search(r'\b(round|circular|sphere|ball)\b', description, re.IGNORECASE):
            return "sphere"
        elif re.search(r'\b(tall|column|pillar|rod)\b', description, re.IGNORECASE):
            return "cylinder"
        elif re.search(r'\b(box|container|case|enclosure)\b', description, re.IGNORECASE):
            # Determine if it should be a rounded box
            if re.search(r'\b(rounded|smooth|chamfered)\b', description, re.IGNORECASE):
                return "rounded_box"
            return "box"
        
        # Default to cube if no shape is detected
        logger.info("No specific shape detected, defaulting to cube")
        return "cube"
    
    def _extract_shape_parameters(self, description: str, model_type: str) -> Dict[str, Any]:
        """Extract parameters for a specific shape type."""
        parameters = {}
        
        # Extract all possible parameters
        for param_name, pattern in self.parameter_patterns.items():
            matches = re.findall(pattern, description, re.IGNORECASE)
            if matches:
                # Take the last match if multiple are found
                value = matches[-1]
                if isinstance(value, tuple):
                    value = value[0]  # Extract from capture group
                parameters[param_name] = self._convert_to_mm(value, description)
        
        # Special case for diameter -> radius conversion
        if 'diameter' in parameters and 'radius' not in parameters:
            parameters['radius'] = parameters['diameter'] / 2
            del parameters['diameter']
        
        # Special case for center parameter
        if 'center' in parameters:
            center_value = parameters['center']
            if isinstance(center_value, (int, float)):
                # Convert numeric value to boolean string
                parameters['center'] = 'true' if center_value > 0 else 'false'
            else:
                # Convert string value to boolean string
                center_str = str(center_value).lower()
                parameters['center'] = 'true' if center_str in ['true', 'yes', 'y', '1'] else 'false'
        
        return parameters
    
    def _convert_to_mm(self, value_str: str, context: str) -> float:
        """
        Convert a value to millimeters.
        As per project requirements, we only use millimeters for design.
        """
        try:
            value = float(value_str)
            
            # Since we're only using millimeters, we just return the value directly
            # This simplifies the conversion logic while maintaining the function interface
            logger.info(f"Using value {value} in millimeters")
            return value
        except ValueError:
            logger.warning(f"Could not convert value to float: {value_str}")
            return 0.0
    
    def _apply_default_parameters(self, model_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default parameters based on the model type."""
        defaults = {
            'cube': {'width': 10, 'depth': 10, 'height': 10, 'center': 'false'},
            'sphere': {'radius': 10, 'segments': 32},
            'cylinder': {'radius': 10, 'height': 20, 'center': 'false', 'segments': 32},
            'box': {'width': 30, 'depth': 20, 'height': 15, 'thickness': 2},
            'rounded_box': {'width': 30, 'depth': 20, 'height': 15, 'radius': 3, 'segments': 32},
            'cone': {'base_radius': 10, 'height': 20, 'center': 'false', 'segments': 32},
            'torus': {'major_radius': 20, 'minor_radius': 5, 'segments': 32},
            'prism': {'width': 20, 'height': 15, 'depth': 20, 'center': 'false'},
            'custom': {'width': 20, 'height': 20, 'depth': 20, 'center': 'false'}
        }
        
        # Get defaults for the model type
        model_defaults = defaults.get(model_type, {})
        
        # Apply defaults for missing parameters
        for param, default_value in model_defaults.items():
            if param not in parameters:
                parameters[param] = default_value
        
        return parameters
    
    def _get_required_parameters(self, model_type: str) -> List[str]:
        """Get the list of required parameters for a model type."""
        required_params = {
            'cube': ['width', 'depth', 'height'],
            'sphere': ['radius'],
            'cylinder': ['radius', 'height'],
            'box': ['width', 'depth', 'height', 'thickness'],
            'rounded_box': ['width', 'depth', 'height', 'radius'],
            'cone': ['base_radius', 'height'],
            'torus': ['major_radius', 'minor_radius'],
            'prism': ['width', 'height', 'depth'],
            'custom': ['width', 'height', 'depth']
        }
        
        return required_params.get(model_type, [])
    
    def _get_parameter_question(self, param: str, model_type: str) -> str:
        """Get a question to ask for a specific parameter."""
        questions = {
            'width': f"What should be the width of the {model_type} in mm?",
            'depth': f"What should be the depth of the {model_type} in mm?",
            'height': f"What should be the height of the {model_type} in mm?",
            'radius': f"What should be the radius of the {model_type} in mm?",
            'thickness': f"What should be the wall thickness of the {model_type} in mm?",
            'segments': f"How many segments should the {model_type} have for smoothness?",
            'base_radius': f"What should be the base radius of the {model_type} in mm?",
            'major_radius': f"What should be the major radius of the {model_type} in mm?",
            'minor_radius': f"What should be the minor radius of the {model_type} in mm?",
            'diameter': f"What should be the diameter of the {model_type} in mm?",
            'angle': f"What should be the angle of the {model_type} in degrees?",
            'scale': f"What should be the scale factor for the {model_type}?",
            'resolution': f"What resolution should the {model_type} have (higher means more detailed)?",
            'center': f"Should the {model_type} be centered? (yes/no)"
        }
        
        return questions.get(param, f"What should be the {param} of the {model_type}?")
    
    def _extract_parameter_value(self, param: str, answer: str) -> Optional[float]:
        """Extract a parameter value from an answer."""
        pattern = self.parameter_patterns.get(param)
        if not pattern:
            # For parameters without specific patterns, try to extract any number
            pattern = r'(\d+(?:\.\d+)?)'
        
        matches = re.findall(pattern, answer, re.IGNORECASE)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[0]  # Extract from capture group
            return self._convert_to_mm(value, answer)
        
        # Try to extract just a number
        matches = re.findall(r'(\d+(?:\.\d+)?)', answer)
        if matches:
            value = matches[-1]
            return self._convert_to_mm(value, answer)
        
        return None
