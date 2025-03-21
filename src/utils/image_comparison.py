"""
Utilities for comparing images for consistency testing.
"""

import os
import logging
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class ImageComparisonMetrics:
    """
    Class for computing various image comparison metrics to evaluate consistency.
    """
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Load an image from path and convert to RGB numpy array.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Numpy array of the image in RGB format
        """
        try:
            img = Image.open(image_path).convert('RGB')
            return np.array(img)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
    
    @staticmethod
    def compute_ssim(image1_path: str, image2_path: str, 
                     resize: bool = True, size: Tuple[int, int] = (512, 512)) -> float:
        """
        Compute the Structural Similarity Index (SSIM) between two images.
        
        Args:
            image1_path: Path to the first image
            image2_path: Path to the second image
            resize: Whether to resize images before comparison
            size: Size to resize images to if resize is True
            
        Returns:
            SSIM score between 0 and 1 (higher is more similar)
        """
        try:
            # Load images
            img1 = ImageComparisonMetrics.load_image(image1_path)
            img2 = ImageComparisonMetrics.load_image(image2_path)
            
            # Resize if needed
            if resize:
                img1 = cv2.resize(img1, size)
                img2 = cv2.resize(img2, size)
            
            # Convert to grayscale for SSIM
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            
            # Compute SSIM
            score, _ = ssim(img1_gray, img2_gray, full=True)
            return score
        except Exception as e:
            logger.error(f"Error computing SSIM: {str(e)}")
            return 0.0
    
    @staticmethod
    def compute_histogram_similarity(image1_path: str, image2_path: str) -> float:
        """
        Compute histogram similarity between two images.
        
        Args:
            image1_path: Path to the first image
            image2_path: Path to the second image
            
        Returns:
            Histogram similarity score between 0 and 1 (higher is more similar)
        """
        try:
            # Load images
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            
            if img1 is None or img2 is None:
                logger.error(f"Failed to load images for histogram comparison")
                return 0.0
            
            # Convert to HSV color space
            hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms
            h_bins = 50
            s_bins = 60
            histSize = [h_bins, s_bins]
            h_ranges = [0, 180]
            s_ranges = [0, 256]
            ranges = h_ranges + s_ranges
            channels = [0, 1]
            
            hist1 = cv2.calcHist([hsv1], channels, None, histSize, ranges, accumulate=False)
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            hist2 = cv2.calcHist([hsv2], channels, None, histSize, ranges, accumulate=False)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # Compare histograms
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return similarity
        except Exception as e:
            logger.error(f"Error computing histogram similarity: {str(e)}")
            return 0.0
    
    @staticmethod
    def compute_color_palette_similarity(image1_path: str, image2_path: str, 
                                        n_colors: int = 5) -> float:
        """
        Extract dominant color palettes from both images and compare them.
        
        Args:
            image1_path: Path to the first image
            image2_path: Path to the second image
            n_colors: Number of dominant colors to extract
            
        Returns:
            Color palette similarity score between 0 and 1 (higher is more similar)
        """
        try:
            # Load images
            img1 = ImageComparisonMetrics.load_image(image1_path)
            img2 = ImageComparisonMetrics.load_image(image2_path)
            
            # Reshape images for k-means
            pixels1 = img1.reshape(-1, 3)
            pixels2 = img2.reshape(-1, 3)
            
            # Extract dominant colors using k-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            
            _, labels1, centers1 = cv2.kmeans(pixels1.astype(np.float32), n_colors, None, criteria, 10, flags)
            _, labels2, centers2 = cv2.kmeans(pixels2.astype(np.float32), n_colors, None, criteria, 10, flags)
            
            # Count pixels in each cluster
            counts1 = np.bincount(labels1.flatten())
            counts2 = np.bincount(labels2.flatten())
            
            # Sort centers by frequency
            centers1 = centers1[np.argsort(-counts1)]
            centers2 = centers2[np.argsort(-counts2)]
            
            # Compute similarity between color palettes
            # (using a simple Euclidean distance between sorted centers)
            total_distance = 0
            for i in range(min(len(centers1), len(centers2))):
                total_distance += np.linalg.norm(centers1[i] - centers2[i])
            
            # Normalize to 0-1 range (higher is more similar)
            max_possible_distance = 255 * np.sqrt(3) * n_colors  # Maximum possible RGB distance
            similarity = 1 - (total_distance / max_possible_distance)
            
            return similarity
        except Exception as e:
            logger.error(f"Error computing color palette similarity: {str(e)}")
            return 0.0
    
    @staticmethod
    def compute_all_metrics(image1_path: str, image2_path: str) -> Dict[str, float]:
        """
        Compute all available similarity metrics between two images.
        
        Args:
            image1_path: Path to the first image
            image2_path: Path to the second image
            
        Returns:
            Dictionary of metric names and scores
        """
        metrics = {
            "ssim": ImageComparisonMetrics.compute_ssim(image1_path, image2_path),
            "histogram_similarity": ImageComparisonMetrics.compute_histogram_similarity(image1_path, image2_path),
            "color_palette_similarity": ImageComparisonMetrics.compute_color_palette_similarity(image1_path, image2_path)
        }
        
        # Compute average similarity score
        metrics["average_similarity"] = sum(metrics.values()) / len(metrics)
        
        return metrics

class MultiViewConsistencyEvaluator:
    """
    Class for evaluating consistency across multiple views of the same 3D object.
    """
    
    def __init__(self, output_dir: str = "output/consistency_evaluation"):
        """
        Initialize the consistency evaluator.
        
        Args:
            output_dir: Directory to store evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = ImageComparisonMetrics()
    
    def evaluate_view_consistency(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Evaluate consistency across multiple views of the same object.
        
        Args:
            image_paths: List of paths to images representing different views
            
        Returns:
            Dictionary containing evaluation results
        """
        if len(image_paths) < 2:
            logger.warning("Need at least 2 images to evaluate consistency")
            return {"error": "Insufficient images"}
        
        results = {
            "image_count": len(image_paths),
            "pairwise_metrics": [],
            "average_metrics": {}
        }
        
        # Compute pairwise metrics
        metric_sums = {"ssim": 0, "histogram_similarity": 0, "color_palette_similarity": 0}
        pair_count = 0
        
        for i in range(len(image_paths)):
            for j in range(i+1, len(image_paths)):
                metrics = ImageComparisonMetrics.compute_all_metrics(image_paths[i], image_paths[j])
                
                pair_result = {
                    "image1": image_paths[i],
                    "image2": image_paths[j],
                    "metrics": metrics
                }
                
                results["pairwise_metrics"].append(pair_result)
                
                # Accumulate for averages
                for key in metric_sums.keys():
                    metric_sums[key] += metrics[key]
                pair_count += 1
        
        # Compute average metrics
        for key in metric_sums.keys():
            results["average_metrics"][key] = metric_sums[key] / pair_count if pair_count > 0 else 0
        
        # Compute overall consistency score (average of all metrics)
        results["overall_consistency_score"] = sum(results["average_metrics"].values()) / len(results["average_metrics"])
        
        return results
    
    def compare_generation_methods(self, 
                                  venice_images: List[str], 
                                  google_images: List[str]) -> Dict[str, Any]:
        """
        Compare consistency between images generated by Venice.ai and Google Gemini.
        
        Args:
            venice_images: List of paths to images generated by Venice.ai
            google_images: List of paths to images generated by Google Gemini
            
        Returns:
            Dictionary containing comparison results
        """
        results = {
            "venice_consistency": self.evaluate_view_consistency(venice_images),
            "google_consistency": self.evaluate_view_consistency(google_images),
            "cross_platform_similarity": []
        }
        
        # Compare corresponding views across platforms
        for i in range(min(len(venice_images), len(google_images))):
            metrics = ImageComparisonMetrics.compute_all_metrics(venice_images[i], google_images[i])
            
            results["cross_platform_similarity"].append({
                "venice_image": venice_images[i],
                "google_image": google_images[i],
                "metrics": metrics
            })
        
        # Compute average cross-platform similarity
        avg_metrics = {"ssim": 0, "histogram_similarity": 0, "color_palette_similarity": 0}
        for comparison in results["cross_platform_similarity"]:
            for key in avg_metrics.keys():
                avg_metrics[key] += comparison["metrics"][key]
        
        if results["cross_platform_similarity"]:
            for key in avg_metrics.keys():
                avg_metrics[key] /= len(results["cross_platform_similarity"])
        
        results["average_cross_platform_similarity"] = avg_metrics
        
        return results
    
    def save_evaluation_report(self, results: Dict[str, Any], 
                              report_name: str = "consistency_report.json") -> str:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Evaluation results dictionary
            report_name: Name of the report file
            
        Returns:
            Path to the saved report
        """
        import json
        
        report_path = os.path.join(self.output_dir, report_name)
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation report saved to {report_path}")
        return report_path
