"""
Error handling and retry mechanisms for remote CUDA MVS processing.

This module provides utilities for handling network errors, implementing
retry mechanisms, and ensuring robust communication with remote CUDA MVS servers.
"""

import time
import random
import logging
import functools
from typing import Callable, Any, Type, Union, List, Dict, Optional, Tuple
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define common exception types for network operations
NETWORK_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.HTTPError,
    requests.exceptions.RequestException,
    ConnectionRefusedError,
    TimeoutError,
)

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exception_types: Tuple[Type[Exception], ...] = NETWORK_EXCEPTIONS,
    jitter_factor: float = 0.1,
    logger_instance: Optional[logging.Logger] = None
) -> Callable:
    """
    Decorator for retrying a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exception_types: Tuple of exception types to catch and retry
        jitter_factor: Factor for random jitter (0.0 to 1.0)
        logger_instance: Logger instance to use (uses module logger if None)
    
    Returns:
        Decorator function
    """
    log = logger_instance or logger
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retries = 0
            last_exception = None
            
            while True:
                try:
                    return func(*args, **kwargs)
                except exception_types as e:
                    retries += 1
                    last_exception = e
                    
                    if retries > max_retries:
                        log.error(f"Max retries ({max_retries}) exceeded: {str(e)}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(max_delay, base_delay * (2 ** (retries - 1)))
                    
                    # Add jitter to avoid thundering herd problem
                    jitter = random.uniform(0, jitter_factor * delay)
                    sleep_time = delay + jitter
                    
                    log.warning(f"Retry {retries}/{max_retries} after {sleep_time:.2f}s: {str(e)}")
                    time.sleep(sleep_time)
                except Exception as e:
                    # Don't retry other exceptions
                    log.error(f"Non-retryable exception: {str(e)}")
                    raise
        
        return wrapper
    
    return decorator

def timeout_handler(
    timeout: float,
    default_value: Any = None,
    exception_types: Tuple[Type[Exception], ...] = (TimeoutError, requests.exceptions.Timeout),
    logger_instance: Optional[logging.Logger] = None
) -> Callable:
    """
    Decorator for handling timeouts in function calls.
    
    Args:
        timeout: Timeout in seconds
        default_value: Value to return if timeout occurs
        exception_types: Tuple of exception types to catch as timeouts
        logger_instance: Logger instance to use (uses module logger if None)
    
    Returns:
        Decorator function
    """
    log = logger_instance or logger
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")
            
            # Set timeout using signal
            original_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                log.warning(f"Timeout in {func.__name__}: {str(e)}")
                return default_value
            finally:
                # Reset signal handler and alarm
                signal.signal(signal.SIGALRM, original_handler)
                signal.alarm(0)
        
        return wrapper
    
    return decorator

class NetworkErrorTracker:
    """
    Tracks network errors and provides information about error patterns.
    
    This class helps identify persistent network issues and can be used
    to make decisions about server availability.
    """
    
    def __init__(
        self,
        error_window: int = 10,
        error_threshold: float = 0.5,
        reset_after: int = 100
    ):
        """
        Initialize the error tracker.
        
        Args:
            error_window: Number of recent requests to consider
            error_threshold: Error rate threshold to consider a server problematic
            reset_after: Number of successful requests after which to reset error count
        """
        self.error_window = error_window
        self.error_threshold = error_threshold
        self.reset_after = reset_after
        
        self.requests = []  # List of (timestamp, success) tuples
        self.consecutive_successes = 0
        self.consecutive_failures = 0
    
    def record_request(self, success: bool) -> None:
        """
        Record the result of a request.
        
        Args:
            success: Whether the request was successful
        """
        timestamp = time.time()
        self.requests.append((timestamp, success))
        
        # Trim old requests outside the window
        self._trim_old_requests()
        
        # Update consecutive counters
        if success:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            
            # Reset error count after enough consecutive successes
            if self.consecutive_successes >= self.reset_after:
                self.requests = [(timestamp, True)]
                self.consecutive_successes = 1
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
    
    def _trim_old_requests(self) -> None:
        """
        Remove requests that are outside the current window.
        """
        if len(self.requests) > self.error_window:
            self.requests = self.requests[-self.error_window:]
    
    def get_error_rate(self) -> float:
        """
        Get the current error rate.
        
        Returns:
            Error rate as a float between 0.0 and 1.0
        """
        if not self.requests:
            return 0.0
        
        failures = sum(1 for _, success in self.requests if not success)
        return failures / len(self.requests)
    
    def is_server_problematic(self) -> bool:
        """
        Check if the server is experiencing persistent issues.
        
        Returns:
            True if the server is problematic, False otherwise
        """
        return self.get_error_rate() >= self.error_threshold
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the error tracker.
        
        Returns:
            Dictionary with status information
        """
        return {
            "error_rate": self.get_error_rate(),
            "is_problematic": self.is_server_problematic(),
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "total_requests": len(self.requests),
            "recent_failures": sum(1 for _, success in self.requests if not success)
        }

class CircuitBreaker:
    """
    Circuit breaker pattern implementation for network requests.
    
    This class helps prevent cascading failures by stopping requests
    to a problematic server until it recovers.
    """
    
    # Circuit states
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # No requests allowed
    HALF_OPEN = "half_open"  # Testing if service is back
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        reset_timeout: float = 60.0,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of consecutive failures before opening
            recovery_timeout: Time in seconds before testing recovery
            reset_timeout: Time in seconds before fully resetting
            logger_instance: Logger instance to use (uses module logger if None)
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.reset_timeout = reset_timeout
        self.log = logger_instance or logger
        
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
    
    def record_success(self) -> None:
        """
        Record a successful request.
        """
        self.last_success_time = time.time()
        
        if self.state == self.HALF_OPEN:
            self.log.info("Circuit breaker reset to closed state after successful test request")
            self.state = self.CLOSED
            self.failure_count = 0
        elif self.state == self.CLOSED:
            # Reset failure count after a successful request
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """
        Record a failed request.
        """
        self.last_failure_time = time.time()
        
        if self.state == self.CLOSED:
            self.failure_count += 1
            
            if self.failure_count >= self.failure_threshold:
                self.log.warning(f"Circuit breaker opened after {self.failure_count} consecutive failures")
                self.state = self.OPEN
        elif self.state == self.HALF_OPEN:
            self.log.warning("Circuit breaker opened again after failed test request")
            self.state = self.OPEN
    
    def allow_request(self) -> bool:
        """
        Check if a request should be allowed.
        
        Returns:
            True if the request should be allowed, False otherwise
        """
        if self.state == self.CLOSED:
            return True
        
        if self.state == self.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.log.info("Circuit breaker entering half-open state to test service")
                self.state = self.HALF_OPEN
                return True
            return False
        
        # In HALF_OPEN state, allow only one request
        return True
    
    def reset(self) -> None:
        """
        Reset the circuit breaker to closed state.
        """
        self.state = self.CLOSED
        self.failure_count = 0
        self.log.info("Circuit breaker manually reset to closed state")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the circuit breaker.
        
        Returns:
            Dictionary with status information
        """
        now = time.time()
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "time_since_last_failure": now - self.last_failure_time if self.last_failure_time > 0 else None,
            "time_since_last_success": now - self.last_success_time if self.last_success_time > 0 else None,
            "recovery_timeout": self.recovery_timeout,
            "reset_timeout": self.reset_timeout
        }

def safe_request(
    url: str,
    method: str = "GET",
    circuit_breaker: Optional[CircuitBreaker] = None,
    error_tracker: Optional[NetworkErrorTracker] = None,
    retry_count: int = 3,
    timeout: float = 30.0,
    **kwargs
) -> Optional[requests.Response]:
    """
    Make a safe HTTP request with circuit breaker and retry logic.
    
    Args:
        url: URL to request
        method: HTTP method (GET, POST, etc.)
        circuit_breaker: Circuit breaker instance
        error_tracker: Error tracker instance
        retry_count: Number of retries
        timeout: Request timeout in seconds
        **kwargs: Additional arguments for requests
    
    Returns:
        Response object or None if request failed
    """
    # Check circuit breaker
    if circuit_breaker and not circuit_breaker.allow_request():
        logger.warning(f"Circuit breaker prevented request to {url}")
        return None
    
    # Set default timeout
    kwargs.setdefault("timeout", timeout)
    
    # Make request with retry
    response = None
    success = False
    
    try:
        for attempt in range(retry_count + 1):
            try:
                response = requests.request(method, url, **kwargs)
                response.raise_for_status()
                success = True
                break
            except NETWORK_EXCEPTIONS as e:
                if attempt < retry_count:
                    delay = 2 ** attempt + random.uniform(0, 1)
                    logger.warning(f"Request to {url} failed (attempt {attempt+1}/{retry_count+1}): {str(e)}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    logger.error(f"Request to {url} failed after {retry_count+1} attempts: {str(e)}")
                    raise
    except Exception as e:
        logger.error(f"Error making request to {url}: {str(e)}")
        success = False
    
    # Update circuit breaker and error tracker
    if circuit_breaker:
        if success:
            circuit_breaker.record_success()
        else:
            circuit_breaker.record_failure()
    
    if error_tracker:
        error_tracker.record_request(success)
    
    return response if success else None
