"""
Retry Handler with Exponential Backoff
Provides robust retry logic for API calls and other operations
"""
import time
import logging
from typing import Callable, TypeVar, Optional, List, Tuple
from functools import wraps

T = TypeVar('T')


class RetryHandler:
    """Handles retries with exponential backoff"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize retry handler
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Add random jitter to prevent thundering herd
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def retry(
        self,
        func: Callable[[], T],
        exceptions: Tuple[Exception, ...] = (Exception,),
        on_retry: Optional[Callable[[int, Exception], None]] = None
    ) -> T:
        """
        Retry a function call with exponential backoff
        
        Args:
            func: Function to retry
            exceptions: Tuple of exceptions to catch and retry on
            on_retry: Optional callback function called on each retry (attempt_num, exception)
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func()
            except exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    
                    if on_retry:
                        on_retry(attempt + 1, e)
                    else:
                        logging.warning(
                            f"⚠️ Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                    
                    time.sleep(delay)
                else:
                    logging.error(f"❌ All {self.max_retries + 1} attempts failed. Last error: {e}")
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for current attempt"""
        import random
        
        # Exponential backoff: delay = initial_delay * (base ^ attempt)
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)  # Ensure non-negative
        
        return delay


def retry_on_failure(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Exception, ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Decorator for retrying functions with exponential backoff
    
    Usage:
        @retry_on_failure(max_retries=3, initial_delay=1.0)
        def my_api_call():
            # Your code here
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            handler = RetryHandler(
                max_retries=max_retries,
                initial_delay=initial_delay,
                max_delay=max_delay
            )
            return handler.retry(
                lambda: func(*args, **kwargs),
                exceptions=exceptions,
                on_retry=on_retry
            )
        return wrapper
    return decorator


# Common exception types for API calls
class APIError(Exception):
    """Base exception for API errors"""
    pass


class RateLimitError(APIError):
    """Exception for rate limit errors"""
    pass


class NetworkError(APIError):
    """Exception for network errors"""
    pass


class TimeoutError(APIError):
    """Exception for timeout errors"""
    pass



