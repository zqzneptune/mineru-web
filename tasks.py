import os
import sys
import uuid
import logging
import threading
import multiprocessing
import gc
from pathlib import Path
from typing import Dict, List, Callable, Optional
from datetime import datetime
import queue

# Global storage for log listeners and queues (with thread-safe locks)
_log_lock = threading.RLock()
log_listeners: Dict[str, List[Callable]] = {}
log_queues: Dict[str, queue.Queue] = {}
task_status: Dict[str, dict] = {}

# This prevents multiple background threads from hitting the GPU at the same time
gpu_processing_lock = threading.Lock()


class LogHandler(logging.Handler):
    """Custom log handler that streams logs to listeners"""
    
    def __init__(self, task_id: str):
        super().__init__()
        self.task_id = task_id
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    def emit(self, record):
        log_message = self.format(record)
        with _log_lock:
            listeners = log_listeners.get(self.task_id, [])
        for callback in listeners:
            try:
                callback(log_message)
            except Exception:
                pass


def register_listener(task_id: str, callback: Callable):
    """Register a listener for log messages (thread-safe)"""
    with _log_lock:
        if task_id not in log_listeners:
            log_listeners[task_id] = []
        log_listeners[task_id].append(callback)


def unregister_listener(task_id: str):
    """Unregister all listeners for a task (thread-safe)"""
    with _log_lock:
        if task_id in log_listeners:
            del log_listeners[task_id]


def create_log_queue(task_id: str) -> queue.Queue:
    """Create a log queue for a task (thread-safe)"""
    with _log_lock:
        log_queues[task_id] = queue.Queue()
        return log_queues[task_id]


def get_log_queue(task_id: str) -> Optional[queue.Queue]:
    """Get the log queue for a task (thread-safe)"""
    with _log_lock:
        return log_queues.get(task_id)


def remove_log_queue(task_id: str):
    """Remove the log queue for a task (thread-safe)"""
    with _log_lock:
        if task_id in log_queues:
            del log_queues[task_id]


def send_log(task_id: str, message: str):
    """Send a log message to all listeners and queue (thread-safe)"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    
    # Send to registered callbacks
    with _log_lock:
        listeners = log_listeners.get(task_id, [])
        queue_obj = log_queues.get(task_id)
    
    for callback in listeners:
        try:
            callback(log_message)
        except Exception:
            pass
    
    # Put in queue for SSE streaming
    if queue_obj:
        try:
            queue_obj.put(log_message, block=False)
        except queue.Full:
            pass


def cleanup_gpu_memory():
    """Clean up GPU and host memory after processing"""
    # First force garbage collection to free up any unreferenced objects
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            # Synchronize CUDA operations to ensure all GPU tasks are complete
            torch.cuda.synchronize()
            # Clear the CUDA cache to free up GPU memory
            torch.cuda.empty_cache()
            # Reset peak memory stats for this process
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
    except ImportError:
        # torch not installed, skip CUDA cleanup
        pass
    except Exception as e:
        # Log any other CUDA-related errors but don't fail
        import logging
        logging.getLogger(__name__).warning(f"GPU memory cleanup warning: {e}")
    
    # Final garbage collection to ensure all memory is freed
    gc.collect()


def _run_mineru_isolated(output_dir, pdf_path, lang, backend, gpu_memory_utilization):
    """Runs in a completely separate OS process to guarantee GPU memory is freed afterwards"""
    import os
    import sys
    
    # Prevents memory fragmentation, which is critical for 8GB GPUs
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Force GPU utilization setting inside the isolated process
    os.environ['MINERU_VLLM_GPU_MEMORY_UTILIZATION'] = str(gpu_memory_utilization)
    
    try:
        from mineru.cli.common import do_parse, read_fn
        from pathlib import Path
        
        pdf_file_name = Path(pdf_path).stem
        pdf_bytes = read_fn(pdf_path)
        
        do_parse(
            output_dir=output_dir,
            pdf_file_names=[pdf_file_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[lang],
            backend=backend,
            parse_method='auto',
            formula_enable=True,
            table_enable=True,
            server_url=None,
            start_page_id=0,
            end_page_id=None,
            gpu_memory_utilization=gpu_memory_utilization
        )
        
        # CRITICAL FIX: Force immediate OS-level exit
        # vLLM and PyTorch leave background workers running that prevent the process from closing naturally.
        # This causes the Flask parent thread to hang forever. os._exit(0) brutally kills the process so Flask can continue.
        os._exit(0)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # If an error happens, exit with code 1 so the parent process knows it failed
        os._exit(1)


def process_document(doc_id: int, pdf_path: str, backend: str = 'hybrid-auto-engine', lang: str = 'ch'):
    """Process a document using MinerU with process isolation to prevent GPU memory issues"""
    task_id = f"task_{doc_id}"
    
    # Initialize task status
    task_status[task_id] = {
        'doc_id': doc_id,
        'status': 'running',
        'start_time': datetime.now(),
        'logs': []
    }
    
    send_log(task_id, f"Starting document processing (ID: {doc_id})")
    send_log(task_id, f"Using backend: {backend}, language: {lang}")
    
    # Set up logging
    logger = logging.getLogger('mineru')
    log_handler = LogHandler(task_id)
    log_handler.setLevel(logging.INFO)
    logger.addHandler(log_handler)
    
    # Import database functions outside try block so they're available in the exception handler
    from database import get_document, update_document_status
    
    try:
        # Get document info from database
        doc = get_document(doc_id)
        
        if not doc:
            send_log(task_id, f"Error: Document not found in database")
            update_document_status(doc_id, 'failed', 'Document not found')
            return
        
        output_dir = doc['output_path']
        
        send_log(task_id, f"Input file: {pdf_path}")
        send_log(task_id, f"Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # --- CHANGE THIS LINE ---
        # 0.75 gives 5.7GB to vLLM and leaves 1.9GB free for the Formula/OCR models
        gpu_memory_utilization = float(os.environ.get('MINERU_VLLM_GPU_MEMORY_UTILIZATION', '0.75'))
        send_log(task_id, f"GPU memory utilization set to: {gpu_memory_utilization}")
        
        # Wait for GPU availability and acquire lock to prevent concurrent GPU usage
        send_log(task_id, "Waiting in queue for GPU availability...")
        
        with gpu_processing_lock:
            send_log(task_id, "GPU acquired. Calling MinerU parser in isolated process...")
            start_time = datetime.now()
            
            # Run MinerU in an isolated process with 'spawn' context to ensure clean GPU memory
            # This prevents vLLM cache block issues that occur when running in Flask threads
            ctx = multiprocessing.get_context('spawn')
            p = ctx.Process(
                target=_run_mineru_isolated,
                args=(output_dir, pdf_path, lang, backend, gpu_memory_utilization)
            )
            p.start()
            p.join()  # Wait for the process to finish
            
            if p.exitcode != 0:
                raise RuntimeError(f"MinerU isolated process failed with exit code {p.exitcode}. See console for details.")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            send_log(task_id, f"Processing completed successfully!")
            send_log(task_id, f"Total time: {duration:.2f} seconds")
        
        # Update database status
        update_document_status(doc_id, 'completed')
        
        # List output files
        send_log(task_id, "Output files:")
        output_path = Path(output_dir)
        for item in output_path.rglob('*'):
            if item.is_file():
                size_kb = item.stat().st_size / 1024
                send_log(task_id, f"  - {item.name} ({size_kb:.1f} KB)")
        
    except Exception as e:
        import traceback
        send_log(task_id, f"Error occurred: {str(e)}")
        send_log(task_id, f"Traceback: {traceback.format_exc()}")
        update_document_status(doc_id, 'failed', str(e))
    
    finally:
        # Remove log handler
        logger.removeHandler(log_handler)
        log_handler.close()
        
        # Clean up
        if task_id in task_status:
            task_status[task_id]['status'] = 'completed' if task_status[task_id]['status'] == 'running' else 'failed'
            task_status[task_id]['end_time'] = datetime.now()
        
        # Unregister listeners after a delay to ensure all logs are sent
        def cleanup():
            import time
            time.sleep(2)
            unregister_listener(task_id)
        
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()


def start_processing_thread(doc_id: int, pdf_path: str, backend: str = 'hybrid-auto-engine', lang: str = 'ch'):
    """Start processing in a background thread"""
    thread = threading.Thread(
        target=process_document,
        args=(doc_id, pdf_path, backend, lang),
        daemon=True
    )
    thread.start()
    return thread


def get_task_status(task_id: str) -> dict:
    """Get task status"""
    return task_status.get(task_id, {'status': 'not_found'})