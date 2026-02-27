import sqlite3
import os
import threading
import functools
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'mineru_web.db')

# Thread-local storage for database connections
_local = threading.local()

# Lock for thread-safe database operations
_db_lock = threading.RLock()

# Cache for folder tree (with TTL)
_folder_tree_cache: Dict[str, tuple] = {}
_folder_tree_cache_ttl = 5  # seconds - reduced for faster updates
_folder_tree_cache_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    """Get a thread-local database connection with proper settings"""
    if not hasattr(_local, 'connection') or _local.connection is None:
        conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent performance
        conn.execute('PRAGMA journal_mode=WAL')
        # Reduce busy timeout
        conn.execute('PRAGMA busy_timeout=5000')
        _local.connection = conn
    return _local.connection


def _close_connection():
    """Close the thread-local database connection"""
    if hasattr(_local, 'connection') and _local.connection is not None:
        try:
            _local.connection.close()
        except Exception:
            pass
        _local.connection = None


def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
    cursor = conn.cursor()
    
    # Enable WAL mode for better concurrent performance
    cursor.execute('PRAGMA journal_mode=WAL')
    cursor.execute('PRAGMA busy_timeout=5000')
    
    # Create documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            upload_time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            output_path TEXT,
            file_size INTEGER,
            error_message TEXT,
            backend TEXT DEFAULT 'hybrid-auto-engine',
            lang TEXT DEFAULT 'ch'
        )
    ''')
    
    # Create index for faster status queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_documents_status 
        ON documents(status)
    ''')
    
    conn.commit()
    conn.close()


@functools.lru_cache(maxsize=1)
def get_all_documents_cached() -> tuple:
    """Cached version of get_all_documents for performance"""
    return tuple(get_all_documents())


def clear_documents_cache():
    """Clear the documents cache"""
    get_all_documents_cached.cache_clear()
    with _folder_tree_cache_lock:
        _folder_tree_cache.clear()


def add_document(filename: str, original_filename: str, file_size: int, 
                 backend: str = 'hybrid-auto-engine', lang: str = 'ch') -> int:
    """Add a new document to the database"""
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        
        upload_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Remove .pdf extension from filename for output folder name
        output_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
        output_path = os.path.join(os.path.dirname(__file__), 'outputs', output_name)
        
        try:
            cursor.execute('''
                INSERT INTO documents (filename, original_filename, upload_time, status, output_path, file_size, backend, lang)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (filename, original_filename, upload_time, 'processing', output_path, file_size, backend, lang))
            
            doc_id = cursor.lastrowid
            if doc_id is None:
                raise RuntimeError("Failed to insert document")
            conn.commit()
            clear_documents_cache()
            return doc_id
        except Exception as e:
            conn.rollback()
            raise e


def update_document_status(doc_id: int, status: str, error_message: Optional[str] = None) -> bool:
    """Update document status with proper error handling"""
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        
        try:
            if error_message:
                cursor.execute('''
                    UPDATE documents SET status = ?, error_message = ? WHERE id = ?
                ''', (status, error_message, doc_id))
            else:
                cursor.execute('''
                    UPDATE documents SET status = ? WHERE id = ?
                ''', (status, doc_id))
            
            if cursor.rowcount == 0:
                return False
                
            conn.commit()
            clear_documents_cache()
            return True
        except Exception as e:
            conn.rollback()
            raise e


def get_document(doc_id: int) -> Optional[dict]:
    """Get document by ID with proper error handling"""
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
        except Exception as e:
            raise e


def get_all_documents() -> List[dict]:
    """Get all documents with proper error handling"""
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM documents ORDER BY upload_time DESC')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            raise e


def delete_document(doc_id: int) -> bool:
    """Delete a document and its files with proper error handling"""
    with _db_lock:
        doc = get_document(doc_id)
        if not doc:
            return False
        
        # Delete output directory if exists
        if doc['output_path'] and os.path.exists(doc['output_path']):
            import shutil
            try:
                shutil.rmtree(doc['output_path'])
            except OSError as e:
                # Log but don't fail if directory deletion fails
                pass
        
        # Delete from database
        conn = _get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
            conn.commit()
            clear_documents_cache()
            return cursor.rowcount > 0
        except Exception as e:
            conn.rollback()
            raise e


def get_document_files(doc_id: int) -> List[dict]:
    """Get list of output files for a document with optimized file listing"""
    doc = get_document(doc_id)
    if not doc or not doc['output_path'] or not os.path.exists(doc['output_path']):
        return []
    
    files = []
    output_path = Path(doc['output_path'])
    
    # Find all files in output directory using os.scandir for better performance
    try:
        for item in output_path.rglob('*'):
            if item.is_file():
                try:
                    rel_path = item.relative_to(output_path)
                    files.append({
                        'name': str(rel_path),
                        'path': str(item),
                        'size': item.stat().st_size
                    })
                except (OSError, PermissionError):
                    # Skip files that can't be accessed
                    continue
    except (OSError, PermissionError):
        # Return empty list if directory can't be read
        pass
    
    return files


def _is_cache_valid(cache_entry: tuple) -> bool:
    """Check if cache entry is still valid"""
    if cache_entry is None:
        return False
    import time
    timestamp, _ = cache_entry
    return (time.time() - timestamp) < _folder_tree_cache_ttl


def get_folder_tree() -> List[dict]:
    """
    Get folder tree structure for all completed documents.
    Returns a nested list of documents with their files.
    Uses caching for improved performance.
    """
    import time
    cache_key = 'folder_tree'
    
    # Check cache first
    with _folder_tree_cache_lock:
        if cache_key in _folder_tree_cache:
            cached_time, cached_tree = _folder_tree_cache[cache_key]
            if _is_cache_valid((cached_time, cached_tree)):
                return cached_tree
    
    # Build tree
    documents = get_all_documents()
    tree = []
    
    for doc in documents:
        if doc['status'] != 'completed':
            continue
            
        doc_tree = {
            'id': doc['id'],
            'name': doc['original_filename'],
            'type': 'folder',
            'path': doc['output_path'],
            'children': []
        }
        
        # Get files for this document
        if doc['output_path'] and os.path.exists(doc['output_path']):
            output_path = Path(doc['output_path'])
            
            # Build nested structure
            try:
                for item in output_path.rglob('*'):
                    if item.is_file():
                        try:
                            rel_path = item.relative_to(output_path)
                            parts = rel_path.parts
                            
                            # Add to tree structure
                            current_level = doc_tree['children']
                            for part in parts[:-1]:
                                # Find or create folder at this level
                                found = None
                                for child in current_level:
                                    if child['name'] == part and child['type'] == 'folder':
                                        found = child
                                        break
                                
                                if found is None:
                                    found = {
                                        'name': part,
                                        'type': 'folder',
                                        'path': str(output_path / part),
                                        'children': []
                                    }
                                    current_level.append(found)
                                
                                current_level = found['children']
                            
                            # Add file at the last level
                            try:
                                file_size = item.stat().st_size
                                current_level.append({
                                    'name': item.name,
                                    'type': 'file',
                                    'path': str(item),
                                    'size': file_size,
                                    'ext': item.suffix.lower()
                                })
                            except (OSError, PermissionError):
                                continue
                        except (OSError, ValueError):
                            continue
            except (OSError, PermissionError):
                pass
        
        # Only add if there are files
        if doc_tree['children'] or os.path.exists(doc['output_path']):
            tree.append(doc_tree)
    
    # Cache the result
    with _folder_tree_cache_lock:
        _folder_tree_cache[cache_key] = (time.time(), tree)
    
    return tree


def get_file_content(file_path: str, max_size: int = 1024 * 1024) -> dict:
    """
    Get file content for preview with path validation.
    Returns metadata and content (truncated if too large).
    """
    # Validate and sanitize the file path to prevent directory traversal
    if not file_path:
        return {'error': 'No file path provided'}
    
    # Resolve the path and check it's within allowed directories
    try:
        abs_path = os.path.abspath(file_path)
        # Only allow files within outputs or uploads directories
        base_dir = os.path.dirname(__file__)
        allowed_dirs = [
            os.path.join(base_dir, 'outputs'),
            os.path.join(base_dir, 'uploads')
        ]
        
        is_allowed = any(abs_path.startswith(os.path.abspath(allowed_dir)) 
                        for allowed_dir in allowed_dirs)
        if not is_allowed:
            return {'error': f'Access denied: File must be in outputs or uploads directory (base: {base_dir})'}
        
    except (ValueError, OSError):
        return {'error': 'Invalid file path'}
    
    if not os.path.exists(abs_path):
        return {'error': 'File not found'}
    
    if not os.path.isfile(abs_path):
        return {'error': 'Not a file'}
    
    file_size = os.path.getsize(abs_path)
    ext = os.path.splitext(abs_path)[1].lower()
    
    # For text-based files, read content
    text_extensions = {'.md', '.txt', '.json', '.xml', '.html', '.css', '.js', '.py'}
    binary_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.zip', '.webp'}
    
    content = None
    if ext in text_extensions:
        try:
            with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_size + 1)
            if len(content) > max_size:
                content = content[:max_size] + '\n... [truncated]'
        except Exception as e:
            content = f'Error reading file: {str(e)}'
    elif ext in binary_extensions:
        content = f'[Binary file: {ext}]'
    else:
        content = '[Unknown file type]'
    
    return {
        'path': abs_path,
        'name': os.path.basename(abs_path),
        'size': file_size,
        'ext': ext,
        'content': content,
        'is_binary': ext in binary_extensions
    }


# Initialize database on module import (but not on import in worker threads)
if __name__ == '__main__' or os.environ.get('FLASK_APP') == 'app':
    init_db()
