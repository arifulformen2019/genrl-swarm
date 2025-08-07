import os
import pickle
import time
import gc
import threading
import sys
import signal
from collections import deque
from typing import Any, Dict, List, Optional, Union
from weakref import WeakValueDictionary

import torch.distributed as dist
from hivemind import DHT, get_dht_time

from genrl.communication.communication import Communication
from genrl.serialization.game_tree import from_bytes, to_bytes

# Constants
class Constants:
    MAX_PEERS = 50
    MAX_MEMORY_MB = 2048
    MAX_OBJECT_SIZE_MB = 50
    CLEANUP_INTERVAL = 30
    MAX_BEAM_SIZE = 10
    EMERGENCY_CLEANUP_THRESHOLD = 0.95
    MAX_BUFFER_SIZE_MB = 100
    MAX_CACHE_SIZE = 50
    MAX_CACHE_BYTES_MB = 200
    TIMEOUT_CAP_SECONDS = 300
    P2PD_WARNING_THRESHOLD_MB = 300
    P2PD_CRITICAL_THRESHOLD_MB = 800

# ✅ SIMPLIFIED: Minimal logging imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class HivemindRendezvouz:
    _STORE = None
    _IS_MASTER = False
    _IS_LAMBDA = False

    @classmethod
    def init(cls, is_master: bool = False) -> None:
        """✅ STREAMLINED: Initialize with timeout"""
        cls._IS_MASTER = is_master
        cls._IS_LAMBDA = bool(os.environ.get("LAMBDA", False))
        
        if cls._STORE is None and cls._IS_LAMBDA:
            try:
                world_size = cls._validate_world_size()
                master_addr = os.environ.get("MASTER_ADDR")
                master_port = os.environ.get("MASTER_PORT")
                
                if not master_addr or not master_port:
                    raise ValueError("MASTER_ADDR and MASTER_PORT must be set")
                
                cls._STORE = dist.TCPStore(
                    host_name=master_addr,
                    port=int(master_port),
                    is_master=is_master,
                    world_size=world_size,
                    wait_for_workers=True,
                    timeout=Constants.TIMEOUT_CAP_SECONDS,
                )
            except Exception as e:
                print(f"Rendezvous init failed: {e}")
                raise

    @classmethod
    def _validate_world_size(cls) -> int:
        """Validate and return world size"""
        try:
            size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
            if size <= 0:
                raise ValueError("World size must be positive")
            return size
        except ValueError as e:
            print(f"Invalid HIVEMIND_WORLD_SIZE: {e}, using default: 1")
            return 1

    @classmethod
    def is_bootstrap(cls) -> bool:
        return cls._IS_MASTER

    @classmethod
    def set_initial_peers(cls, initial_peers: Any) -> None:
        """✅ STREAMLINED: Set peers with size limit"""
        try:
            if cls._STORE is None and cls._IS_LAMBDA:
                cls.init()
                
            if cls._IS_LAMBDA and cls._STORE is not None:
                # Limit peer data size
                if isinstance(initial_peers, list) and len(initial_peers) > Constants.MAX_PEERS:
                    initial_peers = initial_peers[:Constants.MAX_PEERS]
                
                cls._STORE.set("initial_peers", pickle.dumps(initial_peers))
                
        except Exception as e:
            print(f"Set initial peers failed: {e}")
            # Don't raise - allow fallback

    @classmethod
    def get_initial_peers(cls) -> List[Any]:
        """✅ STREAMLINED: Get peers with timeout"""
        try:
            if cls._STORE is None and cls._IS_LAMBDA:
                cls.init()
                
            if not cls._IS_LAMBDA or cls._STORE is None:
                return []
            
            cls._STORE.wait(["initial_peers"], timeout=60)
            peer_bytes = cls._STORE.get("initial_peers")
            return pickle.loads(peer_bytes)
            
        except Exception as e:
            print(f"Get initial peers failed: {e}")
            return []

    @classmethod
    def cleanup_store(cls) -> None:
        """✅ STREAMLINED: Simple cleanup"""
        try:
            if cls._STORE is not None:
                cls._STORE = None
                gc.collect()
        except Exception:
            pass


class HivemindBackend(Communication):
    """✅ COMPATIBLE: Optimized HivemindBackend for older Hivemind versions"""
    
    def __init__(
        self,
        initial_peers: Optional[List[str]] = None,
        timeout: int = 600,
        disable_caching: bool = False,
        beam_size: int = 1000,
        identity_path: Optional[str] = None,
        **kwargs,
    ):
        self.world_size = self._validate_world_size()
        self.timeout = min(timeout, Constants.TIMEOUT_CAP_SECONDS)
        self.bootstrap = HivemindRendezvouz.is_bootstrap()
        
        # ✅ Store identity_path (may not be used if unsupported)
        self.identity_path = identity_path
        
        # ✅ CRITICAL: Cap beam_size to prevent P2PD explosion  
        self.beam_size = min(beam_size, Constants.MAX_BEAM_SIZE)
        self.dht: Optional[DHT] = None

        # ✅ Memory limits configuration
        self.max_memory_mb = kwargs.get('max_memory_mb', Constants.MAX_MEMORY_MB)
        self.max_object_size = kwargs.get('max_object_size', Constants.MAX_OBJECT_SIZE_MB * 1024 * 1024)
        self.emergency_cleanup_threshold = Constants.EMERGENCY_CLEANUP_THRESHOLD

        # ✅ STREAMLINED: Essential memory management only
        self._init_memory_management()

        # ✅ FORCE: Always disable caching
        kwargs["cache_locally"] = False
        kwargs["cache_on_store"] = False

        # ✅ COMPATIBLE: DHT initialization without unsupported kwargs
        try:
            if self.bootstrap:
                self._init_bootstrap_dht(initial_peers, **kwargs)
            else:
                self._init_worker_dht(initial_peers, **kwargs)
        except Exception as e:
            print(f"DHT init failed: {e}")
            raise
            
        self.step_ = 0
        
        # ✅ START: Background cleanup only
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = True
        self._start_cleanup_thread()

    def _validate_world_size(self) -> int:
        """Validate and return world size"""
        try:
            size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
            if size <= 0:
                raise ValueError("World size must be positive")
            return size
        except ValueError as e:
            print(f"Invalid HIVEMIND_WORLD_SIZE: {e}, using default: 1")
            return 1

    def _get_identity_kwargs(self) -> Dict[str, Any]:
        """✅ SAFE: Get identity kwargs only if supported"""
        identity_kwargs: Dict[str, Any] = {}
        
        try:
            if self.identity_path:
                identity_kwargs['identity_path'] = self.identity_path
            else:
                org_id = os.environ.get('ORG_ID')
                if org_id:
                    identity_dir = os.path.expanduser('~/.hivemind_identities')
                    os.makedirs(identity_dir, exist_ok=True)
                    identity_kwargs['identity_path'] = os.path.join(identity_dir, f'identity_{org_id}.pem')
                    print(f"Using identity path: {identity_kwargs['identity_path']}")
        except Exception:
            # Identity not supported in this Hivemind version
            pass
        
        return identity_kwargs

    def _init_memory_management(self) -> None:
        """✅ ENHANCED: Memory management with size tracking"""
        # Message buffers with size tracking
        self.message_buffer: deque = deque()
        self.message_buffer_size = 0
        self.max_buffer_size = Constants.MAX_BUFFER_SIZE_MB * 1024 * 1024
        
        # Operation tracking
        self.operation_counter = 0
        self.last_cleanup = time.time()
        self.cleanup_interval = Constants.CLEANUP_INTERVAL
        
        # Health tracking
        self.consecutive_failures = 0
        self.max_failures = 3  # Aggressive failure limit
        
        # Cache with LRU-like behavior and size tracking
        self.cache: Dict[str, Any] = {}
        self.cache_access_times: Dict[str, float] = {}
        self.cache_size = 0
        self.max_cache_size = Constants.MAX_CACHE_SIZE
        self.max_cache_bytes = Constants.MAX_CACHE_BYTES_MB * 1024 * 1024

        # Track DHT keys for cleanup
        self.active_dht_keys: deque = deque(maxlen=1000)

    def _get_object_size(self, obj: Any) -> int:
        """Calculate object size in bytes"""
        if isinstance(obj, bytes):
            return len(obj)
        return sys.getsizeof(obj)

    def _init_bootstrap_dht(self, initial_peers: Optional[List[str]], **kwargs) -> None:
        """✅ COMPATIBLE: Bootstrap DHT with only supported kwargs"""
        # Only use basic supported parameters
        filtered_kwargs = {
            'cache_locally': False,
            'cache_on_store': False,
        }
        
        # Try to add identity if supported
        try:
            identity_kwargs = self._get_identity_kwargs()
            if identity_kwargs and 'identity_path' in identity_kwargs:
                # Test if identity_path is supported
                filtered_kwargs.update(identity_kwargs)
        except Exception:
            print("Identity path not supported in this Hivemind version")
        
        try:
            self.dht = DHT(
                start=True,
                host_maddrs=[f"/ip4/0.0.0.0/tcp/0"],
                initial_peers=initial_peers,
                **filtered_kwargs,
            )
            
            print(f"Bootstrap DHT initialized with peer ID: {self.dht.peer_id}")
            
            try:
                dht_maddrs = self.dht.get_visible_maddrs(latest=True)
                HivemindRendezvouz.set_initial_peers(dht_maddrs)
            except Exception as e:
                print(f"Bootstrap setup failed: {e}")
                
        except Exception as e:
            print(f"Bootstrap DHT failed: {e}")
            # Fallback: try with minimal kwargs
            self.dht = DHT(
                start=True,
                host_maddrs=[f"/ip4/0.0.0.0/tcp/0"],
                initial_peers=initial_peers,
            )

    def _init_worker_dht(self, initial_peers: Optional[List[str]], **kwargs) -> None:
        """✅ COMPATIBLE: Worker DHT with only supported kwargs"""
        if initial_peers is None:
            try:
                initial_peers = HivemindRendezvouz.get_initial_peers()
            except Exception:
                initial_peers = []
        
        # Only use basic supported parameters
        filtered_kwargs = {
            'cache_locally': False,
            'cache_on_store': False,
        }
        
        # Try to add identity if supported
        try:
            identity_kwargs = self._get_identity_kwargs()
            if identity_kwargs and 'identity_path' in identity_kwargs:
                filtered_kwargs.update(identity_kwargs)
        except Exception:
            print("Identity path not supported in this Hivemind version")
        
        try:
            self.dht = DHT(
                start=True,
                host_maddrs=[f"/ip4/0.0.0.0/tcp/0"],
                initial_peers=initial_peers,
                **filtered_kwargs,
            )
            
            print(f"Worker DHT initialized with peer ID: {self.dht.peer_id}")
            
        except Exception as e:
            print(f"Worker DHT failed: {e}")
            # Fallback: try with minimal kwargs
            self.dht = DHT(
                start=True,
                host_maddrs=[f"/ip4/0.0.0.0/tcp/0"],
                initial_peers=initial_peers,
            )

    def _monitor_p2pd_memory(self) -> None:
        """✅ SAFER: Gentle P2PD memory management without training disruption"""
        if not PSUTIL_AVAILABLE:
            return
            
        try:
            current_time = time.time()
            
            # Check every 3 minutes (less frequent)
            if hasattr(self, '_last_p2pd_check'):
                if current_time - self._last_p2pd_check < 180:
                    return
            
            self._last_p2pd_check = current_time
            
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    if proc.info['name'] and 'p2pd' in proc.info['name']:
                        memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                        
                        if memory_mb > Constants.P2PD_WARNING_THRESHOLD_MB:
                            print(f"⚠️ P2PD memory: {memory_mb:.1f}MB")
                            
                            # Try gentle memory reduction first
                            if memory_mb > 500:
                                self._gentle_p2pd_cleanup(proc)
                            
                            # Only apply pressure in extreme cases, NO RESTART
                            if memory_mb > Constants.P2PD_CRITICAL_THRESHOLD_MB:
                                print(f"🚨 P2PD high memory: {memory_mb:.1f}MB - applying memory pressure")
                                self._apply_memory_pressure(proc)
                                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            print(f"P2PD monitoring error: {e}")

    def _gentle_p2pd_cleanup(self, proc) -> None:
        """Try to reduce p2pd memory without killing it"""
        try:
            # Clear our own cache to reduce p2pd pressure
            if self.dht and len(self.cache) > 10:
                old_size = len(self.cache)
                # Keep only recent cache entries
                current_time = time.time()
                expired_keys = []
                
                for key, access_time in self.cache_access_times.items():
                    if current_time - access_time > 60:  # 1 minute old
                        expired_keys.append(key)
                
                for key in expired_keys[:old_size//2]:  # Remove half of expired
                    if key in self.cache:
                        self.cache_size -= self._get_object_size(self.cache[key])
                        del self.cache[key]
                        del self.cache_access_times[key]
                        
                print(f"✅ Cleaned cache: removed {len(expired_keys)} entries")
                
            # Force garbage collection
            gc.collect()
                
        except Exception as e:
            print(f"Gentle cleanup failed: {e}")

    def _apply_memory_pressure(self, proc) -> bool:
        """Apply memory pressure without killing process"""
        try:
            pid = proc.pid
            
            # Try to reduce process priority to limit memory growth
            try:
                os.system(f"renice +10 {pid} 2>/dev/null")  # Lower CPU priority
                print(f"✅ Reduced p2pd {pid} priority")
            except:
                pass
                
            # Clear our own buffers more aggressively
            if len(self.message_buffer) > 10:
                # Keep only recent messages
                cutoff_time = time.time() - 60  # 1 minute
                while self.message_buffer and self.message_buffer[0][1] < cutoff_time:
                    self.message_buffer.popleft()
                    
                self.message_buffer_size = sum(self._get_object_size(item) for item in self.message_buffer)
                print("✅ Cleared old message buffer")
                
            return True
            
        except Exception as e:
            print(f"Memory pressure failed: {e}")
            return False

    def _start_cleanup_thread(self) -> None:
        """✅ ENHANCED: Background cleanup with safer p2pd monitoring"""
        def cleanup_worker() -> None:
            while self._running:
                try:
                    time.sleep(self.cleanup_interval)
                    self._periodic_cleanup()
                    self._check_memory_usage()
                    self._monitor_p2pd_memory()  # ✅ Monitor p2pd safely
                    
                except Exception as e:
                    print(f"Cleanup error: {e}")
                    time.sleep(60)  # Back off on error
                    
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def _check_memory_usage(self) -> None:
        """Monitor and enforce memory limits"""
        if not PSUTIL_AVAILABLE:
            return
            
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.max_memory_mb * self.emergency_cleanup_threshold:
                print(f"WARNING: Memory usage {memory_mb:.1f}MB approaching limit")
                self._emergency_cleanup()
                
            elif memory_mb > self.max_memory_mb:
                raise MemoryError(f"Memory usage {memory_mb:.1f}MB exceeds limit {self.max_memory_mb}MB")
                
        except psutil.Error:
            pass

    def _periodic_cleanup(self) -> None:
        """✅ ENHANCED: Comprehensive cleanup"""
        try:
            current_time = time.time()
            
            # Skip if cleaned recently
            if current_time - self.last_cleanup < self.cleanup_interval:
                return
            
            # 1. Clean expired messages from buffer
            cutoff_time = current_time - 300  # 5 minutes
            while self.message_buffer and self.message_buffer[0][1] < cutoff_time:
                key, timestamp = self.message_buffer.popleft()
                # Try to delete from DHT if possible
                try:
                    if self.dht and hasattr(self.dht, 'delete'):
                        self.dht.delete(key)
                except:
                    pass
            
            # 2. Clean cache based on LRU and size
            if len(self.cache) > self.max_cache_size or self.cache_size > self.max_cache_bytes:
                # Sort by access time (LRU)
                sorted_keys = sorted(self.cache_access_times.items(), key=lambda x: x[1])
                
                # Remove oldest entries until within limits
                while (len(self.cache) > self.max_cache_size * 0.7 or 
                       self.cache_size > self.max_cache_bytes * 0.7) and sorted_keys:
                    key, _ = sorted_keys.pop(0)
                    if key in self.cache:
                        obj_size = self._get_object_size(self.cache[key])
                        del self.cache[key]
                        del self.cache_access_times[key]
                        self.cache_size -= obj_size
            
            # 3. Recalculate buffer size
            self.message_buffer_size = sum(self._get_object_size(item) for item in self.message_buffer)
            
            # 4. Force garbage collection periodically
            if self.operation_counter % 100 == 0:
                gc.collect()
            
            self.last_cleanup = current_time
            
        except Exception as e:
            print(f"Periodic cleanup error: {e}")

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup when memory is critical"""
        print("EMERGENCY CLEANUP: Clearing all non-essential data")
        
        try:
            # Clear all caches
            self.cache.clear()
            self.cache_access_times.clear()
            self.cache_size = 0
            
            # Clear message buffer
            self.message_buffer.clear()
            self.message_buffer_size = 0
            
            # Clear DHT keys tracking
            self.active_dht_keys.clear()
            
            # Reset counters
            self.consecutive_failures = 0
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Emergency cleanup error: {e}")

    def all_gather_object(self, obj: Any) -> Dict[Union[str, int], Any]:
        """✅ ENHANCED: Training-safe all_gather with better error handling"""
        key = str(self.step_)
        max_retries = 2  # Reduced retries to avoid long delays
        
        for retry in range(max_retries):
            try:
                self.operation_counter += 1
                
                # ✅ PERIODIC CLEANUP
                if self.operation_counter % 20 == 0:
                    self._periodic_cleanup()
                
                # ✅ SERIALIZE WITH SIZE CHECK
                try:
                    obj_bytes = to_bytes(obj)
                    obj_size = len(obj_bytes)
                    
                    # Check object size limit
                    if obj_size > self.max_object_size:
                        raise ValueError(f"Object too large: {obj_size/1024/1024:.1f}MB exceeds limit {self.max_object_size/1024/1024:.1f}MB")
                    
                    # Warn about large objects
                    if obj_size > 10 * 1024 * 1024:  # 10MB
                        print(f"Warning: Large object {obj_size/1024/1024:.1f}MB")
                    
                except Exception as e:
                    print(f"Serialization failed (attempt {retry+1}): {e}")
                    if retry == max_retries - 1:
                        return {str(self.dht.peer_id): obj} if self.dht else {"unknown": obj}
                    time.sleep(1)
                    continue
                
                # ✅ DHT STORE with tracking
                try:
                    if self.dht is None:
                        raise RuntimeError("DHT not initialized")
                        
                    self.dht.store(
                        key,
                        subkey=str(self.dht.peer_id),
                        value=obj_bytes,
                        expiration_time=get_dht_time() + self.timeout,
                        beam_size=self.beam_size,
                    )
                    
                    # Track in buffer with size limit
                    if self.message_buffer_size + obj_size > self.max_buffer_size:
                        # Remove oldest entries
                        while self.message_buffer and self.message_buffer_size + obj_size > self.max_buffer_size:
                            old_key, _ = self.message_buffer.popleft()
                            # Approximate size reduction
                            self.message_buffer_size = max(0, self.message_buffer_size - self.max_object_size // 10)
                    
                    self.message_buffer.append((key, time.time()))
                    self.message_buffer_size += obj_size
                    self.active_dht_keys.append(key)
                    
                except Exception as e:
                    print(f"DHT store failed (attempt {retry+1}): {e}")
                    if retry == max_retries - 1:
                        self.consecutive_failures += 1
                        return {str(self.dht.peer_id): obj} if self.dht else {"unknown": obj}
                    time.sleep(2)  # Wait before retry
                    continue
                
                # ✅ RETRIEVE WITH TIMEOUT
                time.sleep(0.5)  # Brief wait
                
                start_time = time.monotonic()
                max_wait = min(self.timeout, 60)  # Reduced to 60 seconds
                
                results: Dict[str, Any] = {}
                while True:
                    try:
                        if self.dht is None:
                            break
                            
                        output_, _ = self.dht.get(key, beam_size=self.beam_size, latest=True)
                        
                        elapsed = time.monotonic() - start_time
                        
                        # Process results as they come in
                        for peer_key, value in output_.items():
                            if peer_key not in results:
                                try:
                                    deserialized = from_bytes(value.value)
                                    results[peer_key] = deserialized
                                    
                                    # Cache with size tracking
                                    if len(self.cache) < self.max_cache_size:
                                        cache_key = f"{key}_{peer_key}"
                                        self.cache[cache_key] = deserialized
                                        self.cache_access_times[cache_key] = time.time()
                                        self.cache_size += self._get_object_size(deserialized)
                                        
                                except Exception:
                                    continue  # Skip failed deserialization
                        
                        if len(results) >= self.world_size or elapsed > max_wait:
                            break
                            
                        time.sleep(0.5)  # Short wait between retries
                        
                    except Exception as e:
                        print(f"DHT get failed (attempt {retry+1}): {e}")
                        break
                
                # Ensure self is included
                if self.dht and str(self.dht.peer_id) not in results:
                    results[str(self.dht.peer_id)] = obj
                
                # Clean up this operation's data from DHT if possible
                try:
                    if self.dht and hasattr(self.dht, 'delete'):
                        self.dht.delete(key, subkey=str(self.dht.peer_id))
                except:
                    pass
                
                self.step_ += 1
                
                # Reset failure counter on success
                if len(results) > 1:
                    self.consecutive_failures = 0
                
                return dict(sorted(results.items()))
                
            except (BlockingIOError, EOFError) as io_error:
                print(f"I/O error in all_gather (attempt {retry+1}): {io_error}")
                if retry < max_retries - 1:
                    time.sleep(2 ** retry)  # Exponential backoff
                    continue
                    
            except Exception as e:
                print(f"Unexpected error in all_gather (attempt {retry+1}): {e}")
                if retry < max_retries - 1:
                    time.sleep(2 ** retry)  # Exponential backoff  
                    continue
        
        # All retries failed - return fallback
        print("❌ All all_gather retries failed - using fallback")
        self.consecutive_failures += 1
        
        # Emergency cleanup after multiple failures
        if self.consecutive_failures >= self.max_failures:
            print("Too many failures, emergency cleanup")
            self._emergency_cleanup()
            self.consecutive_failures = 0
        
        # Check memory on any error
        self._check_memory_usage()
        
        return {str(self.dht.peer_id): obj} if self.dht else {"unknown": obj}

    def get_id(self) -> str:
        """✅ SAFE: Get peer ID"""
        try:
            return str(self.dht.peer_id) if self.dht else "unknown"
        except Exception:
            return "error"

    def get_p2pd_stats(self) -> Dict[str, Any]:
        """✅ Get p2pd memory statistics"""
        if not PSUTIL_AVAILABLE:
            return {}
            
        try:
            p2pd_stats = []
            total_p2pd_memory = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
                try:
                    if (proc.info['name'] and 'p2pd' in proc.info['name']) or \
                       (proc.info['cmdline'] and any('p2pd' in cmd for cmd in proc.info['cmdline'])):
                        
                        memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                        total_p2pd_memory += memory_mb
                        
                        p2pd_stats.append({
                            'pid': proc.info['pid'],
                            'memory_mb': memory_mb,
                            'cmdline': ' '.join(proc.info['cmdline'][:3]) if proc.info['cmdline'] else 'unknown'
                        })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            return {
                'p2pd_processes': p2pd_stats,
                'total_p2pd_memory_mb': total_p2pd_memory,
                'p2pd_count': len(p2pd_stats)
            }
            
        except Exception as e:
            return {'error': str(e)}

    def cleanup(self) -> None:
        """✅ ENHANCED: Comprehensive cleanup"""
        try:
            self._running = False
            
            # Wait for cleanup thread to finish
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                try:
                    self._cleanup_thread.join(timeout=5)
                except:
                    pass
            
            # Clear all data structures
            self.cache.clear()
            self.cache_access_times.clear()
            self.cache_size = 0
            
            self.message_buffer.clear()
            self.message_buffer_size = 0
            
            self.active_dht_keys.clear()
            
            # Shutdown DHT
            if self.dht:
                try:
                    self.dht.shutdown()
                except Exception as e:
                    print(f"DHT shutdown error: {e}")
                finally:
                    self.dht = None
            
            # Clean up rendezvous
            HivemindRendezvouz.cleanup_store()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Cleanup error: {e}")

    def __del__(self) -> None:
        """✅ DESTRUCTOR: Cleanup on destruction"""
        try:
            self.cleanup()
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed stats for monitoring"""
        try:
            stats = {
                'operations': self.operation_counter,
                'failures': self.consecutive_failures,
                'cache_entries': len(self.cache),
                'cache_size_mb': self.cache_size / 1024 / 1024,
                'buffer_entries': len(self.message_buffer),
                'buffer_size_mb': self.message_buffer_size / 1024 / 1024,
                'active_dht_keys': len(self.active_dht_keys),
                'beam_size': self.beam_size,
                'world_size': self.world_size,
                'timeout': self.timeout,
                'max_memory_mb': self.max_memory_mb,
                'running': self._running,
            }
            
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    stats.update({
                        'memory_rss_mb': memory_info.rss / 1024 / 1024,
                        'memory_vms_mb': memory_info.vms / 1024 / 1024,
                        'memory_percent': process.memory_percent(),
                        'cpu_percent': process.cpu_percent(),
                    })
                except psutil.Error as e:
                    stats['psutil_error'] = str(e)
                
            # Add P2PD stats
            p2pd_stats = self.get_p2pd_stats()
            if p2pd_stats:
                stats.update(p2pd_stats)
            
            return stats
            
        except Exception as e:
            return {'error': str(e), 'fallback_stats': True}
