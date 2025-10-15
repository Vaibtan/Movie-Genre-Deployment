import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Abstract base class for storage backends
class StorageBackend(ABC):    
    @abstractmethod
    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        pass
    
    @abstractmethod
    def download_file(self, remote_path: str, local_path: Path) -> bool:
        pass
    
    @abstractmethod
    def file_exists(self, remote_path: str) -> bool:
        pass
    
    @abstractmethod
    def list_files(self, prefix: str) -> List[str]:
        pass


class LocalStorage(StorageBackend):    
    def __init__(self, base_path: Path = Path("artifacts")):
        self.base_path: Path = base_path
        self.base_path.mkdir(parents = True, exist_ok = True)
    
    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        try:
            dest_path: Path = self.base_path / remote_path
            dest_path.parent.mkdir(parents = True, exist_ok = True)
            if local_path != dest_path: shutil.copy2(local_path, dest_path)
            return True
        except Exception as e: logger.error(f"Local upload failed: {e}"); return False
    
    def download_file(self, remote_path: str, local_path: Path) -> bool:
        try:
            source_path: Path = self.base_path / remote_path
            if not source_path.exists(): return False
            local_path.parent.mkdir(parents = True, exist_ok = True)
            if source_path != local_path: shutil.copy2(source_path, local_path)
            return True
        except Exception as e: logger.error(f"Local download failed: {e}"); return False
    
    def file_exists(self, remote_path: str) -> bool:
        return (self.base_path / remote_path).exists()
    
    def list_files(self, prefix: str) -> List[str]:
        prefix_path: Path = self.base_path / prefix
        if not prefix_path.exists(): return []
        files: List[str] = []
        if prefix_path.is_file(): files.append(prefix)
        else:
            for file_path in prefix_path.rglob("*"):
                if file_path.is_file():
                    rel_path: Path = file_path.relative_to(self.base_path)
                    files.append(str(rel_path))
        return files


class S3Storage(StorageBackend):    
    def __init__(self, bucket_name: str, region: Optional[str] = None, \
        prefix: str = "movie-genre-artifacts") -> None:
        self.bucket_name: str = bucket_name
        self.prefix: str = prefix.strip("/")
        self.region: str = region or os.getenv("AWS_REGION", "us-east-1")
        self.s3_client = boto3.client("s3", region_name = self.region)
        try:
            self.s3_client.head_bucket(Bucket = self.bucket_name)
            logger.info(f"✅ Connected to S3 bucket: {self.bucket_name}")
        except ClientError as e:
            logger.error(f"Cannot access S3 bucket {self.bucket_name}: {e}")
            raise
        
    # Construct full S3 key with prefix
    def get_s3_key(self, remote_path: str) -> str:
        remote_path = remote_path.lstrip("/")
        if self.prefix: return f"{self.prefix}/{remote_path}"
        return remote_path
    
    # Upload file to S3 with progress tracking
    def upload_file(self, local_path: Path, remote_path: str) -> bool:        
        if not local_path.exists():
            logger.error(f"Local file not found: {local_path}")
            return False
        s3_key: str = self.get_s3_key(remote_path)
        try:
            file_size: int = local_path.stat().st_size
            # Use multipart upload for files > 100MB
            if file_size > 100 * 1024 * 1024: self.multipart_upload(local_path, s3_key, file_size)
            else: self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
            logger.info(f"✅ Uploaded to S3: {remote_path} ({file_size/1024/1024:.2f} MB)")
            return True    
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            return False
    
    def multipart_upload(self, local_path: Path, s3_key: str, file_size: int) -> None:
        chunk_size: int = 10 * 1024 * 1024  # 10MB chunks
        self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key, \
            Config = self.get_transfer_config(chunk_size))
    
    # Get transfer configuration for efficient uploads
    def get_transfer_config(self, chunk_size: int):
        return TransferConfig(multipart_threshold = chunk_size, \
            multipart_chunksize = chunk_size, max_concurrency = 10, use_threads = True)
    
    # Download file from S3 with caching
    def download_file(self, remote_path: str, local_path: Path) -> bool:        
        s3_key: str = self.get_s3_key(remote_path)
        try:
            local_path.parent.mkdir(parents = True, exist_ok = True)
            # Download file
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            file_size: int = local_path.stat().st_size
            logger.info(f"✅ Downloaded from S3: {remote_path} ({file_size/1024/1024:.2f} MB)")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.warning(f"File not found in S3: {remote_path}")
            else: logger.error(f"S3 download failed: {e}")
            return False
    
    def file_exists(self, remote_path: str) -> bool:        
        s3_key: str = self.get_s3_key(remote_path)
        try:
            self.s3_client.head_object(Bucket = self.bucket_name, Key = s3_key)
            return True
        except ClientError: return False
    
    def list_files(self, prefix: str) -> List[str]:        
        s3_prefix: str = self.get_s3_key(prefix)
        files: List[str] = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket = self.bucket_name, Prefix = s3_prefix)
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Remove prefix to get relative path
                        key: str = obj['Key']
                        if self.prefix and key.startswith(self.prefix + "/"):
                            key = key[len(self.prefix) + 1:]
                        files.append(key)
        except ClientError as e: logger.error(f"S3 list failed: {e}")
        return files

# Unified storage manager with automatic fallback and caching
class StorageManager:
    def __init__(self, backend: StorageBackend, cache_dir: Path = \
        Path(".cache/artifacts"), use_cache: bool = True) -> None:
        self.backend: StorageBackend = backend
        self.cache_dir: Path = cache_dir
        self.use_cache: bool = use_cache
        if self.use_cache: self.cache_dir.mkdir(parents = True, exist_ok = True)
    
    def save_artifact(self, local_path: Path, remote_path: str) -> bool:
        return self.backend.upload_file(local_path, remote_path)
    
    # Load artifact from storage with caching
    def load_artifact(self, remote_path: str, local_path: Optional[Path] = None) -> Optional[Path]:
        # Determine download destination
        if local_path is None:
            if self.use_cache: local_path = self.cache_dir / remote_path
            else: local_path = Path(remote_path)
        if local_path.exists():
            logger.info(f"✅ Using cached artifact: {remote_path}")
            return local_path
        if self.backend.download_file(remote_path, local_path): return local_path
        return None
    
    def artifact_exists(self, remote_path: str) -> bool:
        return self.backend.file_exists(remote_path)
    
    def list_artifacts(self, prefix: str = "") -> List[str]:
        return self.backend.list_files(prefix)
    
    # Create StorageManager from environment variables
    @classmethod
    def create_from_env(cls, cache_dir: Path = Path(".cache/artifacts")) -> "StorageManager":
        """
        env variables:
            STORAGE_BACKEND: 'local' or 's3' (default: local)
            S3_BUCKET_NAME: S3 bucket name (required if backend=s3)
            S3_PREFIX: S3 key prefix (default: movie-genre-artifacts)
            AWS_REGION: AWS region (default: us-east-1)
            LOCAL_ARTIFACTS_DIR: Local storage directory (default: artifacts)
        @returns: Configured StorageManager instance
        """
        backend_type: str = os.getenv("STORAGE_BACKEND", "local").lower()
        if backend_type == "s3":
            bucket_name: Optional[str] = os.getenv("S3_BUCKET_NAME")
            if not bucket_name: raise ValueError("S3_BUCKET_NAME environment variable required for S3 storage")
            backend: StorageBackend = S3Storage(bucket_name = bucket_name, \
                region = os.getenv("AWS_REGION"), prefix = os.getenv("S3_PREFIX", "movie-genre-artifacts"))
            logger.info(f"🗄️  Using S3 storage: s3://{bucket_name}")
        else:
            local_dir: Path = Path(os.getenv("LOCAL_ARTIFACTS_DIR", "artifacts"))
            backend = LocalStorage(base_path = local_dir)
            logger.info(f"🗄️  Using local storage: {local_dir}")
        return cls(backend = backend, cache_dir = cache_dir, use_cache = True)