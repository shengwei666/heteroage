import os
import joblib
import pandas as pd
import pickle
import logging
from typing import Any, Optional, Dict, Callable

# Use the standard project logger
logger = logging.getLogger(__name__)

class BaseArtifact:
    """
    Base class for artifact management.
    Provides automated serialization and deserialization with format inference.
    """
    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir
        try:
            os.makedirs(self.artifact_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create artifact directory {artifact_dir}: {e}")
            raise

    def _get_path(self, filename: str) -> str:
        return os.path.join(self.artifact_dir, filename)

    def save(self, name: str, data: Any, file_format: Optional[str] = None) -> None:
        """
        Serializes and saves a data object as an artifact.

        Args:
            name: Identifier for the artifact.
            data: Object to be persisted.
            file_format: Explicit format ('csv', 'pkl', 'joblib'). Inferred if None.
        """
        # Automatic format inference
        if file_format is None:
            if isinstance(data, pd.DataFrame):
                file_format = 'csv'
            elif hasattr(data, 'fit') and hasattr(data, 'predict'):
                file_format = 'joblib'
            else:
                file_format = 'pkl'

        filename = name if name.endswith(f".{file_format}") else f"{name}.{file_format}"
        path = self._get_path(filename)
        
        logger.info(f"Saving artifact: {filename}")

        try:
            if file_format == 'csv':
                data.to_csv(path, index=False)
            elif file_format == 'joblib':
                joblib.dump(data, path)
            elif file_format == 'pkl':
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise ValueError(f"Unsupported serialization format: {file_format}")
        except Exception as e:
            logger.error(f"Failed to save artifact {name}: {e}")
            raise

    def load(self, name: str, file_format: Optional[str] = None) -> Any:
        """
        Loads and deserializes an artifact.

        Args:
            name: Identifier or full filename of the artifact.
            file_format: Explicit format. Inferred from extension if None.
        """
        path = self._get_path(name)
        
        # Extension hunting if path does not exist
        if not os.path.exists(path):
            for ext in ['pkl', 'joblib', 'csv']:
                test_path = f"{path}.{ext}"
                if os.path.exists(test_path):
                    path = test_path
                    file_format = ext
                    break
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Artifact {name} not found in {self.artifact_dir}")

        # Infer format from file extension
        if file_format is None:
            ext = os.path.splitext(path)[1].lower().strip('.')
            file_format = ext

        logger.info(f"Loading artifact: {os.path.basename(path)}")

        try:
            if file_format == 'csv':
                return pd.read_csv(path)
            elif file_format == 'joblib':
                return joblib.load(path)
            elif file_format == 'pkl':
                with open(path, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"Unrecognized format '{file_format}' for artifact: {path}")
        except Exception as e:
            logger.error(f"Failed to load artifact {name}: {e}")
            raise