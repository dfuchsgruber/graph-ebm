import logging
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Generic, Sequence, TypeVar

from filelock import FileLock
from rich.console import Console
from rich.table import Table
from tinydb import Query, TinyDB
from typeguard import typechecked

K = TypeVar("K")
V = TypeVar("V")


@typechecked
class Registry(Generic[K, V]):
    """Thread-safe registry for storing artifacts using TinyDB."""

    def __init__(self, database_path: str, lockfile_path: str, key_fn=lambda x: x):
        self.database_path = Path(database_path)
        self.lockfile_path = Path(lockfile_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.lockfile_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(lockfile_path)
        self.key_fn = key_fn

    def __getitem__(self, key: K) -> V:
        key = self.key_fn(key)
        with self.lock:
            db = TinyDB(self.database_path)
            query = Query()
            result = db.search(query.key == key)
            if len(result) == 0:
                raise KeyError(key)
            elif len(result) > 1:
                raise RuntimeError(f"Multiple artifacts found for {key}")
            else:
                return result[0]["value"]

    def __setitem__(self, key: K, value: V):
        key = self.key_fn(key)
        with self.lock:
            db = TinyDB(self.database_path)
            query = Query()
            db.remove(query.key == key)
            db.insert(
                {"key": key, "value": value, "timestamp": str(datetime.now(tz=None))}
            )

    def __delitem__(self, key: K):
        key = self.key_fn(key)
        with self.lock:
            db = TinyDB(self.database_path)
            query = Query()
            db.remove(query.key == key)

    def items(self) -> Generator[tuple[K, V], None, None]:
        """Iterates over the items."""
        with self.lock:
            db = TinyDB(self.database_path)
            for item in db:
                yield (item["key"], item["value"])

    def __contains__(self, key: K) -> bool:
        key = self.key_fn(key)
        with self.lock:
            db = TinyDB(self.database_path)
            query = Query()
            return len(db.search(query.key == key)) > 0


T = TypeVar("T")


class StorageRegistry(Registry[K, str], Generic[K, T]):
    """Class for storing artifacts on disk using `torch`."""

    suffix: str = ".pt"

    def __init__(
        self,
        database_path: str,
        lockfile_path: str,
        storage_path: str,
        key_fn=(lambda args: "_".join(map(str, args))),
    ):
        super().__init__(
            database_path=database_path, lockfile_path=lockfile_path, key_fn=key_fn
        )
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def clean(self):
        """Clean the registry by deleting all files that are not referenced"""
        with self.lock:
            db = TinyDB(self.database_path)
            for path in self.storage_path.iterdir():
                if not any(path == Path(entry["value"]) for entry in db):
                    logging.info(f"Deleting {path}")
                    path.unlink()

    def __getitem__(self, key: Any) -> T:
        path = super().__getitem__(key)
        return self.deserialize(Path(path))

    def __setitem__(self, key: K, value: T):
        path = self.generate_path(key)
        self.serialize(value, path)
        super().__setitem__(key, str(path.resolve().absolute()))

    def items(self) -> Generator[tuple[K, T], None, None]:
        for key, path in super().items():
            yield key, self.deserialize(Path(path))

    @staticmethod
    def humaize_filesize(
        bytes: int,
        units: Sequence[str] = (" bytes", "KB", "MB", "GB", "TB", "PB", "EB"),
    ):
        """Returns a human readable string representation of bytes"""
        return (
            str(bytes) + units[0]
            if bytes < 1024
            else StorageRegistry.humaize_filesize(bytes >> 10, units[1:])
        )

    def list(self):
        """Lists the registry elements."""
        table = Table(title="Dataset Registry")
        table.add_column("Key", justify="left")
        table.add_column("Size", justify="right")
        table.add_column("Path", justify="right")
        for key, path in sorted(super().items()):
            path = Path(path)
            if path.exists():
                table.add_row(
                    str(key),
                    StorageRegistry.humaize_filesize(path.stat().st_size),
                    str(path),
                )
            else:
                table.add_row(str(key), "", "Not Found", style="red")
        Console().print(table)

    @abstractmethod
    def serialize(self, value: T, path: Path):
        """Serialize the value to the storage path."""
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, path: Path) -> T:
        """Deserialize the value from the storage path."""
        raise NotImplementedError

    def generate_path(self, key: K) -> Path:
        return self.generate_path_from_str(str(key))

    def generate_path_from_str(self, s: str) -> Path:
        return (self.storage_path / s).with_suffix(self.suffix)
