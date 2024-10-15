from typing import TypedDict


class RegistryConfig(TypedDict):
    """Config for a registry."""

    database_path: str
    lockfile_path: str
    storage_path: str
