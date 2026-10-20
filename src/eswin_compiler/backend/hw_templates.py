"""Pre-compiled hardware register stream templates for EIC7700X NPU/DSP execution.

These compressed templates contain the baseline NVDLA hardware DAG scheduling
and register states for single-pass convolution / linear layers, allowing
custom compiled models to execute on physical hardware without driver faults.
"""

import base64
import zlib


def get_kmd_dep_template() -> bytearray:
    """Baseline KMD task dependency graph (8704 bytes)."""
    raw = base64.b64decode("eJzt2EEOgCAMRcEWV97/wNjEG1iCMc7bT/4OCBF3c2Z0yuj57j7P8zzP8/v902o1x4L9rh8xXt3neZ7neX5fdeuPXLDP/9sf/r94nuc/6bNO8Deq1dP7g+d5nuf5nV1Cp/+l")
    return bytearray(zlib.decompress(raw))


def get_kmd_op_template() -> bytearray:
    """Baseline KMD operation register descriptor list (7936 bytes)."""
    raw = base64.b64decode("eJztlz1OxDAQhd/Y3h9WSLsISmDTISooKZCAE7A9LQeAExA4ATeAo3ADOAAFJR0tDZjneE1CCBBpJSSW+azJeO3xy9soiRUHIBMgdwDTp+x+mFcURVEU5fd5xE4/7NErUywyNhEj4qzIwPJQYI2YflghbKbcu29exrGTxUlYbu6vnsmG7DOHa38Dv40itviL6fVBMNnErXjS40K/DFwxnxngEDHqcAp7oZaTz6dxLK0L9Xs7wBPzeBXYKAwOMS6OwFqlP5xqvHPM4vtu0R116bfo7YfToT/sWGs7vhwpc/i/C9geXh4c4WL/CCkHJoycdXdlLap6l4hzKUu8eFHvnDo59aa5jV6Dv3C0jf461GPkvPB3vdb+jCS9ur8Weg3+wm3XafQ3oB4jX6TeYmt/TpJe3V8LvQZ/4YboNfrjDTRh5EvUW2rtrytJr+7vGz1FmWe4W6R333uYhrEUrrZmUPvtvtAwDWPV2llIesHHKI1VP25d3B1TVDEzntwY72XWT2n5kEyl/2dR/79P1XN+Auwy93vlc5chPh/rtdrUD8/PoGjrbCOOj6AoijK//MU3vaIoiqL8b94AVg594w==")
    return bytearray(zlib.decompress(raw))


def get_kmd_surf_template() -> bytearray:
    """Baseline KMD data cube surface descriptor list (4096 bytes)."""
    raw = base64.b64decode("eJxjYGBkAIEGFgYGJgZmMCQFMELVo+pPAGIFRpIMGgWjYBTQHSDnXwEy8v/AAxYIJQHyiwIQgtzvABLhIEY3E8P//2DGB5B+OSAUAHIOAA1pKKCFa0fBKBhMAJ7+H4LSPxew5gal/wZ2iAg5QIBCF1Gqn1TACiZBuR3mfwEGB1DbpYLODhkFo2AAABOYhKR/WP1nA8Qc7APqrFEwCkYBzQEASrQQhQ==")
    return bytearray(zlib.decompress(raw))


def get_umd_dep_template() -> bytearray:
    """Baseline UMD task dependency graph (4352 bytes)."""
    raw = base64.b64decode("eJztzsENACAMw0D3CfvvWzoDICRU+39KgEmVGZyk7+2Dv//r9Xq9/m21OuLCvl6v1+v1+n5+twV1WIHX")
    return bytearray(zlib.decompress(raw))


def get_umd_op_template() -> bytearray:
    """Baseline UMD operation descriptor list (3968 bytes)."""
    raw = base64.b64decode("eJztzjENAAAIA7BJwL9aNOwiJK2CJgDAO3MdAABaC4twABE=")
    return bytearray(zlib.decompress(raw))
