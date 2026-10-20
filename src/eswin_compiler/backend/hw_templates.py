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


# ==============================================================================
# Multi-Pass Linear / GEMM Templates (3-Pass Tiled M-Concat)
# ==============================================================================

def get_linear3p_kmd_dep_template() -> bytearray:
    """3-pass Linear / GEMM KMD task dependency graph (21760 bytes)."""
    raw = base64.b64decode("eNrt2VsOREAUBNDb3Z77X7AxwwJkWjRxCp8nhJ9KidiS1qMmpdIvC8/zPM/zT/P/Ju1XruwPvf7B8zzP8/rHweS9f9Tef9Q/eJ7neV7/OJiy1o98wv1r94/u14R8f57neZ5/Q//oInK6wfMP9hOe53mev9yX7xLRIH1EKtH+/8tg/+B5nuf51+wfQ0TON3j+yf7B8zzP85f7vtH+MZ60f9T3D/sHz/M8z79l/5husn/wz/az/Yrnef6Rfmy0f8zrmbx/nud5nucvzAeVy3Oj")
    return bytearray(zlib.decompress(raw))


def get_linear3p_kmd_op_template() -> bytearray:
    """3-pass Linear / GEMM KMD operation register descriptor list (19840 bytes)."""
    raw = base64.b64decode("eNrtlzFv00AYhr+7M00qIsVjByIilAEkhkhZMmZgyJixY8cODIzZEiEkGDrkpzDwA8KABANiYWAg4m+0Uqn7+XxRzpGDh0Qpw/NYn8/fOe/d2fH5/CYiMje6OwmlnIovQpnUnAcAAIDjU7c+s34DAACwfgMAAMD+fJVh02qZBnq6KuuWWqM4UyL1C7Y/L611A8u/T50Wj0NqpWM1bqSjm6yyTFZNWYnJVFOE3GnxR56JvJAfZqA0VDeLohFijYuO2xqZjmP6pshjTV7fKOnbfp+ZzX4WtRG3Kb/fazcvZeKS0dx/tRQ9J06cczLwaS//aum7d59TebtMS/exuev7aK0fa/I66L+V9fnRWZ1+ocmnan1X4/kO/SNnVG8G8kuTW/EfYnkbi+/lNvoaw3/139Ub9sp47YcvZe1IY1w3/kvVX1XrJxrndfqPqv9Zrb/QuKzTX6v+iS30W/9f/ihNeRUAHB0TRbKV59GqqDNhTdiua0Zt2BAmyrd/G+eHuIaRLd7lvi5qNA1R1Ze1+/VtbZYvZWbvC9gUVg5wT/6H54rxP+CYW5s5GM+9ecU8MKV5tN7EmvAsAgDgv/Hf+G/8NwDgv/Hf+FfGj/8GAMB/47/x3/hvAPw3/hv/zfjx3wAAvOkBAADgMNwDpXTWrw==")
    return bytearray(zlib.decompress(raw))


def get_linear3p_kmd_surf_template() -> bytearray:
    """3-pass Linear / GEMM KMD data cube surface descriptor list (10240 bytes)."""
    raw = base64.b64decode("eNrtlrEOAiEQBR8rmiOxorXxU/g0//w0ELG5BknUc2cSCA0ECG8HKaiRVB7joGIaIOiwMf+q1gDgtwk1u3P5T+QfYLf5z5P5z+QfYJfpf/3fZeP5/z6x9reoWn1G649pXetgec4X9QvcsP3+i1p7c8nZLX2UYy8j/fxG/sGT/xP+x/+A/336P+J/8Oz/jP/xP+B/n/5f8D949v9FOvf3f2oRNC4H4M+5AzFdGF0=")
    return bytearray(zlib.decompress(raw))


def get_linear3p_umd_dep_template() -> bytearray:
    """3-pass Linear / GEMM UMD task dependency graph (4352 bytes)."""
    raw = base64.b64decode("eNrt1aERACAQA8G8hP77fagBZjDs+TUxSTKz667cxP/tK/bjeZ7/0Z//RkbZj+d5nuf5hy11WIHX")
    return bytearray(zlib.decompress(raw))


def get_linear3p_umd_op_template() -> bytearray:
    """3-pass Linear / GEMM UMD operation descriptor list (3968 bytes)."""
    raw = base64.b64decode("eNrt0AEJAAAAAqAm9H9tOwKdYAIA3KkCAHgzi3AAEQ==")
    return bytearray(zlib.decompress(raw))


def get_linear3p_umd_surf_template() -> bytearray:
    """3-pass Linear / GEMM UMD data cube surface descriptor list (2048 bytes)."""
    raw = base64.b64decode("eNrt08EJACAQA8E5BV92YP99CtqFZpaUkNKxmNTJoLmLiJdV/h/xrQ2j/gBx")
    return bytearray(zlib.decompress(raw))

