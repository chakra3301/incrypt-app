#!/usr/bin/env python3
"""
gpt1_pointillist_stego.py  (c) 2025 • CC-0

Encode the entire GPT-1 checkpoint inside the LSBs of a freshly-painted
pointillist canvas that loosely resembles a small source portrait.

▶ Encode:
    python gpt1_pointillist_stego.py encode portrait.png out.png  --bits 8

▶ Verify & hash-compare:
    python gpt1_pointillist_stego.py verify out.png --bits 8

Depends: pillow, numpy, tqdm, zstandard, torch>=2.2, transformers
"""

import argparse, math, json, os, sys, hashlib, signal
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
import zstandard as zstd
import torch
from transformers import OpenAIGPTModel

Image.MAX_IMAGE_PIXELS = None

# ---------- helpers ------------------------------------------------------------
def graceful_exit(sig, frame): sys.exit(1)
signal.signal(signal.SIGINT, graceful_exit)

def sha256_stream(chunks: Iterator[bytes]) -> str:
    h = hashlib.sha256()
    for blk in chunks: h.update(blk)
    return h.hexdigest()

def capacity(w: int, h: int, bits: int) -> int:   # bytes a canvas can carry
    if bits == 0: return 0
    return (w * h * 3 * bits) // 8

# ---------- GPT-1 download / cache --------------------------------------------
def gpt1_flat(cache: Path) -> Tuple[bytes, List[Tuple[int, ...]]]:
    cache.mkdir(parents=True, exist_ok=True)
    bin_f  = cache / "gpt1_f32.bin"
    shape_f= cache / "gpt1_shapes.json"
    if bin_f.exists() and shape_f.exists():
        return bin_f.read_bytes(), json.loads(shape_f.read_text())
    print("⏬  Downloading GPT-1 (first run only)…")
    m = OpenAIGPTModel.from_pretrained("openai-gpt", torch_dtype=torch.float32)
    shapes, flat_tensors = [], []
    for t in m.state_dict().values():
        shapes.append(tuple(t.shape))
        flat_tensors.append(t.reshape(-1).cpu())
    arr = torch.cat(flat_tensors).numpy().astype(np.float32)
    bin_f.write_bytes(arr.tobytes()); shape_f.write_text(json.dumps(shapes))
    return arr.tobytes(), shapes

# ---------- pointillist canvas -------------------------------------------------
def paint_canvas(src: Image.Image, side: int, dots: int, jitter: int = 6,
                 dot_r: int = 1) -> np.ndarray:
    src_np = np.asarray(src.convert("RGB"), np.uint8)
    sh, sw = src_np.shape[:2]
    canvas_data = np.full((side, side, 3), 255, np.uint8)

    rng = np.random.default_rng()
    # Generate random coordinates normalized to [0,1)
    xs = rng.random(dots); ys = rng.random(dots)

    def bilinear(xf, yf):
        u, v = xf*(sw-1), yf*(sh-1) # Map normalized coords to source image coords
        x0, y0 = int(u), int(v)
        x1, y1 = min(x0+1, sw-1), min(y0+1, sh-1) # Clamp to image bounds
        a, b = u-x0, v-y0 # Interpolation weights

        c00, c10 = src_np[y0, x0], src_np[y0, x1]
        c01, c11 = src_np[y1, x0], src_np[y1, x1]
        
        # Perform bilinear interpolation and ensure result is uint8
        color = (1-b)*((1-a)*c00 + a*c10) + b*((1-a)*c01 + a*c11)
        return color.astype(np.uint8)

    for x_norm, y_norm in zip(tqdm(xs, desc="🎨 Painting canvas", unit="dot", smoothing=0.1), ys):
        # Scale normalized coordinates to canvas coordinates
        cx, cy = int(x_norm*side), int(y_norm*side)
        
        base_col = bilinear(x_norm, y_norm)
        # Add jitter, using int16 to prevent overflow during addition before clipping
        noise = rng.integers(-jitter, jitter+1, 3, dtype=np.int16)
        col_with_jitter = base_col.astype(np.int16) + noise
        final_col = np.clip(col_with_jitter, 0, 255).astype(np.uint8)
        
        # Define dot boundaries, ensuring they are within the canvas
        ymin, ymax = max(0, cy - dot_r), min(side, cy + dot_r + 1)
        xmin, xmax = max(0, cx - dot_r), min(side, cx + dot_r + 1)
        
        if ymin < ymax and xmin < xmax: # Check if the slice is valid (non-empty)
            canvas_data[ymin:ymax, xmin:xmax] = final_col
            
    return canvas_data

# ---------- stego core ---------------------------------------------------------
def embed(payload: bytes, cover_canvas: np.ndarray, bits: int) -> Image.Image:
    # cover_canvas is the numpy array from paint_canvas; it will be modified in-place.
    
    p_bits = np.unpackbits(np.frombuffer(payload, np.uint8))
    num_payload_bits_total = len(p_bits)

    # flat_view_of_cover is a view into cover_canvas's data.
    # Modifications to flat_view_of_cover will modify cover_canvas.
    flat_view_of_cover = cover_canvas.reshape(-1, 3) 
    
    clear_mask = 0xFF ^ ((1 << bits) - 1) # e.g., bits=2, mask=0b11111100
    flat_view_of_cover &= clear_mask # Clear LSBs in place

    if num_payload_bits_total == 0: # No data to embed
        return Image.fromarray(cover_canvas, "RGB")

    for b_val in range(bits): # For each bit plane (0th LSB, 1st LSB, ...)
        payload_bits_for_this_plane = p_bits[b_val::bits] # 1D array of bits (0 or 1)
        len_payload_bits_for_plane = len(payload_bits_for_this_plane)

        if len_payload_bits_for_plane == 0:
            continue # No more payload bits for this plane or subsequent planes
            
        # Create a temporary array of the same shape as flat_view_of_cover, filled with zeros.
        bits_to_embed_shaped = np.zeros_like(flat_view_of_cover, dtype=np.uint8)
        
        # Place the payload_bits_for_this_plane (after shifting by b_val) into the temporary array.
        # These bits go into the first 'len_payload_bits_for_plane' locations of the flattened temp array.
        bits_to_embed_shaped.ravel()[:len_payload_bits_for_plane] = \
            (payload_bits_for_this_plane.astype(np.uint8) << b_val)
        
        flat_view_of_cover |= bits_to_embed_shaped # Embed bits into cover_canvas
        
    return Image.fromarray(cover_canvas, "RGB")

def extract(stego: Image.Image, bits: int) -> bytes:
    arr_flat_channels = np.asarray(stego.convert("RGB"), np.uint8).reshape(-1) 

    payload_cap_bytes = capacity(stego.width, stego.height, bits)
    total_payload_bits = payload_cap_bytes * 8
    
    if total_payload_bits == 0:
        return b""

    bits_arr = np.zeros(total_payload_bits, dtype=np.uint8)
    
    num_channels_to_process_per_plane = total_payload_bits // bits
    if num_channels_to_process_per_plane > arr_flat_channels.size:
        # This case should ideally not happen if capacity is calculated correctly
        # and matches image dimensions. It means we expect to extract more bits
        # than there are channels in the image. Truncate to available channels.
        num_channels_to_process_per_plane = arr_flat_channels.size


    for b_val in range(bits):
        if num_channels_to_process_per_plane == 0:
             break 
        # Extract the b_val-th LSB from the initial segment of channel values
        lsb_plane_data = (arr_flat_channels[:num_channels_to_process_per_plane] >> b_val) & 1
        bits_arr[b_val::bits] = lsb_plane_data
        
    return np.packbits(bits_arr).tobytes()

# ---------- CLI ---------------------------------------------------------------
def encode(args):
    raw, shapes = gpt1_flat(Path(args.cache))
    comp = zstd.ZstdCompressor(level=args.zstd, threads=os.cpu_count() or 1).compress(raw)
    size_bytes = len(comp)
    print(f"✔  GPT-1 compressed → {size_bytes/1e6:.1f} MB (Zstd-{args.zstd})")

    bits = args.bits
    if bits == 0:
        print("Error: --bits cannot be 0 for encoding.")
        sys.exit(1)

    # Calculate minimum pixels needed for the compressed data
    # Bytes per pixel for payload = (3 channels * bits_per_channel) / 8 bits_per_byte
    bytes_per_pixel_payload = (3 * bits) / 8.0
    if bytes_per_pixel_payload == 0: # Should be caught by bits == 0 check
        min_total_pixels = float('inf') 
    else:
        min_total_pixels = math.ceil(size_bytes / bytes_per_pixel_payload)
    
    if min_total_pixels == 0 and size_bytes > 0: # If size is >0 but somehow min_total_pixels is 0
        min_total_pixels = 1 # Need at least one pixel to store anything

    side = math.ceil(math.sqrt(min_total_pixels)) if min_total_pixels > 0 else 0
    if side > 0 : side += side % 2 # make it even for nice dot symmetry, if side is positive
    else: side = 2 # Minimum even side if no pixels were theoretically needed (e.g. empty payload)
    
    canvas_cap_bytes = capacity(side, side, bits)
    print(f"📐  Canvas {side}×{side} for {bits}-LSBs (target capacity "
          f"{canvas_cap_bytes/1e6:.1f} MB)")

    if size_bytes > canvas_cap_bytes:
        # This might happen if side calculation leads to a canvas smaller than required.
        # Attempt to increase side.
        print(f"Warning: Initial calculated canvas {side}x{side} with capacity {canvas_cap_bytes/1e6:.1f}MB is too small for {size_bytes/1e6:.1f}MB payload.")
        while size_bytes > canvas_cap_bytes:
            side += 2 # Increase by 2 to keep it even
            canvas_cap_bytes = capacity(side, side, bits)
            if side > 32767 : # PIL image dimension limit might be an issue
                raise RuntimeError(f"Failed to find a suitable canvas size. Current attempt {side}x{side} still too small or excessively large.")
        print(f"Adjusted canvas to {side}×{side} with capacity {canvas_cap_bytes/1e6:.1f} MB.")


    payload_to_embed = comp + b"\0" * (canvas_cap_bytes - size_bytes)

    dots = args.dots or max(1, side*side//16) # Ensure at least 1 dot if side is very small
    painted_canvas_array = paint_canvas(Image.open(args.src), side, dots,
                                        jitter=args.jitter, dot_r=args.radius)

    print("🔐 embedding…")
    stego_img = embed(payload_to_embed, painted_canvas_array, bits)
    
    stego_img.save(args.out, format="PNG", compress_level=9)
    Path(args.out).with_suffix(".shapes.json").write_text(json.dumps(shapes))
    print(f"✅  wrote {args.out}  ({Path(args.out).stat().st_size/1e6:.1f} MB)")

def verify(args):
    png = Image.open(args.png)
    extracted_payload = extract(png, args.bits)
    
    try:
        # The extracted payload includes padding to fill capacity.
        # Zstd decompressor should handle trailing garbage if the actual compressed stream is shorter.
        # However, it's better if Zstd knows the exact size or if the payload doesn't have non-Zstd garbage.
        # For now, we assume decompress will work.
        decompressed = zstd.ZstdDecompressor().decompress(extracted_payload)
    except zstd.ZstdError as e:
        print(f"❌ Zstd Decompression Error: {e}")
        print("This might indicate data corruption or incorrect extraction.")
        h_extracted = hashlib.sha256(extracted_payload).hexdigest()
        print(f"SHA256 of extracted (potentially corrupt) payload: {h_extracted}")
        sys.exit(1)

    ref_raw, _ = gpt1_flat(Path(args.cache))

    if sha256_stream([decompressed]) == sha256_stream([ref_raw]):
        print("✅  Hashes match — PNG carries the exact checkpoint.")
    else:
        print("❌  Hash mismatch.")
        h_decomp = sha256_stream([decompressed])
        h_ref = sha256_stream([ref_raw])
        print(f"Hash of decompressed data from PNG: {h_decomp}")
        print(f"Hash of reference GPT-1 data    : {h_ref}")
        if len(decompressed) != len(ref_raw):
            print(f"Length mismatch: Decompressed {len(decompressed)} bytes, Reference {len(ref_raw)} bytes")
        # Optionally, save the problematic decompressed file for inspection
        # Path("decompressed_debug.bin").write_bytes(decompressed)
        # Path("extracted_payload_debug.bin").write_bytes(extracted_payload)
        sys.exit(1)


def make_parser():
    p = argparse.ArgumentParser(prog="gpt1_pointillist_stego",
                                description="Hide GPT-1 inside a painted canvas")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode", help="Encode GPT-1 into an image")
    e.add_argument("src",   help="Path to the small source portrait image (e.g., PNG, JPG)")
    e.add_argument("out",   help="Path for the output steganographic PNG image")
    e.add_argument("--bits", type=int, default=8, choices=range(1,9), 
                   help="Number of LSBs per color channel to use for encoding (1-8, default: 8)")
    e.add_argument("--zstd", type=int, default=22, choices=range(1,23), 
                   help="Zstandard compression level (1-22, default: 22)")
    e.add_argument("--dots", type=int, help="Number of dots for pointillist painting (default: auto based on canvas size)")
    e.add_argument("--jitter", type=int, default=6, help="Color noise ±value for dots (default: 6)")
    e.add_argument("--radius", type=int, default=1, help="Dot half-size (radius, default: 1)")
    e.add_argument("--cache", default=os.path.join(os.path.expanduser("~"), ".cache", "gpt1_stego"), 
                   help="Directory for caching the GPT-1 checkpoint (default: ~/.cache/gpt1_stego)")
    
    v = sub.add_parser("verify", help="Verify if an image contains the correct GPT-1 checkpoint")
    v.add_argument("png", help="Path to the steganographic PNG image to verify")
    v.add_argument("--bits", type=int, required=True, choices=range(1,9), 
                   help="Number of LSBs per color channel used during encoding (1-8)")
    v.add_argument("--cache", default=os.path.join(os.path.expanduser("~"), ".cache", "gpt1_stego"), 
                   help="Directory for caching the GPT-1 checkpoint (default: ~/.cache/gpt1_stego)")
    return p
    
def encode_text_into_image(text: str, src_img_path: str, out_img_path: str, bits: int = 8):
    """
    Embeds user-provided text (instead of GPT-1) into a pointillist canvas.
    """
    raw = text.encode("utf-8")
    comp = zstd.ZstdCompressor(level=22, threads=os.cpu_count() or 1).compress(raw)
    size_bytes = len(comp)

    bytes_per_pixel_payload = (3 * bits) / 8.0
    min_total_pixels = math.ceil(size_bytes / bytes_per_pixel_payload)
    side = max(2, math.ceil(math.sqrt(min_total_pixels)))
    side += side % 2  # ensure even for symmetry

    canvas_cap_bytes = capacity(side, side, bits)
    if size_bytes > canvas_cap_bytes:
        raise ValueError("Text too long for selected encoding depth and canvas size")

    payload_to_embed = comp + b"\0" * (canvas_cap_bytes - size_bytes)
    dots = max(1, side * side // 16)

    painted_canvas_array = paint_canvas(Image.open(src_img_path), side, dots, jitter=6, dot_r=1)
    stego_img = embed(payload_to_embed, painted_canvas_array, bits)

    stego_img.save(out_img_path, format="PNG", compress_level=9)
if __name__ == "__main__":
    args = make_parser().parse_args()
    
    args.cache = Path(args.cache).expanduser().resolve()

    if args.cmd == "encode": encode(args)
    else: verify(args)