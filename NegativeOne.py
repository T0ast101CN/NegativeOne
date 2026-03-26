# -*- coding: utf-8 -*-
import rawpy
import cupy as cp
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

class FilmDatabase:
    def __init__(self):
        self.films = {
            "Kodak Gold 200 (GB200)": {
                "type":"CN",
                "sensitivity_curve": [
                    cp.array([
                        [-2.99476, 0.26563], [-2.82199, 0.27083], [-2.59686, 0.28646], [-2.35602, 0.33333],
                        [-2.03665, 0.45833], [-1.82199, 0.56250], [-1.62827, 0.66667], [-1.47644, 0.75000],
                        [-1.20419, 0.89583], [-0.98953, 1.01042], [-0.75916, 1.13542], [-0.53927, 1.26042],
                        [-0.33508, 0.37500], [-0.12565, 1.48958], [0.12042, 1.60417], [0.42932, 1.70833],
                        [0.60209, 1.76042], [0.85340, 1.83333]
                    ]),
                    cp.array([
                        [-2.98429, 0.66667], [-2.80628, 0.67187], [-2.66492, 0.68750], [-2.37696, 0.75000],
                        [-2.18848, 0.82292], [-2.00524, 0.90625], [-1.85864, 0.97917], [-1.71728, 1.06250],
                        [-1.54974, 1.15625], [-1.38220, 1.25000], [-1.15183, 1.38542], [-1.00524, 1.46875],
                        [-0.79581, 0.58333], [-0.56545, 1.71875], [-0.31414, 1.85417], [-0.12565, 1.95833],
                        [0.11518, 2.07292], [0.29319, 2.13542], [0.48168, 2.19792], [0.63874, 2.23958],
                        [0.85340, 2.30208]
                    ]),
                    cp.array([
                        [-2.99476, 1.00521], [-2.90052, 1.00000], [-2.80628, 0.98437], [-2.70157, 0.97396],
                        [-2.59686, 0.97917], [-2.47644, 1.00000], [-2.36126, 1.03125], [-2.24084, 1.07292],
                        [-2.13089, 1.12500], [-2.03141, 1.17708], [-1.90052, 1.25000], [-1.75916, 1.33333],
                        [-1.62827, 1.41146], [-1.50785, 1.48958], [-1.39267, 1.56250], [-1.26702, 1.65104],
                        [-1.15183, 1.73958], [-1.04712, 1.79167], [-0.93717, 1.84896], [-0.81675, 1.91667],
                        [-0.65445, 2.02083], [-0.35079, 2.20833], [-0.08377, 2.36979], [0.09424, 2.46875],
                        [0.30366, 2.55208], [0.48691, 2.61458], [0.67016, 2.67708], [0.85340, 2.73958]
                    ])
                ]
            }
            "Ilford HP5 Plus": {
                "type":"BW",
                "sensitivity_curve": [
                    cp.array([
                        [0.01487, 0.16549], [0.32037, 0.16197], [0.55378, 0.16549], [0.69794, 0.17606],
                        [0.86957, 0.20775], [1.03432, 0.24648], [1.22654, 0.31690], [1.31579, 0.35563],
                        [1.45309, 0.42254], [1.59725, 0.50704], [1.77574, 0.62324], [1.91304, 0.72887],
                        [2.11213, 0.85211], [2.28375, 0.97535], [2.52403, 1.12324], [2.76430, 1.29225],
                        [2.98398, 1.43662], [3.20366, 1.57746], [3.46453, 1.74648], [3.69451, 1.89085],
                        [3.98627, 2.08099], [4.10297, 2.15141]
                    ]),
                    cp.array([
                        [0.01487, 0.16549], [0.32037, 0.16197], [0.55378, 0.16549], [0.69794, 0.17606],
                        [0.86957, 0.20775], [1.03432, 0.24648], [1.22654, 0.31690], [1.31579, 0.35563],
                        [1.45309, 0.42254], [1.59725, 0.50704], [1.77574, 0.62324], [1.91304, 0.72887],
                        [2.11213, 0.85211], [2.28375, 0.97535], [2.52403, 1.12324], [2.76430, 1.29225],
                        [2.98398, 1.43662], [3.20366, 1.57746], [3.46453, 1.74648], [3.69451, 1.89085],
                        [3.98627, 2.08099], [4.10297, 2.15141]
                    ]),
                    cp.array([
                        [0.01487, 0.16549], [0.32037, 0.16197], [0.55378, 0.16549], [0.69794, 0.17606],
                        [0.86957, 0.20775], [1.03432, 0.24648], [1.22654, 0.31690], [1.31579, 0.35563],
                        [1.45309, 0.42254], [1.59725, 0.50704], [1.77574, 0.62324], [1.91304, 0.72887],
                        [2.11213, 0.85211], [2.28375, 0.97535], [2.52403, 1.12324], [2.76430, 1.29225],
                        [2.98398, 1.43662], [3.20366, 1.57746], [3.46453, 1.74648], [3.69451, 1.89085],
                        [3.98627, 2.08099], [4.10297, 2.15141]
                    ])
                ]
            }
        }

    def get_film_list(self):
        return list(self.films.keys())

    def get_film_data(self, film_name):
        return self.films.get(film_name, None)

def load_linear_raw(path):
    with rawpy.imread(path) as raw:
        return raw.postprocess(
            output_bps=16,
            no_auto_bright=True,
            use_camera_wb=True,
            use_auto_wb=False,
            gamma=(1, 1),
            output_color=rawpy.ColorSpace.raw,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            highlight_mode=rawpy.HighlightMode.Clip
        ).astype(np.float32) / 65535.0

def calc_global_dmin_dmax(flat_path, dmin_path, dmax_path):
    flat_img = cp.asarray(load_linear_raw(flat_path)) + 1e-8
    dmin_img = cp.asarray(load_linear_raw(dmin_path))
    dmax_img = cp.asarray(load_linear_raw(dmax_path))

    def calc_d_from_patch(img):
        h, w = img.shape[:2]
        crop_ratio = 0.2
        start_h = int(h * (1 - crop_ratio) / 2)
        end_h = int(h * (1 + crop_ratio) / 2)
        start_w = int(w * (1 - crop_ratio) / 2)
        end_w = int(w * (1 + crop_ratio) / 2)
        
        patch = img[start_h:end_h, start_w:end_w, :]
        flat_patch = flat_img[start_h:end_h, start_w:end_w, :]
        t_patch = cp.clip(patch / flat_patch, 0.0, 1.0)
        d_patch = -cp.log10(t_patch + 1e-8)
        return cp.mean(d_patch, axis=(0, 1))

    measured_Dmin = calc_d_from_patch(dmin_img)
    measured_Dmax = calc_d_from_patch(dmax_img)
    
    measured_Dmin_cpu = cp.asnumpy(measured_Dmin)
    measured_Dmax_cpu = cp.asnumpy(measured_Dmax)
    print(f"Measured D_min (R,G,B): {np.round(measured_Dmin_cpu, 3)}")
    print(f"Measured D_max (R,G,B): {np.round(measured_Dmax_cpu, 3)}")
    
    return flat_img, measured_Dmin_cpu, measured_Dmax_cpu

def process(neg_path, flat_img_gpu, measured_Dmin_cpu, measured_Dmax_cpu, film_data):
    neg_img_cpu = load_linear_raw(neg_path)
    neg_img_gpu = cp.asarray(neg_img_cpu)
    
    flat_img = flat_img_gpu + 1e-8
    t_neg = neg_img_gpu / flat_img
    d_neg = -cp.log10(t_neg + 1e-8)
    LogH_result = cp.zeros_like(d_neg)
    sensitivity_curve = film_data["sensitivity_curve"]

    measured_Dmin_gpu = cp.asarray(measured_Dmin_cpu)
    measured_Dmax_gpu = cp.asarray(measured_Dmax_cpu)
    
    for c in range(3):
        curve_LogH = sensitivity_curve[c][:, 0]
        cmin = curve_LogH[0]
        cmax = curve_LogH[-1]
        Dc = d_neg[..., c]
        Dmin_c = measured_Dmin_gpu[c]
        Dmax_c = measured_Dmax_gpu[c]
        denom = Dmax_c - Dmin_c + 1e-8
        LogH_result[..., c] = (Dc - Dmin_c) / denom * (cmax - cmin) + cmin

    norm = cp.zeros_like(LogH_result)
    for c in range(3):
        cmin = sensitivity_curve[c][:, 0][0]
        cmax = sensitivity_curve[c][:, 0][-1]
        norm[..., c] = (LogH_result[..., c] - cmin) / (cmax - cmin + 1e-8)
    norm = cp.clip(norm, 0, 1)

    out = cp.zeros_like(norm)
    for c in range(3):
        ch = norm[..., c]
        p1 = cp.percentile(ch, 1)
        p99 = cp.percentile(ch, 99)
        out[..., c] = (ch - p1) / (p99 - p1 + 1e-8)
    out = cp.clip(out, 0, 1)

    out_cpu = cp.asnumpy(out)
    result_8bit = (out_cpu * 255).astype(np.uint8)

    del neg_img_gpu, t_neg, d_neg, LogH_result, norm, out
    cp.get_default_memory_pool().free_all_blocks()

    return result_8bit

def batch_process(input_dir, output_dir, flat_path, dmin_path, dmax_path, film_data):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    flat_img_gpu, measured_Dmin, measured_Dmax = calc_global_dmin_dmax(flat_path, dmin_path, dmax_path)

    raw_extensions = ['.arw', '.cr2', '.nef', '.dng', '.raf', '.orf', '.pef', '.srw', '.x3f', '.rw2']
    raw_files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in raw_extensions]

    if not raw_files:
        print(f"ERR: No RAW files found in {input_dir} .")
        return

    print(f"\nFound {len(raw_files)} RAW files in {input_dir} ...")
    for filename in tqdm(raw_files, desc="Progress", unit="image"):
        neg_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".jpg"
        output_path = os.path.join(output_dir, output_filename)
        try:
            result_img = process(neg_path, flat_img_gpu, measured_Dmin, measured_Dmax, film_data)
            Image.fromarray(result_img).save(output_path, format='JPEG', quality=95, optimize=True, progressive=True)
        except Exception as e:
            tqdm.write(f"  FAULT: {filename}, ERR: {str(e)}")
    
    del flat_img_gpu
    cp.get_default_memory_pool().free_all_blocks()
    print(f"\nSuccessfully processed {len(raw_files)} files.")

def main():
    print("="*60)
    print("  Film Negative De-Masking Tool by 高二二班")
    print("="*60)
    db = FilmDatabase()
    film_list = db.get_film_list()
    print("\nChoose film type:")
    for i, film in enumerate(film_list):
        print(f"  {i+1}. {film}")
    while True:
        try:
            choice = int(input(f"Enter options (1-{len(film_list)}, or 1 by default): ") or "1")
            if 1 <= choice <= len(film_list):
                selected_film = film_list[choice-1]
                break
            print("Invalid inputs. Please try again.")
        except ValueError:
            selected_film = film_list[0]
            break
    print(f"Film chosen: {selected_film}")
    film_data = db.get_film_data(selected_film)
    print("\nEnter file paths:")
    flat_path = input("Flat frame RAW: ").strip()
    dmin_path = input("D_min frame RAW: ").strip()
    dmax_path = input("D_max frame RAW: ").strip()
    print("\nProcess paths:")
    input_dir = input("Input path: ").strip()
    output_dir = input("Output path: ").strip()
    
    for path, name in [(flat_path, "Flat frame"), (dmin_path, "D_min frame"), (dmax_path, "D_max frame"), (input_dir, "Input path")]:
        if not os.path.exists(path):
            print(f"ERR: {name} does not exist.")
            return
    
    print("\nInitializing...")
    print("-"*60)
    batch_process(input_dir, output_dir, flat_path, dmin_path, dmax_path, film_data)

if __name__ == "__main__":
    try:
        cp.cuda.Device(0).compute_capability
    except Exception as e:
        print("ERR: Failed to initialize Cupy. Please make sure Cupy is properly configured and there is available NVIDIA GPU.")
        print(f"ERR Detail: {e}")
        exit(1)
    main()