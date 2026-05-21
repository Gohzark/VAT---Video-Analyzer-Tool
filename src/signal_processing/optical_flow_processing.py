import os
import numpy as np
import argparse            
                    
def flow_to_magnitudes(video: str, algo: str) -> None:
    flow_path = os.path.join("outputs", video, algo, "optical_flow.npy")
    output_path = os.path.join(os.path.dirname(flow_path), "magnitudes.npy")

    if os.path.exists(output_path):
        answer = input(f"[flow_to_magnitudes] {output_path} existe déjà. Écraser ? (o/n) : ").strip().lower()
        if answer != 'o':
            print("Fichier conservé, conversion annulée.")
            return

    flows = np.load(flow_path)
    result = []
    for flow in flows:
        u = flow[..., 0]
        v = flow[..., 1]
        mag = np.sqrt(u**2 + v**2)
        score = float(np.sum(mag)) / (mag.shape[0] * mag.shape[1])
        result.append((score))

    np.save(output_path, np.array(result))
    print(f"Magnitudes sauvegardées dans : {output_path}")
    
def main(args):
    flow_to_magnitudes(args.input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert optical flow to magnitudes and angles')
    parser.add_argument(
        'input',
        type=str,
        help='Path to the input optical flow file'
    )
    args = parser.parse_args()
    main(args)