import argparse
import os
import pandas as pd
import numpy as np
import torch

from model.embedding.autoEncoder import autoAE, conv_layers_config


def load_and_preprocess(csv_path, nrows=None):
    df = pd.read_csv(csv_path, nrows=nrows)
    # existing code in autoEncoder.py uses df = df.iloc[:,2:]
    if df.shape[1] > 2:
        df_proc = df.iloc[:, 2:]
        df_idx = df.iloc[:,:2]
    else:
        df_proc = df
    X = df_proc.values.astype('float32')
    # assume 64x64 images stored row-wise
    X = X.reshape(-1, 1, 64, 64) / 255.0
    return X, df_idx


def encode_to_csv(input_csv, model_path, output_csv, batch_size=64, nrows=None, verify=False):
    print(f"Loading data from: {input_csv} (nrows={nrows})")
    X, df_idx = load_and_preprocess(input_csv, nrows=nrows)
    N = X.shape[0]
    print(f"Found {N} images -> tensor shape {X.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {model_path}")
    ae = autoAE(conv_config=conv_layers_config, input_dim=64, load_path=model_path)
    ae.to(device)
    ae.eval()

    x_tensor = torch.tensor(X, dtype=torch.float32)

    latents_list = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = x_tensor[i:i+batch_size].to(device)
            z = ae.encode(xb)  # z shape: [B, C, H, W]
            z_flat = z.cpu().numpy().reshape(z.size(0), -1)
            latents_list.append(z_flat)

            # optional verification on the first batch
            if verify:
                # reshape flattened back to original z shape and decode
                z_shape = (z.size(0), z.size(1), z.size(2), z.size(3))
                z_from_flat = torch.tensor(z_flat, dtype=torch.float32).view(*z_shape).to(device)

                recon_from_decode = ae.decode(z_from_flat)
                recon_from_forward = ae(xb)

                # bring to cpu and compare
                d1 = recon_from_decode.cpu()
                d2 = recon_from_forward.cpu()
                max_abs_diff = float((d1 - d2).abs().max())
                mean_abs_diff = float((d1 - d2).abs().mean())
                print(f"Verification (first batch): max_abs_diff={max_abs_diff:.6e}, mean_abs_diff={mean_abs_diff:.6e}")
                if max_abs_diff > 1e-6:
                    print("Warning: significant difference detected between decode(flat->reshape) and forward output. Check flatten/reshape ordering or numeric precision.")
                # only verify on first batch
                verify = False

    latents = np.vstack(latents_list)
    print(f"Encoded latents shape: {latents.shape}")

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df_latents = pd.DataFrame(latents)
    df_latents = pd.concat([df_idx.reset_index(drop=True), df_latents], axis=1)
    df_latents.to_csv(output_csv, index=False)
    print(f"Latents saved to: {output_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", default="./clean_sprites.csv", help="Path to clean_sprites.csv")
    p.add_argument("--model-path", default="./model/embedding/trained/autoAE_20251026_194448_mm34.pt",
                   help="Path to trained .pt file")
    p.add_argument("--output-csv", default="./clean_sprites_latents.csv", help="Output CSV path")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--nrows", type=int, default=None, help="Limit number of rows (for quick test)")
    p.add_argument("--verify", action="store_true", help="Verify flatten/reshape correctness on first batch")

    args = p.parse_args()
    encode_to_csv(args.input_csv, args.model_path, args.output_csv, batch_size=args.batch_size, nrows=args.nrows, verify=args.verify)


if __name__ == '__main__':
    main()
