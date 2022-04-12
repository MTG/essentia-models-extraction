import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from feature_embeddings import EmbeddingModel
from feature_melspectrogram import feature_melspectrogram


def extract(
    in_dir,
    out_dir,
    filelist,
    feature_type,
    model_path=None,
    force=False,
    dry_run=False,
    from_melspectrogram=True,
    batch_size=128,
    hop_time=1,
    output="embeddings",
):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    if "melspectrogram" in feature_type:
        extractor = feature_melspectrogram(feature_type, hop_time)
        if from_melspectrogram:
            raise Exception(
                f"`--from-melspectrogram` not available for feature {feature_type}"
            )
    else:
        assert Path(model_path).exists(), f"{model_path} does not exist!"

        extractor = EmbeddingModel(
            feature_type,
            model_path,
            hop_time=hop_time,
            batch_size=batch_size,
            output=output,
        )

    with open(filelist, "r") as ff:
        sources = ff.readlines()

    files_to_process = []
    for src_path in sources:
        src = in_dir / src_path.rstrip()
        tgt = out_dir / src_path.rstrip()
        if tgt.suffix != ".mmap":
            tgt = tgt.with_suffix(tgt.suffix + ".mmap")

        if not src.exists():
            raise FileNotFoundError(
                f"`{str(src)}` does not exist. All source files must exist."
            )

        if not tgt.exists() or force:
            files_to_process.append((src, tgt))

    if not files_to_process:
        print("no files to process!")
    else:
        print(f"processing ({len(files_to_process)}) files")
        print(f"first src/tgt pair {files_to_process[0]}")

    if not dry_run:
        pbar = tqdm(files_to_process, desc="Extractor")
        for src, tgt in files_to_process:
            tgt.parent.mkdir(parents=True, exist_ok=True)

            if from_melspectrogram:
                output = extractor.compute_from_melspectrogram(src)
            else:
                output = extractor.compute(src)

            fp = np.memmap(tgt, dtype="float16", mode="w+", shape=output.shape)
            fp[:] = output[:]
            del fp

            pbar.update()
        pbar.close()
    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "batch extractor for melspectrograms, embeddings, and activations for the Essentia models"
    )
    parser.add_argument("input_folder", help="base folder with the audio")
    parser.add_argument("output_folder", help="base folder for the output features")
    parser.add_argument("filelist", help="path to files from `input_folder`")
    parser.add_argument(
        "--feature",
        choices=[
            # melspectrograms
            "musicnn_melspectrogram",
            "vggish_melspectrogram",
            "openl3_melspectrogram",
            # embeddings
            "effnet_b0",
            "effnet_b0_3M",
            "musicnn",
            "openl3",
            "vggish",
            "yamnet",
        ],
        help="the feature to extract. Can be melspectrogram or embeddings",
    )
    parser.add_argument(
        "--model-path",
        help="path to the model file (only required for embeddings and activations)",
    )
    parser.add_argument(
        "--from-melspectrogram",
        action="store_true",
        help="extract the embeddings or activations from pre-computed melspectrograms (speed-up the process)",
    )
    parser.add_argument(
        "--force", action="store_true", help="recompute if the targets already exists"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list files to process without computing"
    )
    parser.add_argument(
        "--output",
        choices=["embeddings", "activations"],
        default="embeddings",
        help="whether to extract embeddings or activatons",
    )
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        help="batch size for the analysis (only noticeable when a GPU is available)",
    )
    parser.add_argument(
        "--hop-time",
        default=1.0,
        type=float,
        help="hop time for the embeddings or activations in seconds.",
    )

    args = parser.parse_args()

    extract(
        args.input_folder,
        args.output_folder,
        args.filelist,
        args.feature,
        model_path=args.model_path,
        force=args.force,
        dry_run=args.dry_run,
        from_melspectrogram=args.from_melspectrogram,
        batch_size=args.batch_size,
        hop_time=args.hop_time,
        output=args.output,
    )
