import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.inference_d_sandbox import generate_model_d_sandbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    args = parser.parse_args()

    image_path = os.path.abspath(args.image_path)
    result = generate_model_d_sandbox(image_path, base_dir=ROOT)

    print('Model D Sandbox Result')
    print('=' * 60)
    print(f'Image        : {image_path}')
    print(f'Checkpoint   : {result["model_b_ckpt"]}')
    print(f'Label        : {result["clinical_label"]}')
    print(f'Report       : {result["text"]}')
    print('Retrieved')
    for case in result['retrieved_cases']:
        print(f'  {case["rank"]}. {case["label"]} | {case["score"]:.4f}')
        print(f'     {case["report"][:220]}')


if __name__ == '__main__':
    main()
