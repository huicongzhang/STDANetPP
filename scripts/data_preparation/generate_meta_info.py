import os
from os import path as osp
from PIL import Image

from basicsr.utils import scandir

def generate_meta_info_div2k():
    """Generate meta info for DIV2K dataset.
    """
    dataset = ["GoPro"]
    # dataset = ["DVD"]
    # /home/hczhang/datasets/BSD_BasicsrReorder/BSD_ALL/test
    phase = ["train","test"]
    for dataname in dataset:
        for phasename in phase:
            gt_folder = '/home/hczhang/datasets/{}/{}/gt'.format(dataname,phasename)
            meta_info_txt = '/home/hczhang/CODE/TCSVT/basicsr/data/meta_info/{}_{}.txt'.format(dataname,phasename)
            # img_list = sorted(list(scandir(gt_folder,recursive=True)))
            img_list = sorted(list(os.listdir(gt_folder)))

            with open(meta_info_txt, 'w') as f:
                for idx, img_path in enumerate(img_list):
                    print(osp.join(gt_folder, img_path,"00000.png"))
                    seqs = list(scandir(osp.join(gt_folder, img_path)))
                    img = Image.open(osp.join(gt_folder, img_path,"00000.png"))  # lazy load
                    width, height = img.size
                    mode = img.mode
                    if mode == 'RGB':
                        n_channel = 3
                    elif mode == 'L':
                        n_channel = 1
                    else:
                        raise ValueError(f'Unsupported mode {mode}.')

                    info = f'{img_path} {len(seqs)} ({height},{width},{n_channel})'
                    print(idx + 1, info)
                    f.write(f'{info}\n')


if __name__ == '__main__':
    generate_meta_info_div2k()
