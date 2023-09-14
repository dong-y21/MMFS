import random
import torch

def load_patches(patch_batch_size, batch_size, patch_size, num_patch, diff_patch, index, data, transforms, return_dict):
    if patch_size > 0:
        assert (patch_batch_size % batch_size == 0), \
            "patch_batch_size is not divisible by batch_size."
        if 'paired_A' in return_dict or 'paired_B' in return_dict:
            if not diff_patch:
                # load patch from current image
                patchA = return_dict['paired_A'].clone()
                patchB = return_dict['paired_B'].clone()
            else:
                # load patch from a different image
                pathA = data['paired_A_path'][(index + 1) % len(data['paired_A_path'])]
                pathB = data['paired_B_path'][(index + 1) % len(data['paired_B_path'])]
                patchA, patchB = transforms['paired'](pathA, pathB)
        else:
            if not diff_patch:
                # load patch from current image
                patchA = return_dict['unpaired_A'].clone()
                patchB = return_dict['unpaired_B'].clone()
            else:
                # load patch from a different image
                pathA = data['unpaired_A_path'][(index + 1) % len(data['unpaired_A_path'])]
                pathB = data['unpaired_B_path'][(index + 1) % len(data['unpaired_B_path'])]
                patchA, patchB = transforms['unpaired'](pathA, pathB)

        # crop patch
        patchAs = []
        patchBs = []
        _, h, w = patchA.size()

        for _ in range(num_patch):
            r = random.randint(0, h - patch_size - 1)
            c = random.randint(0, w - patch_size - 1)
            patchAs.append(patchA[:, r:r + patch_size, c:c + patch_size])
            patchBs.append(patchB[:, r:r + patch_size, c:c + patch_size])

        patchAs = torch.cat(patchAs, 0)
        patchBs = torch.cat(patchBs, 0)

        return_dict['patch_A'] = patchAs
        return_dict['patch_B'] = patchBs
