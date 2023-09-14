import argparse
import os, sys
import cv2
import numpy as np
from natsort import natsorted
import subprocess
from tensorboard.backend.event_processing import event_accumulator


def decode_from_buffer(encoded_image_string):
    s = np.frombuffer(encoded_image_string, dtype=np.uint8)
    image = cv2.imdecode(s, cv2.IMREAD_COLOR)
    return image


def main(args):

    if args.log_dir == '':
        print("Did not specify the log directory to compile video from. Please check again.")
        sys.exit()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not os.path.exists(os.path.join(args.output_dir, 'frames')):
        os.mkdir(os.path.join(args.output_dir, 'frames'))

    for filename in os.listdir(args.log_dir):
        if 'events.out.tfevents' not in filename: # There should just be one file in the entire folder anyway, but just in case.
            continue
        event_file = os.path.join(args.log_dir, filename)

        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        keys = ea.images._buckets.keys()

        entry_to_image_dict = {} # matches each entry of the form "epoch X iteration Y training_video img_category_name" to its associated images
        epoch_dict = {} # each key contains all entries for that epoch. Keys are integers.

        for entry in keys:
            if 'epoch' not in entry or 'iteration' not in entry or 'training_video' not in entry:
                continue
            entry_to_image_dict[entry] = ea.images._buckets[entry].items[0].encoded_image_string

            epoch = int(entry.split(" ")[1])
            if epoch not in epoch_dict:
                epoch_dict[epoch] = []
            epoch_dict[epoch].append(entry)

        epoch_list = epoch_dict.keys()
        epoch_list_sorted = natsorted(epoch_list) # e.g. [1,2,3]

        for epoch in epoch_list_sorted:
            entries_in_this_epoch = epoch_dict[epoch]
            entries_in_this_epoch = natsorted(entries_in_this_epoch)

            # key are all the image category names. e.g. real_A, fake_B.
            # values are the the entries with those names, in this epoch.
            img_category_names = {}

            for entry in entries_in_this_epoch:
                img_category_name = entry.split(' ')[5]
                if img_category_name not in img_category_names:
                    img_category_names[img_category_name] = []
                img_category_names[img_category_name].append(entry)

            epoch_img = [] # the final output image of this epoch
            for img_category_name, entries_having_that_name in img_category_names.items():
                imgs_for_one_category = []
                for entry in entries_having_that_name:
                    img = decode_from_buffer(entry_to_image_dict[entry])
                    imgs_for_one_category.append(img)

                imgs_for_one_category = np.concatenate(imgs_for_one_category, axis=1)  # concatenate images along width
                imgs_for_one_category = cv2.putText(imgs_for_one_category, text='epoch ' + str(epoch) + ' ' + img_category_name,
                                     org=(0, imgs_for_one_category.shape[0] - 20), color=(0, 255, 0), thickness=2,
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2)

                epoch_img.append(imgs_for_one_category)
            epoch_img = np.concatenate(epoch_img, axis=0) # concatenate images along height

            outpath = os.path.join(os.path.join(args.output_dir, 'frames'), '{:06}.jpg'.format(epoch))
            cv2.imwrite(outpath, epoch_img)


        command = 'ffmpeg -y -framerate ' + str(args.fps) + ' -i ' + \
        os.path.join(os.path.join(args.output_dir, 'frames'), '%06d.jpg') + ' ' + os.path.join(args.output_dir, 'training_timeline.mp4')
        sp = subprocess.Popen(command, shell=True)
        while sp.poll() is None:
            continue



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Style Master')
    parser.add_argument('--log_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--fps', type=int, default=2)
    args = parser.parse_args()
    main(args)
