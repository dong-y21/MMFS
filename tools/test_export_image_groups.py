import os, sys, cv2, subprocess
import yaml, shutil
from tqdm import tqdm
from multiprocessing import Process, Pool
from tools.concat_images_or_videos import ConcatImages, default_concat_mapping, default_model_postfix_mapping

def revise_yaml_config(yaml_folder, save_img_folder, save_yaml_folder, gpu, test_folder):

    yaml_config = yaml.safe_load(open(yaml_folder, "r"))
    yaml_config["common"]["gpu_ids"] = list(map(int, gpu.split()))
    yaml_config["testing"]["results_dir"] = save_img_folder
    yaml_config["testing"]["image_format"] = "png"
    yaml_config["dataset"]["data_type"] = ['paired']
    t_yaml_folder = os.path.join(save_yaml_folder, yaml_folder.split('/')[-1])
    yaml.safe_dump(yaml_config, open(t_yaml_folder, "w"), default_flow_style=False)


def run_test_single_process(gpu_ids_list, yaml_list, test_folder_list, ckpt_list):

    assert len(gpu_ids_list) == len(yaml_list)
    assert len(gpu_ids_list) == len(test_folder_list)
    assert len(gpu_ids_list) == len(ckpt_list)

    total_single_process_number = len(gpu_ids_list)

    for process_idx in range(total_single_process_number):
        t_gpu_ids = gpu_ids_list[process_idx]
        t_yaml = yaml_list[process_idx]
        t_test_folder = test_folder_list[process_idx]
        t_ckpt = ckpt_list[process_idx]

        print("python3 -u test.py " + \
                  " --cfg_file " + t_yaml + \
                  " --test_folder " + t_test_folder + \
                  " --ckpt " + t_ckpt)
        p = subprocess.Popen("python3 -u test.py " + \
                              " --cfg_file " + t_yaml + \
                              " --test_folder " + t_test_folder + \
                              " --ckpt " + t_ckpt, shell=True)
        p.wait()
        print("finish " + t_yaml.split('/')[-1])


def concat_images(save_folder_list):
    # Concat images
    concat_images_manager = ConcatImages()
    total_save_folder_number = len(save_folder_list)

    print("export_list:\n", "\n".join(map(str, export_list)))
    t_export_name_list = [str(export_list[t_idx]["export_name"]) for t_idx in range(total_save_folder_number)]
    t_export_name_list = [concat_ori_image_name] + t_export_name_list

    concat_ori_image_path_list = list(os.listdir(concat_ori_image_path))

    t_model_type_list = []
    for t_idx in range(total_save_folder_number):
            t_model_type_list += [str(export_list[t_idx]["export_type"])]

    for t_idx, t_ori_image_name in enumerate(tqdm(concat_ori_image_path_list, total=len(concat_ori_image_path_list))):
        t_ori_image_prefix_name = t_ori_image_name.split(".")[0]
        t_image_path_list = []

        for save_folder_idx in range(total_save_folder_number):
            t_image_name = t_ori_image_prefix_name + default_model_postfix_mapping[t_model_type_list[save_folder_idx]]
            t_image_path = os.path.join(save_folder_list[save_folder_idx], t_image_name)
            assert os.path.isfile(t_image_path), "No file in {}".format("%s"%t_image_path)
            t_image_path_list += [t_image_path]

        t_image_path_list = [os.path.join(concat_ori_image_path, t_ori_image_name)] + t_image_path_list
        assert len(t_image_path_list) == len(t_export_name_list), "Error image({}) and text({}) number".format("%d"%len(t_image_path_list), "%d"%len(t_export_name_list))

        concat_images_manager.run_concat_images(concat_save_image_path, t_image_path_list, t_export_name_list, *default_concat_mapping[total_save_folder_number + 1])


def run():
    # init env
    os.makedirs(save_yaml_folder, exist_ok=True)
    shutil.rmtree(save_yaml_folder)
    os.makedirs(save_yaml_folder, exist_ok=True)

    t_gpu_list = [[] for t_idx in range(gpu_number)]
    t_yaml_folder_list = [[] for t_idx in range(gpu_number)]
    t_test_folder_list = [[] for t_idx in range(gpu_number)]
    t_ckpt_list = [[] for t_idx in range(gpu_number)]
    t_concat_folder_list = []

    # Load yaml config
    export_number = len(export_list)
    for t_idx in range(export_number):
        t_gpu = str(gpu_ids[t_idx % gpu_number])
        t_yaml_folder = str(export_list[t_idx]["export_yaml"])
        t_ckpt = str(export_list[t_idx]["export_ckpt"])

        t_test_type = str(export_list[t_idx]["export_test_type"])
        t_test_folder = str(globals()["test_" + t_test_type + "_data_path"])
        t_save_folder_root = str(export_list[t_idx]["export_save_folder"])

        t_yaml_config = yaml.safe_load(open(t_yaml_folder, "r"))
        t_export_exp_name = t_yaml_config["common"]["name"]

        t_save_folder = os.path.join(t_save_folder_root, t_export_exp_name, t_export_exp_name)

        t_flag_concat_images = export_list[t_idx]["export_flag_concat"]
        if t_flag_concat_images == True:
            t_concat_folder_list += [t_save_folder]

        t_generate = export_list[t_idx]["export_flag_generate"]

        assert t_generate == True or t_generate == False, "Error generate type..."
        print(t_generate, type(t_generate))
        if t_generate == True:
            os.makedirs(t_save_folder, exist_ok=True)
            shutil.rmtree(t_save_folder)
            os.makedirs(t_save_folder, exist_ok=True)
            revise_yaml_config(t_yaml_folder, t_save_folder_root, save_yaml_folder, t_gpu, t_test_folder)
            t_yaml_folder = os.path.join(save_yaml_folder, t_yaml_folder.split('/')[-1])

            t_gpu_list[t_idx % gpu_number] += [t_gpu]
            t_yaml_folder_list[t_idx % gpu_number] += [t_yaml_folder]
            t_test_folder_list[t_idx % gpu_number] += [t_test_folder]
            t_ckpt_list[t_idx % gpu_number] += [t_ckpt]
        else:
            assert os.path.isdir(t_save_folder) and len(os.listdir(t_save_folder)) != 0, "No files in {}".format("%s"%t_save_folder)

    print("begin export images...")
    # Multi-process inference
    p = Pool(export_number)
    for gpu_idx in range(gpu_number):
        p.apply_async(run_test_single_process, args=(t_gpu_list[gpu_idx],
                                                     t_yaml_folder_list[gpu_idx],
                                                     t_test_folder_list[gpu_idx],
                                                     t_ckpt_list[gpu_idx]))

    p.close()
    p.join()
    print("finish export images...")

    if flag_concat_images == True:
        print("begin concat {} images...".format("%d"%(len(t_concat_folder_list) + 1)))
        concat_images(t_concat_folder_list)
        print("finish concat images...")

    print("finish all process steps...")


if __name__ == '__main__':

    yaml_path = sys.argv[1]

    config = yaml.safe_load(open(yaml_path))
    for k, v in config.items():
        globals()[k] = v

    run()
