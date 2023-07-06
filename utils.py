from lib import *


LOGGER = logging.getLogger(__name__)
def plot_segmentation_imags(
        savefolder,
        image_paths,
        segmentations,
        anomaly_scores = None,
        mask_paths = None,
        image_transform = lambda x: x,
        mask_transform = lambda x: x,
        save_depth = 4
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]
    
    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).covert('RGB')
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            iamge = image.numpy()
        
        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert('RGB')
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)
        
        savename = image_path.split('/')
        savename = '_'.join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)
        f, axes = plt.subplots(1, 2 + int(masks_provided))
        axes[0].imshow(image.transpose(1, 2, 0))
        axes[1].imshow(mask.transpose(1, 2, 0))
        axes[2].imshow(segmentation)
        f.set_size_inches(3 * (2 + int(masks_provided)), 3)
        f.tight_layout()
        f.savefig(savename)
        plt.close()

def create_storage_folder(
        main_folder_path, project_folder, group_folder, run_name, mode = 'iterate'
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_folder, group_folder, run_name)
    if mode == 'iterate':
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path,
                                     group_folder + '_' + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == 'overwrite':
        os.makedirs(save_path, exist_ok=True)
    
    return save_path

def compute_and_store_final_results(
        results_path,
        results,
        row_names = None,
        column_names = [
            "Instance AUROC",
            "Full Pixel AUROC",
            "Full PRO",
            "Anomaly Pixel AUROC",
            "Anomaly PRO"
        ]
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results)
    
    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))
    savename = os.path.join(results_path, 'results.csv')
    with open(savename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        header = column_names
        if row_names is not None:
            header = ['Row Names'] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ['Mean'] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {'mean_{0}'.format(key): item for key, item in mean_metrics.items()}
    return mean_metrics