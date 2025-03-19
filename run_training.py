#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#

import gc
import os
import sys
import numpy as np
import torch.backends
import cls_feature_class
import cls_data_generator
import parameters
import time
from time import gmtime, strftime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from SELD_evaluation_metrics import distance_between_cartesian_coordinates
import seldnet_model as seldnet_model
from training_augmentations import CompositeCutout, RandomShiftUpDownNp
from training_utils import *
from datetime import datetime
from manual_dataset import *
from rich.progress import Progress
from all_models import ResNet, ResNetConf18
from torchvision import transforms
import platform
from torch.optim.lr_scheduler import LambdaLR
from cst_former.CST_former_model import CST_former

# Clear torch cache
torch.cuda.empty_cache()

# Initialize the garbage collection and collect the garbage
gc.enable()
gc.collect()


def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = np.sqrt(x**2 + y**2 + z**2) > 0.5

    return sed, accdoa_in


def get_multi_accdoa_labels(accdoa_in, nb_classes, params=None):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*4*13]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=13]
        doaX:       [batch_size, frames, num_axis*num_class=3*13]
        distX:      [batch_size, frames, num_class=13]
    """

    # Determine if we are normalizing distances
    norm_d = params['normalize_distance']

    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    dist0 = accdoa_in[:, :, 3*nb_classes:4*nb_classes]
    if norm_d == False:
        dist0[dist0 < 0.] = 0.
    elif norm_d == True:
        dist0 = (dist0 * params['dmax'] * params['dstd']) + params['dmean']
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > 0.5
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes], accdoa_in[:, :, 6*nb_classes:7*nb_classes]
    dist1 = accdoa_in[:, :, 7*nb_classes:8*nb_classes]
    if norm_d == False:
        dist1[dist1<0.] = 0.
    elif norm_d == True:
        dist1 = (dist1 * params['dmax'] * params['dstd']) + params['dmean']
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > 0.5
    doa1 = accdoa_in[:, :, 4*nb_classes: 7*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 8*nb_classes:9*nb_classes], accdoa_in[:, :, 9*nb_classes:10*nb_classes], accdoa_in[:, :, 10*nb_classes:11*nb_classes]
    dist2 = accdoa_in[:, :, 11*nb_classes:]
    if norm_d == False:
        dist2[dist2<0.] = 0.
    elif norm_d == True:
        dist2 = (dist2 * params['dmax'] * params['dstd']) + params['dmean']
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > 0.5
    doa2 = accdoa_in[:, :, 8*nb_classes:11*nb_classes]

    return sed0, doa0, dist0, sed1, doa1, dist1, sed2, doa2, dist2


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def test_epoch(data_generator, model, criterion, dcase_output_folder, params, device):
    test_filelist = data_generator.get_filelist()

    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0

    with Progress(transient=True) as progress:
        task = progress.add_task("[green]Validation Loop : ", total=len(test_filelist))
        with torch.no_grad():
            for values in data_generator.generate():

                # Get the data and target
                data, target = values
                data = torch.tensor(data, device=device).float().contiguous()
                target = torch.tensor(target, device=device).float().contiguous()

                # Split the data into managable chunks, in this case 32 items
                data_chunks = torch.split(data, 32)
                output_chunks = []
                for data_chunk in data_chunks:
                    output_chunk = model(data_chunk)
                    output_chunks.append(output_chunk)
                output = torch.cat(output_chunks, dim=0)

                # Remove the variables
                del output_chunk, output_chunks, data_chunk, data_chunks

                loss = criterion(output, target)

                # Extract the SED, DOA and DISTANCE predictions
                sed_pred0, doa_pred0, dist_pred0, sed_pred1, doa_pred1, dist_pred1, sed_pred2, doa_pred2, dist_pred2 = get_multi_accdoa_labels(output.detach().cpu().numpy(), 
                                                                                                                                                params['unique_classes'],
                                                                                                                                                params)
                sed_pred0 = reshape_3Dto2D(sed_pred0)
                doa_pred0 = reshape_3Dto2D(doa_pred0)
                dist_pred0 = reshape_3Dto2D(dist_pred0)
                sed_pred1 = reshape_3Dto2D(sed_pred1)
                doa_pred1 = reshape_3Dto2D(doa_pred1)
                dist_pred1 = reshape_3Dto2D(dist_pred1)
                sed_pred2 = reshape_3Dto2D(sed_pred2)
                doa_pred2 = reshape_3Dto2D(doa_pred2)
                dist_pred2 = reshape_3Dto2D(dist_pred2)

                # dump SELD results to the corresponding file
                output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))

                file_cnt += 1

                if platform.system() == 'Linux':
                    output_dict = optimize_output_dict_creation(params, sed_pred0, sed_pred1, sed_pred2,
                                                                doa_pred0, doa_pred1, doa_pred2,
                                                                dist_pred0, dist_pred1, dist_pred2)
                else:
                    output_dict = {}

                    for frame_cnt in range(sed_pred0.shape[0]):
                        for class_cnt in range(sed_pred0.shape[1]):
                            # determine whether track0 is similar to track1
                            flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                            flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                            flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                            # unify or not unify according to flag
                            if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                                if sed_pred0[frame_cnt][class_cnt]>0.5:
                                    if frame_cnt not in output_dict:
                                        output_dict[frame_cnt] = []
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred0[frame_cnt][class_cnt]])
                                if sed_pred1[frame_cnt][class_cnt]>0.5:
                                    if frame_cnt not in output_dict:
                                        output_dict[frame_cnt] = []
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred1[frame_cnt][class_cnt]])
                                if sed_pred2[frame_cnt][class_cnt]>0.5:
                                    if frame_cnt not in output_dict:
                                        output_dict[frame_cnt] = []
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred2[frame_cnt][class_cnt]])
                            elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                if flag_0sim1:
                                    if sed_pred2[frame_cnt][class_cnt]>0.5:
                                        output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred2[frame_cnt][class_cnt]])
                                    doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                    dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt]) / 2
                                    output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
                                elif flag_1sim2:
                                    if sed_pred0[frame_cnt][class_cnt]>0.5:
                                        output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred0[frame_cnt][class_cnt]])
                                    doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                    dist_pred_fc = (dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 2
                                    output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
                                elif flag_2sim0:
                                    if sed_pred1[frame_cnt][class_cnt]>0.5:
                                        output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred1[frame_cnt][class_cnt]])
                                    doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                    dist_pred_fc = (dist_pred2[frame_cnt] + dist_pred0[frame_cnt]) / 2
                                    output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
                            elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                                dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 3
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])

                data_generator.write_output_format_file(output_file, output_dict)

                test_loss += loss.item()
                nb_test_batches += 1
                progress.update(task, advance=1)

        test_loss /= nb_test_batches

    del data, target, output
    torch.cuda.empty_cache()
    return test_loss


def train_epoch(data_generator, optimizer, model, criterion, device, scheduler=None, nb_batches=1000, step_scheduler=None):
    nb_train_batches, train_loss = 0, 0.
    model.train()


    with Progress(transient=True) as progress:
        task = progress.add_task("[red]Training : ", total=nb_batches)

        for data, target in data_generator:

            data , target = data.to(device), target.to(device)

            # Training step
            output = model(data)
            loss = criterion(output, target)

            # Clear gradients
            optimizer.zero_grad(set_to_none=True)

            # Backprop
            loss.backward()

            # Update parameters
            optimizer.step()

            train_loss += loss.item()
            nb_train_batches += 1
            progress.update(task, advance=1)

            # For step based schedulers
            if step_scheduler is not None : step_scheduler.step()

    train_loss /= nb_train_batches
    # For epoch based schedulers
    if scheduler is not None : scheduler.step()

    del data, target, output
    torch.cuda.empty_cache()
    return train_loss


def main(argv):
    """
    Main wrapper for training sound event localization and detection network.

    :param argv: expects two optional inputs.
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1

    """
    print(argv)
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two optional inputs')
        print('\t>> python seld.py <task-id> <job-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')

    # Make the log file directory
    os.makedirs("./logs", exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.autograd.set_detect_anomaly(True)

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    job_id = '1' if len(argv) < 3 else argv[-1]

    # Create the logging file
    log_file = os.path.join("./logs", "{}_{}.txt".format(datetime.now().strftime("%d%m%y_%H%M"), str(task_id)))
    logger = open(log_file, "w")

    # Starting up the wandb logger
    project_title = "{}_{}_{}".format(task_id, job_id, datetime.now().strftime("%d%m%y_%H%M%S"))
    write_and_print(logger, project_title)


    # Training setup 
    """ Data splits definition """
    test_splits = [[4]]
    val_splits = [[4]]
    training_folds = [3]
    if params['use_fold1']:
        training_folds.append(1)
    if params['use_fold2']:
        training_folds.append(2)
    if params['use_fold5']:
        training_folds.append(5)
    if params['use_fold6']:
        training_folds.append(6)
    if params['use_fold7']:
        training_folds.append(7)
    if params['use_fold8']:
        training_folds.append(8)
    if params['use_fold9']:
        training_folds.append(9)
    if platform.system() == 'Linux':
        if len(training_folds) != 1:
            n_workers = 4
        else:
            n_workers = 2
        print("Running on Linux server, increasing num_workers to {}".format(n_workers))
    else:
        print("Running on Windows server, reducing num_workers to 0")
        n_workers = 0


    for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
        print('---------------------------------------------------------------------------------------------------')

        # Unique name for the run
        loc_feat = params['dataset']
        loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'

        cls_feature_class.create_folder(params['model_dir'])
        unique_name = '{}_{}_{}_split{}_{}_{}_{}'.format(
            task_id, job_id, params['mode'], split_cnt, loc_output, loc_feat, job_id
        )
        model_name = '{}_model.h5'.format(os.path.join(params['model_dir'], unique_name))
        write_and_print(logger, out_string="Unique Name: {}".format(unique_name))
        write_and_print(logger, out_string="Training started : {}".format(datetime.now().strftime("%d%m%y_%H%M%S")))

        # Load train and validation data
        print('Loading validation dataset:')
        data_gen_val = cls_data_generator.DataGenerator(
            params=params, split=val_splits[split_cnt], shuffle=False, per_file=True
        )
        data_in, data_out = data_gen_val.get_data_sizes()

        if "64" in job_id:
            params['batch_size'] = 64
        else:
            params['batch_size'] = 32


        if "trans" in job_id.lower():
            print("Training transforms are used!")
            training_transforms = transforms.Compose([
                RandomShiftUpDownNp(freq_shift_range=10),
                CompositeCutout(image_aspect_ratio=(data_in[-2]/data_in[-1]), n_zero_channels=int(data_in[1] - 4))
            ])
        else:
            training_transforms = None

        if "iter" in job_id or platform.system() == 'Linux':
            print("Creating iterated dataset!")
            # Initialize Database and get data
            db = Iterated_Database(feat_label_dir=params['feat_label_dir'],
                                   audio_format=params['dataset'],
                                   n_channels=data_in[1],
                                   n_bins=data_in[3],
                                   training_folds=training_folds)
            db_data = db.get_split()
            # Create dataset and dataloader
            dataset = Iterated_Dataset(db_data,
                                       transform=training_transforms)
            print("Iterated dataset loaded!")

        else:
            print("Creating manual training dataset!")
            # Initialize Database and get data
            train_db = Database(feat_label_dir=params['feat_label_dir'],
                                audio_format=params['dataset'],
                                n_channels=data_in[1],
                                n_bins=data_in[3],
                                training_folds=training_folds)
            db_data = train_db.get_split()
            # Create dataset and dataloader
            dataset = seldDataset(db_data=db_data,
                                  transform=training_transforms)
            print("Manual training dataset loaded!")

        # Creating the training dataloader
        training_dataloader = DataLoader(dataset=dataset,
                                         batch_size=params['batch_size'], shuffle=True,
                                         num_workers=n_workers, drop_last=False,
                                         pin_memory=True, prefetch_factor=2)
        n_batches = len(training_dataloader)
        print("Manual training dataloader created with {} batches using batch size of {}!".format(n_batches, params['batch_size']))

        # Deciding on model architecture
        if "baseline" in job_id:
            model = seldnet_model.SeldModel(data_in, data_out, params).to(device)
            write_and_print(logger, "Using baseline method!")
        elif "cst" in job_id:
            model = CST_former(in_feat_shape=data_in,
                               out_shape=data_out,
                               params=params).to(device)
            params['nb_epochs'] = 150
            params['lr'] = 1e-3
            write_and_print(logger, "Using CST-Former model architecture!")
        elif "conf" in job_id:
            model = ResNetConf18(in_feat_shape=data_in,
                                out_feat_shape=data_out,
                                normd=params['normalize_distance'],
                                device=device,
                                use_se=True, use_stemse=True, use_finalse=True).to(device)
            print("Using ResNet-Conformer!")
        else:
            use_niu = False if "max" in job_id.lower() else True
            model = ResNet(in_feat_shape=data_in,
                           out_feat_shape=data_out,
                           normd=params['normalize_distance'],
                           niu_resnet=use_niu).to(device)
            print("Using ResNet-GRU!")

        write_and_print(logger, 'FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out)) # Get input and output shape sizes
        print(model)

        # Dump results in DCASE output format for calculating final scores
        dcase_output_val_folder = os.path.join(params['dcase_output_dir'], '{}_{}_val'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
        cls_feature_class.delete_and_create_folder(dcase_output_val_folder)
        write_and_print(logger, 'Dumping recording-wise val results in: {}'.format(dcase_output_val_folder))

        # Initialize evaluation metric class
        score_obj = ComputeSELDResults(params)

        # start training
        best_val_epoch = -1
        best_ER, best_F, best_LE, best_LR, best_seld_scr, best_dist_err, best_rel_dist_err = 1., 0., 180., 0., 9999, 999999., 999999.
        best_val_seld_error = 1.

        # Setting up number of training epochs
        nb_epoch = params['nb_epochs']
        print("Training the model for a total of : {} epochs.".format(nb_epoch))

        # Change back to original Adam
        if "adam" in job_id.lower(): 
            optimizer = optim.Adam(model.parameters(), lr=params["lr"])
            print("Using vanilla Adam")
        else:

            # Define weight decay settings (exclude bias and normalization layers)
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    'weight_decay': 5e-2 if "cst" in job_id.lower() else 1e-4
                },
                {
                    'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0
                }
            ]

            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=params["lr"])
            print("Using Adam with Weight Decay optimizer")

        # Now we set the learning rate scheduler
        batch_scheduler = None
        step_scheduler = None
        if "cosine" in job_id.lower():

            num_batches_per_epoch = len(training_dataloader)
            total_steps = nb_epoch * num_batches_per_epoch

            step_scheduler = CosineWarmup_StepScheduler(optimizer, total_steps=total_steps)
            print("Cosine Annealing w/ Warmup Step Scheduler is used!\nTotal Number of Steps : {}".format(total_steps))

        elif params['scheduler'] == "cosine_step":

            batch_scheduler = CosineWarmupScheduler(optimizer, warmup=5, max_iters=nb_epoch)
            print("Cosine annealing w/ Warmup Batch scheduler is used!")

        else:
            batch_scheduler = DecayScheduler(optimizer, min_lr=params['final_lr'])
            print("Decay Scheduler used!")

        optimizer.zero_grad()
        optimizer.step()

        # Defining the loss function to be used, which is dependent on our output format
        criterion = seldnet_model.MSELoss_ADPIT()
        if params['finetune_mode']:
            model.load_state_dict(torch.load(params['model_weights'], map_location='cpu'))

        # Misc. print statements for viewing the training configurations
        write_and_print(logger, "Device used : {}".format(device))
        write_and_print(logger, "Using Augmentations : {}".format("trans" in job_id.lower()))
        write_and_print(logger, "Normalize Distance : {}".format(params['normalize_distance']))
        write_and_print(logger, "Number of params : {:.3f}M".format(count_parameters(model)/(10**6)))

        try:
            for epoch_cnt in range(nb_epoch):
                # ---------------------------------------------------------------------
                # TRAINING
                # ---------------------------------------------------------------------
                start_time = time.time()
                train_loss = train_epoch(training_dataloader, optimizer, model, criterion, device, scheduler=batch_scheduler, nb_batches = n_batches, step_scheduler=step_scheduler)
                train_time = time.time() - start_time

                # ---------------------------------------------------------------------
                # VALIDATION
                # ---------------------------------------------------------------------

                if "fast" in job_id:
                    interval = 80
                elif "full" in job_id:
                    interval = 1
                else:
                    interval = 20 if "cst" in job_id.lower() else 5

                if (epoch_cnt > int(0.80 * nb_epoch)) or \
                    (epoch_cnt % interval == 0):

                    start_time = time.time()
                    val_loss = test_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device)
                    # Getting the validation DCASE metrics
                    val_ER, val_F, val_LE, val_dist_err, val_rel_dist_err, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_folder)
                    if "seld" in job_id.lower():
                        val_seld_err = ((1 - val_F) + (val_LE / 180) + val_rel_dist_err) / 3
                    else:
                        val_seld_err = (2 * (1 - val_F) + (val_LE / 180) + val_rel_dist_err) / 4
                    if math.isnan(val_seld_err):
                        val_seld_err = 1.

                    # Save model if F-score is good
                    if epoch_cnt < int(0.1 * nb_epoch) or "f1" in job_id:
                        indicator_signal = val_F >= best_F
                    else:
                        indicator_signal = val_seld_err <= best_val_seld_error

                    if indicator_signal:
                        best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr, best_dist_err = epoch_cnt, val_ER, val_F, val_LE, val_LR, val_seld_scr, val_dist_err
                        best_rel_dist_err = val_rel_dist_err
                        best_val_seld_error = val_seld_err
                        torch.save(model.state_dict(), model_name)

                    val_time = time.time() - start_time

                # ---------------------------------------------------------------------
                # LOGGING METRICS AND VARIABLES
                # ---------------------------------------------------------------------

                # Print stats
                write_and_print(logger, 
                    'epoch: {}, time: {:0.2f}/{:0.2f}, '
                    'train_loss: {:0.4f}, val_loss: {:0.4f}, '
                    'F/AE/DE/RDE/SELD: {}, '
                    'best_val_epoch: {} {}'.format(
                        epoch_cnt, train_time, val_time,
                        train_loss, val_loss,
                        '{:0.3f}/{:0.2f}/{:0.3f}/{:0.3f}/{:0.3f}'.format(val_F, val_LE, val_dist_err, val_rel_dist_err, val_seld_err),
                        best_val_epoch,
                        '({:0.3f}/{:0.2f}/{:0.3f}/{:0.3f}/{:0.3f})'.format(best_F, best_LE, best_dist_err, best_rel_dist_err, best_val_seld_error))
                )

        except KeyboardInterrupt:
            write_and_print(logger, "Training ended prematurely. Calculating results now.")

        # ---------------------------------------------------------------------
        # Evaluate on unseen test data
        # ---------------------------------------------------------------------

        print('Load best model weights')
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

        # In this case, the unseen test dataset is just the validation dataset but still want to get the classwise results
        print('Loading unseen test dataset:')
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=test_splits[split_cnt], shuffle=False, per_file=True
        )

        # Dump results in DCASE output format for calculating final scores
        dcase_output_test_folder = os.path.join(params['dcase_output_dir'], '{}_{}_test'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
        cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
        write_and_print(logger, 'Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))

        test_loss = test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device)

        use_jackknife=True
        test_ER, test_F, test_LE, test_dist_err, test_rel_dist_err, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(dcase_output_test_folder, is_jackknife=use_jackknife )

        write_and_print(logger, 'SELD score (early stopping metric): {:0.3f} {}'.format(test_seld_scr[0] if use_jackknife else test_seld_scr, '[{:0.2f}, {:0.2f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''))
        write_and_print(logger, 'SED metrics: F-score: {:0.3f} {}'.format(100* test_F[0]  if use_jackknife else 100* test_F, '[{:0.2f}, {:0.2f}]'.format(100* test_F[1][0], 100* test_F[1][1]) if use_jackknife else ''))
        write_and_print(logger, 'DOA metrics: Angular error: {:0.3f} {}'.format(test_LE[0] if use_jackknife else test_LE, '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else ''))
        write_and_print(logger, 'Distance metrics: {:0.3f} {}'.format(test_dist_err[0] if use_jackknife else test_dist_err, '[{:0.2f} , {:0.2f}]'.format(test_dist_err[1][0], test_dist_err[1][1]) if use_jackknife else ''))
        write_and_print(logger, 'Relative Distance metrics: {:0.3f} {}'.format(test_rel_dist_err[0] if use_jackknife else test_rel_dist_err, '[{:0.2f} , {:0.2f}]'.format(test_rel_dist_err[1][0], test_rel_dist_err[1][1]) if use_jackknife else ''))

        write_and_print(logger, 'Classwise results on unseen test data')
        write_and_print(logger, 'Class\tF\t\t\tAE\tdist_err\treldist_err\tSELD_score')
        for cls_cnt in range(params['unique_classes']):
            write_and_print(logger, 
                            '{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                                cls_cnt,
                                classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
                                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                                            classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                                classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
                                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                                            classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                                classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
                                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                                            classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                                classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
                                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                                            classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else '',
                                classwise_test_scr[0][6][cls_cnt] if use_jackknife else classwise_test_scr[6][cls_cnt],
                                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][6][cls_cnt][0],
                                                            classwise_test_scr[1][6][cls_cnt][1]) if use_jackknife else ''))

if __name__ == "__main__":
    # Record the start time
    start_time = time.time()

    try:
        # Execute the main function
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        # Handle exceptions and exit with the error
        sys.exit(e)
    finally:
        # Record the end time
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time

        # Convert elapsed_time to a human-readable format
        # One minute or more: display in minutes and seconds
        hours = int(elapsed_time // 3600)
        remaining_time = elapsed_time % 3600
        minutes = int(remaining_time // 60)
        seconds = remaining_time % 60
        print(f"Execution time: {hours}h {minutes}min {seconds:.2f}s")
