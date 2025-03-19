import os
import numpy as np
import cls_feature_class
import cls_data_generator
import parameters
from time import gmtime, strftime
import torch
import sys
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from SELD_evaluation_metrics import distance_between_cartesian_coordinates
import seldnet_model as seldnet_model
from ext_resconf import External_ResConf
from all_models import ResNetConf18, CRNN_Conformer, Seld_CBAM, DCASE_Model, ResNetConf8


def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = np.sqrt(x**2 + y**2 + z**2) > 0.5

    return sed, accdoa_in


def get_multi_accdoa_labels(accdoa_in, nb_classes, params=None):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]
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

def eval_epoch(data_generator, model, dcase_output_folder, params, device):
    eval_filelist = data_generator.get_filelist()
    model.eval()
    file_cnt = 0
    with torch.no_grad():
        for values in data_generator.generate():
            if len(values) == 2: # audio visual
                data, vid_feat = values
                data, vid_feat = torch.tensor(data).to(device).float(), torch.tensor(vid_feat).to(device).float()
                output = model(data, vid_feat)
            else:
                data = values
                data = torch.tensor(data).to(device).float()
                output = model(data)

            if params['multi_accdoa'] is True:
                sed_pred0, doa_pred0, dist_pred0, sed_pred1, doa_pred1, dist_pred1, sed_pred2, doa_pred2, dist_pred2 = get_multi_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred0 = reshape_3Dto2D(sed_pred0)
                doa_pred0 = reshape_3Dto2D(doa_pred0)
                dist_pred0 = reshape_3Dto2D(dist_pred0)
                sed_pred1 = reshape_3Dto2D(sed_pred1)
                doa_pred1 = reshape_3Dto2D(doa_pred1)
                dist_pred1 = reshape_3Dto2D(dist_pred1)
                sed_pred2 = reshape_3Dto2D(sed_pred2)
                doa_pred2 = reshape_3Dto2D(doa_pred2)
                dist_pred2 = reshape_3Dto2D(dist_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred = reshape_3Dto2D(sed_pred)
                doa_pred = reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file

            output_file = os.path.join(dcase_output_folder, eval_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}
            if params['multi_accdoa'] is True:
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
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt]>0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+params['unique_classes']], doa_pred[frame_cnt][class_cnt+2*params['unique_classes']]])
            data_generator.write_output_format_file(output_file, output_dict)

def test_epoch(data_generator, model, criterion, dcase_output_folder, params, device,
               extreme_verbose=False):
    # Number of frames for a 60 second audio with 100ms hop length = 600 frames
    # Number of frames in one batch (batch_size* sequence_length) consists of all the 600 frames above with zero padding in the remaining frames
    test_filelist = data_generator.get_filelist()

    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0
    with torch.no_grad():
        for values in data_generator.generate():
            if len(values) == 2:
                data, target = values
                data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
                output = model(data)

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

            # dump SELD results to the correspondin file
            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
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

        test_loss /= nb_test_batches

    return test_loss


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

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    job_id = 1 if len(argv) < 3 else argv[-1]

    # Training setup 
    """ Data splits definition """
    test_splits = [[4]]
    val_splits = [[4]]
    train_splits = [[3]]
    
    if params['mode'] == 'dev':

        for split_cnt, split in enumerate(test_splits):
            print('\n\n---------------------------------------------------------------------------------------------------')
            print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
            print('---------------------------------------------------------------------------------------------------')

            # Unique name for the run
            loc_feat = params['dataset']
            if params['dataset'] == 'mic':
                if params['use_salsalite']:
                    loc_feat = '{}_salsa'.format(params['dataset'])
                else:
                    loc_feat = '{}_gcc'.format(params['dataset'])
            loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'

            cls_feature_class.create_folder(params['model_dir'])
            unique_name = '{}_{}_{}_split{}_{}_{}'.format(
                task_id, job_id, params['mode'], split_cnt, loc_output, loc_feat
            )
            model_name = '{}_model.h5'.format(os.path.join(params['model_dir'], unique_name))

            # Load train and validation data
            print('Loading training dataset:')
            data_gen_train = cls_data_generator.DataGenerator(
                params=params, split=train_splits[split_cnt]
            )

            if params['use_dcase_model']:
                data_in, data_out = data_gen_train.get_data_sizes()
                model = DCASE_Model(in_feat_shape=data_in, out_feat_shape=data_out,
                                    normd = params['normalize_distance'], use_se = params['add_se'],
                                    use_stemse = params['add_stemse']).to(device)

            elif params["use_ext_conformer"]:
                data_in, data_out = data_gen_train.get_data_sizes()
                model = External_ResConf(n_input_channels=data_in[1], p_dropout=0.1, device=device, norm_d=params['normalize_distance']).to(device)

            elif params["use_resnet_conformer"]:
                data_in, data_out = data_gen_train.get_data_sizes()
                model = ResNetConf18(in_feat_shape  = data_in, 
                                    out_feat_shape = data_out,
                                    normd          = params['normalize_distance'],
                                    num_classes    = params['unique_classes'],
                                    use_se         = params['add_se'], 
                                    use_stemse     = params['add_stemse'],
                                    use_finalse    = params['use_finalse'], 
                                    use_cbam       = params['use_cbam'],
                                    use_taconf     = params['torchaudioconf'],
                                    device         = device).to(device)
            
            elif params['use_resnet8']:
                data_in, data_out = data_gen_train.get_data_sizes()
                model = ResNetConf8(in_feat_shape  = data_in, 
                                    out_feat_shape = data_out,
                                    normd          = params['normalize_distance'],
                                    use_se         = params['add_se'], 
                                    use_finalse    = params['use_finalse'], 
                                    use_cbam       = params['use_cbam'],
                                    use_taconf     = params['torchaudioconf']).to(device)

            elif params['use_crnn_conformer']:
                data_in, data_out = data_gen_train.get_data_sizes()
                model = CRNN_Conformer(in_feat_shape=data_in, out_feat_shape=data_out,
                                    norm_distance=params['normalize_distance'],
                                    p_dropout=0.1, use_finalse=params['use_finalse']).to(device)

            elif params['use_seld_cbam']:
                data_in, data_out = data_gen_train.get_data_sizes()
                model = Seld_CBAM(in_feat_shape=data_in, out_shape=data_out,
                                params=params).to(device)

            else: # Use DCASE Baseline architecture
                data_in, data_out = data_gen_train.get_data_sizes()
                model = seldnet_model.SeldModel(data_in, data_out, params).to(device)

            # Dump results in DCASE output format for calculating final scores
            dcase_output_val_folder = os.path.join(params['dcase_output_dir'], '{}_{}_val'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
            cls_feature_class.delete_and_create_folder(dcase_output_val_folder)

            # Initialize evaluation metric class
            score_obj = ComputeSELDResults(params)
            
            # Defining the loss function to be used, which is dependent on our output format
            criterion = seldnet_model.MSELoss_ADPIT()

            # ---------------------------------------------------------------------
            # Evaluate on unseen test data
            # ---------------------------------------------------------------------

            print('Load best model weights from : {}'.format(params['model_weights']))
            model.load_state_dict(torch.load(params['model_weights'], map_location='cpu'))

            # In this case, the unseen test dataset is just the validation dataset but still want to get the classwise results
            print('Loading unseen test dataset:')
            data_gen_test = cls_data_generator.DataGenerator(
                params=params, split=test_splits[split_cnt], shuffle=False, per_file=True
            )

            # Dump results in DCASE output format for calculating final scores
            dcase_output_test_folder = os.path.join(params['dcase_output_dir'], '{}_{}_test'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
            cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
            print('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))

            test_loss = test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device)

            use_jackknife=True
            test_ER, test_F, test_LE, test_dist_err, test_rel_dist_err, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(dcase_output_test_folder, is_jackknife=use_jackknife )

            print('SELD score (early stopping metric): {:0.3f} {}'.format(test_seld_scr[0] if use_jackknife else test_seld_scr, '[{:0.2f}, {:0.2f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''))
            print('SED metrics: F-score: {:0.3f} {}'.format(100* test_F[0]  if use_jackknife else 100* test_F, '[{:0.2f}, {:0.2f}]'.format(100* test_F[1][0], 100* test_F[1][1]) if use_jackknife else ''))
            print('DOA metrics: Angular error: {:0.3f} {}'.format(test_LE[0] if use_jackknife else test_LE, '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else ''))
            print('Distance metrics: {:0.3f} {}'.format(test_dist_err[0] if use_jackknife else test_dist_err, '[{:0.2f} , {:0.2f}]'.format(test_dist_err[1][0], test_dist_err[1][1]) if use_jackknife else ''))
            print('Relative Distance metrics: {:0.3f} {}'.format(test_rel_dist_err[0] if use_jackknife else test_rel_dist_err, '[{:0.2f} , {:0.2f}]'.format(test_rel_dist_err[1][0], test_rel_dist_err[1][1]) if use_jackknife else ''))

            print('Classwise results on unseen test data')
            print('Class\tF\tAE\tdist_err\treldist_err\tSELD_score')
            for cls_cnt in range(params['unique_classes']):
                print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
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

    if params['mode'] == 'eval':

        print('Loading evaluation dataset:')
        data_gen_eval = cls_data_generator.DataGenerator(
            params=params, shuffle=False, per_file=True, is_eval=True)

        if params['use_dcase_model']:
            data_in, data_out = data_gen_train.get_data_sizes()
            model = DCASE_Model(in_feat_shape=data_in, out_feat_shape=data_out,
                                normd = params['normalize_distance'], use_se = params['add_se'],
                                use_stemse = params['add_stemse']).to(device)

        elif params["use_ext_conformer"]:
            data_in, data_out = data_gen_train.get_data_sizes()
            model = External_ResConf(n_input_channels=data_in[1], p_dropout=0.1, device=device, norm_d=params['normalize_distance']).to(device)

        elif params["use_resnet_conformer"]:
            data_in, data_out = data_gen_train.get_data_sizes()
            model = ResNetConf18(in_feat_shape  = data_in, 
                                out_feat_shape = data_out,
                                normd          = params['normalize_distance'],
                                num_classes    = params['unique_classes'],
                                use_se         = params['add_se'], 
                                use_stemse     = params['add_stemse'],
                                use_finalse    = params['use_finalse'], 
                                use_cbam       = params['use_cbam'],
                                use_taconf     = params['torchaudioconf'],
                                device         = device).to(device)
        
        elif params['use_resnet8']:
            data_in, data_out = data_gen_train.get_data_sizes()
            model = ResNetConf8(in_feat_shape  = data_in, 
                                out_feat_shape = data_out,
                                normd          = params['normalize_distance'],
                                use_se         = params['add_se'], 
                                use_finalse    = params['use_finalse'], 
                                use_cbam       = params['use_cbam'],
                                use_taconf     = params['torchaudioconf']).to(device)

        elif params['use_crnn_conformer']:
            data_in, data_out = data_gen_train.get_data_sizes()
            model = CRNN_Conformer(in_feat_shape=data_in, out_feat_shape=data_out,
                                norm_distance=params['normalize_distance'],
                                p_dropout=0.1, use_finalse=params['use_finalse']).to(device)

        elif params['use_seld_cbam']:
            data_in, data_out = data_gen_train.get_data_sizes()
            model = Seld_CBAM(in_feat_shape=data_in, out_shape=data_out,
                            params=params).to(device)

        else: # Use DCASE Baseline architecture
            data_in, data_out = data_gen_train.get_data_sizes()
            model = seldnet_model.SeldModel(data_in, data_out, params).to(device)

        print('Load best model weights from : {}'.format(params['model_weights']))
        model.load_state_dict(torch.load(params['model_weights'], map_location='cpu'))

        # Dump results in DCASE output format for calculating final scores
        loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'

        dcase_output_test_folder = os.path.join(params['dcase_output_dir'], '{}_{}_{}_eval'.format(params['dataset'], loc_output, strftime("%Y%m%d%H%M%S", gmtime())))
        cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
        print('Dumping recording-wise eval results in: {}'.format(dcase_output_test_folder))

        eval_epoch(data_gen_eval, model, dcase_output_test_folder, params, device)
        
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)