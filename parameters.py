# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.
def get_params(argv='1', verbose=True):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,  # To do quick test. Trains/test on small subset of dataset, and # of epochs

        finetune_mode=False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='3_1_dev_split0_multiaccdoa_foa_model.h5',
        model_weights = "",

        # INPUT PATH
        dataset_dir='./DCASE2024_SELD_dataset/',

        # OUTPUT PATHS
        feat_label_dir='./DCASE2024_SELD_dataset/seld_feat_label/',

        model_dir='models',  # Dumps the trained models and training curves in this folder
        dcase_output_dir='results',  # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',  # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',  # 'foa' - ambisonic or 'mic' - microphone signals
        unique_classes = 13, # fixed for DCASE 2024, don't change

        # FEATURE PARAMS
        nb_channels = 2,
        fs=24000,
        hop_len_s=0.0125,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=256,
        use_salsalite=False,  # If MIC - SALSA-Lite, FOA - SALSA (FOA)
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_salsa_foa = 9000,
        fmax_salsa_mic = 4000,
        fmax_spectra_salsalite=9000,

        # MODEL TYPE
        modality='audio',  # 'audio' or 'audio_visual'
        multi_accdoa=True,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=30,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=50,   # Feature sequence length
        batch_size=32,              # Batch size
        dropout_rate=0.1,          # Dropout rate, constant for all layers
        nb_cnn2d_filt=128,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 4],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_heads=8,
        nb_self_attn_layers=2,
        nb_transformer_layers=2,

        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=256,  # FNN contents, length of list = number of layers, list value = number of nodes

        # Learning/Dataset gen parameters
        nb_epochs=100,              # Train for maximum epochs
        lr=3e-4,                    # Initial learning rate
        milestones = (0.1, 0.7),    # Tuple of two floats, first for warmup epochs, last for finetune epochs
        final_lr = 1e-5,            # Final learning rate
        use_augmentations = False,  # Default just always do augmentations unless special testing
        scheduler = "none",         # Default always just use a learning rate scheduler, choose one of (cosine, custom, noam)
        normalize_distance = False, # Default to just leaving in terms of meters(m)

        # Model Selection
        use_resnet = False, # Use vanilla ResNet

        # Additional folds
        use_fold1 = False, # Synth train
        use_fold2 = False, # Synth test
        use_fold5 = False, # Audio channel swapped
        use_fold6 = False, # Spatial Scaper generated dataset using all rooms (Large Self-Synth)
        use_fold7 = False, # Spatial Scaper generation that is within distance distribution of STARSS23 (Small Self-Synth)
        use_fold8 = False, # Only ACS data
        use_fold9 = False, # Only MIC SpatialScaper data

        # METRIC
        average='macro',                 # Supports 'micro': sample-wise average and 'macro': class-wise average,
        segment_based_metrics=False,     # If True, uses segment-based metrics, else uses frame-based metrics
        evaluate_distance=True,          # If True, computes distance errors and apply distance threshold to the detections
        lad_doa_thresh=20,               # DOA error threshold for computing the detection metrics
        lad_dist_thresh=float('inf'),    # Absolute distance error threshold for computing the detection metrics
        lad_reldist_thresh=float('1'),  # Relative distance error threshold for computing the detection metrics

        # CST-Former Specific Metrics
        t_pooling_loc = "front",
        FreqAtten = True,
        ChAtten_ULE = True,
        CMT_block = True
    )

    ############ User defined parameters ##############
    
    if argv == "bin":
        print("Baseline + Binaural Stack + DCASE Dataset + No Dnorm")
        params['dataset'] = 'bin'
        params['normalize_distance'] = False
        
    elif argv == "bin_d":
        print("Baseline + Binaural Stack + DCASE Dataset + Dnorm")
        params['dataset'] = 'bin'
        params['normalize_distance'] = True
    
    elif argv == "bin_synd":
        print("Baseline + Binaural Stack + DCASE Synthetic Dataset + Dnorm")
        params['dataset'] = 'bin'
        params['normalize_distance'] = True
        params['use_fold1'] = True
        params['use_fold2'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    # Defining the feat label dir based on parameter selection (prevent forgetting to change)
    feature_format = "binaural"

    # Normalized Distances
    if params['normalize_distance']:
        normd = "dnorm"
    else:
        normd = "no_dnorm"

    # Additional Fold Data
    add_fold = ""
    if params['use_fold1']:
        add_fold += "1"
    if params['use_fold2']:
        add_fold += "2"
    if params["use_fold5"]:
        add_fold += "5"
    if params['use_fold6']:
        add_fold += "6"
    if params['use_fold7']:
        add_fold += "7"
    if params['use_fold8']:
        add_fold += "8"
    if params['use_fold9']:
        add_fold += "9"

    final_folder = "_".join([feature_format, normd, add_fold])
    params['feat_label_dir'] = "./DCASE2024_SELD_dataset/seld_feat_label/{}".format(final_folder)

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s']) # Default : 5 / 8 for (0.1s divided by 0.01/0.0125 hop_len_s)
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution # Default : 250 / 400
    params['t_pool_size'] = [feature_label_resolution, 1, 1] # pooling at the end is just better practice 


    params['patience'] = int(params['nb_epochs'])  # Stop training if patience is reached
    params['model_dir'] = params['model_dir'] + '_' + params['modality']
    params['dcase_output_dir'] = params['dcase_output_dir'] + '_' + params['modality']

    # If we want to normalize distances, we need these dmean, dmax, dstddev for normalization,
    # and these will vary based on the dataset we choose to use
    if params['normalize_distance']:
        if params["use_fold2"]:
            if params["use_fold1"]: 
                if params['use_fold5']: # Fold 1 + Fold 2 + ACS + STARSS23
                    params['dmean'] = 2.6054558397352534
                    params['dstd'] = 1.2478729644216462
                    params['dmax'] = 3.641832373835437
                else: # Fold 1 + Fold 2 + STARSS23
                    params['dmean'] = 3.0541368010205234 
                    params['dstd'] = 1.3731086519097455
                    params['dmax'] = 2.9829126728484834
            else: # Fold 2 + STARSS23
                params['dmean'] = 2.7827428036270456 
                params['dstd'] = 1.2485838291723967
                params['dmax'] = 3.3936505482227415
        else:
            if params['use_fold5']: # ACS + STARSS23
                params['dmean'] = 1.9727609207444998 
                params['dstd'] = 0.6753750604758425
                params['dmax'] = 7.5
            else: # STARSS23
                params['dmean'] = 1.8981285708779436 
                params['dstd'] = 0.6642250166015196
                params['dmax'] = 8
        
        # For DCASE Submission
        if params['use_fold5'] and params['use_fold6']:
            params['dmean'] = 2.48237649623756 
            params['dstd'] = 1.1750451332433416
            params['dmax'] = 4.38929821318986
            print("FOLDS USED FOR TRAINING -- FOLD 3,5 AND 6")
            print("\t\tSTARSS : 3\n\t\tACS SUBSET : 5\n\t\tSpatial Scaper : 6")
            
        if params['use_fold8']:
            params['dmean'] = 2.48237649623756 
            params['dstd'] = 1.1750451332433416
            params['dmax'] = 4.38929821318986
            print("Using Fold 8, using the same scaling weights for ACS realdata")
        
        # For DCASE Task work
        if params['use_fold9']:
            params['dmean'] = 2.060804351919315  
            params['dstd'] = 0.7985113234984598
            params['dmax'] = 6.210551437584279 
            print("This is only meant for MIC dataset!!!")
        
        if params['use_fold5'] and params['use_fold9']:
            params['dmean'] = 2.026096046056667 
            params['dstd'] = 0.7398851275626984
            params['dmax'] = 6.7495666123119165 
            print("This is only meant for MIC dataset final ablation experiment!")

        print("Mean : {} , Max : {} , Std.Dev : {}".format(params['dmean'], params['dmax'], params['dstd']))

    if verbose:
        for key, value in params.items():
            print("\t{}: {}".format(key, value))
    return params