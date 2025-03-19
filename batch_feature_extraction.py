# Extracts the features, labels, and normalizes the development and evaluation split features.

import cls_feature_class
import parameters
import time
import sys


def main(argv):
    # Expects one input - task-id - corresponding to the configuration given in the parameter.py file.
    # Extracts features and labels relevant for the task-id
    # It is enough to compute the feature and labels once. 

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    # -------------- Extract features and labels for development set -----------------------------
    if params['mode'] == 'dev':
        dev_feat_cls = cls_feature_class.FeatureClass(params, is_eval=False)

        # # Extract features and normalize them
        dev_feat_cls.extract_all_feature()
        dev_feat_cls.preprocess_features()

        # # Extract labels
        dev_feat_cls.extract_all_labels()

    else:
        dev_feat_cls = cls_feature_class.FeatureClass(params, is_eval=True)

        # # Extract features and normalize them
        dev_feat_cls.extract_all_feature()
        dev_feat_cls.preprocess_features()



if __name__ == "__main__":
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
