import os

import preprocess, train, eval
from utils.setup import fetch_args, parse_args, load_arg_file, set_seeds


def main():
    args = parse_args()
    config = load_arg_file(args)
    config.mode = args.mode
    
    print("===================================================")
    print(f"Config Imported from {args.exp_config_file}")
    print(f"Run ID: {config.run_id}")
    
    #sometimes on the cluster the dir where the job is started is different from
    #the dir on the node we want to use - so this moves us to the working directory
    #'wkdir' specified in the config
    os.chdir(config.wkdir) 

    set_seeds(config)
    print("===================================================")
    
    if args.mode == 'preprocess':
        preprocess.main(config)

    if args.mode == 'train':
        train.main(config)

    if args.mode == 'eval':
        eval.main(config)
    


if __name__ == '__main__':
    main()
