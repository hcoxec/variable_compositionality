function(command_line_args={}) {
    project_name: 'variable_comp',
    wandb_name: 'variable_comp',
    local _default_args = {
        #Defaults that can be overwritten by command line args
        exp_id: 1000,
        att: 2,
        params: 250,
        wkdir: '/Users/hcoxec/research/public_code/variable_compositionality',
        logdir: '/Users/hcoxec/research/public_code/variable_compositionality/checkpoints',
        eval_section: "test_mcd",  # gen_samples, gen, test
        sender_entropy: 0.5,
        decay_coeff: 0.0,
        dropout: 0.0,
        mode: null,
        task: 'train',
        max_epochs: 800,
    },
    args: _default_args + command_line_args,

    #DATA
    dataset: 'two_predicates',
    all_splits:['train_mcd', 'val_mcd', 'test_mcd', 'iid_test_mcd', 'all_mcd'],
    dependency_parse: true,
    preprocess_version: '0.01',
    data_scaling: 20,
    input_one_hot: true,
    output_one_hot: true,
    n_roles: 3,
    n_atoms: 52, #total across all roles
    n_atoms_per_role: [25,25,25],
    
    #TRAINING
    train_split: 'train_mcd',
    val_split: 'test_mcd',
    wkdir: $.args.wkdir,
    run_id: ['hidden', $.args.params, 'd', $.args.dropout, 'l', $.args.decay_coeff],
    seed: $.args.att,
    epochs: 15,
    load_from_checkpoint: null,
    batch_size: 5000,
    learning_rate: 0.001,
    sender_entropy: $.args.sender_entropy,
    decay_coeff: $.args.decay_coeff,
    sender_dropout: $.args.dropout,
    receiver_dropout: $.args.dropout,
    callbacks:['saver', 'analysis'], #wandb
    validation_freq:20,
    update_freq: 1,
    entropy_update: 1,
    use_wandb: false,
    early_stopping_thr: 0.999999, #currently disabled

    #IN STREAM ANALYSIS
    analysis_splits:['test_mcd', 'iid_test_mcd'],
    topsim_freq:1,
    posdis_freq:1,

    #MODEL
    rnn_cell: 'gru',
    hidden_size: $.args.params,
    embedding_size: 52,
    signal_len: 6,
    signal_alphabet_size: 26,
    check_every: 1,

    #EVALUATION
    eval_splits:['all_mcd'],
    eval_steps: {'start':1, 'stop': $.epochs, 'step': $.check_every},
    metrics: ['variation', 'topsim', 'posdis'],

    #ETC
    notebook: false, #tries to do some stuff to make things run well in nb

}