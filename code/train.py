import torch


from torch.utils.data import DataLoader



from loss import ReconstructionLoss as Loss
from agents import SenderInput, ReceiverOutput, RnnSenderReinforce, RnnReceiverDeterministic
from game import ReinforceGame as Game
from utils import registry, setup_streamer

from callbacks import CheckpointSaver, WandbLogger, StreamAnalysis
from trainer import Trainer

def build_model(config):
    data_size = config.n_roles*config.n_atoms
    in_size, out_size = data_size, data_size
    sender_in = SenderInput(in_size, config.hidden_size, dropout=config.sender_dropout)
    receiver_out = ReceiverOutput(out_size, config.hidden_size)

    sender = RnnSenderReinforce( #trained via RL
        sender_in, 
        config.signal_alphabet_size, 
        config.embedding_size, 
        config.hidden_size,
        cell=config.rnn_cell, 
        max_len=config.signal_len
    )
    receiver = RnnReceiverDeterministic( #trained via standard SGD
        receiver_out, 
        config.signal_alphabet_size, 
        config.embedding_size, 
        config.hidden_size,
        cell=config.rnn_cell, 
        dropout_p=config.receiver_dropout
    )

    loss = Loss(config.n_roles, config.n_atoms)

    game = Game(
        sender, 
        receiver, 
        loss, 
        sender_entropy_coeff=config.sender_entropy, 
        receiver_entropy_coeff=0.0, #RL isn't used for the receiver
    )

    return game


def reconstruction_game(config):
    print("Beginning Run Setup")
    #Create a streamer to write run data to disk
    streamer = setup_streamer(config)

    #load the desired dataset class from the object registry
    Dataset = registry.lookup("dataset", config.dataset)

    
    all_data = {}
    for split in config.all_splits:
        all_data[split] = Dataset(config, split, scaling=1)

    #instantiate Training Dataset and loader
    train_data = all_data[config.train_split]
    train_data.data_scaling = config.data_scaling
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    #Also setup data for each validation step
    val_data = all_data[config.val_split]
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=True)

    print(f'Current Training Data Size: {train_data.true_len}')
    print(f'Current Scaled Training Data Size: {len(train_data)}')
    print(f'Current Validation Data Size: {len(val_data)}')
    
    #Some analyses need to know how many roles/atoms the dataset has
    #if not specified in config, infer from the training data
    try:
        n_roles = config.n_roles
        n_atoms = config.n_atoms

    except Exception as e:
        #get dependency role labels as a proxy for roles
        n_roles = len(train_data.enc_lang.dep_words.keys())
        #get words in the input langueage as a proxy for atoms
        n_atoms = train_data.enc_lang.n_words

        setattr(config, 'n_roles', n_roles)
        setattr(config, 'n_atoms', n_atoms)
        print(f'attributes and values assumed: {n_roles}, {n_atoms}')
    
    game = build_model(config)

    callbacks = []
    print(f"logdir: {config.logdir}")
    if 'saver' in config.callbacks:
        checkpoint_dir = '{}/{}/{}/{}/seed_{}'.format(
                    config.logdir,
                    config.dataset,
                    config.project_name,
                    config.run_id,
                    config.seed
                )
        callbacks.append(
            CheckpointSaver(
                    checkpoint_path=checkpoint_dir,
                    checkpoint_freq=config.check_every,
                    prefix=f'sd_{config.seed}_hd_{config.hidden_size}_sg_{config.signal_len}'
                )
        )
    
    if 'analysis' in config.callbacks:
        analysis_splits = [all_data[x] for x in config.analysis_splits]
        callbacks.append(
            StreamAnalysis(
                train_data=train_data,
                test_data=analysis_splits,
                streamer=streamer,
                config=config
            )
        )
    
    if 'wandb' in config.callbacks:
        callbacks.append(
            WandbLogger(
                config=config
            )
        )

    optimizer = torch.optim.Adam(
        [
            {'params': game.sender.parameters(), 'lr': config.learning_rate},
            {'params': game.receiver.parameters(), 'lr': config.learning_rate}
        ]
    )

    trainer = Trainer(
        config=config,
        streamer=streamer,
        game=game, 
        optimizer=optimizer, 
        train_data=train_loader,
        validation_data=val_data,
        wandb_project= config.wandb_name,
        callbacks=callbacks,
    )
    
    if config.notebook:
        return trainer
    
    else:
        trainer.train(config.epochs)

def main(config):
    reconstruction_game(config)
