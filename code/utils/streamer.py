import json

from os import makedirs, path, remove

class Streamer(object):

    def __init__(self, config) -> None:
        self.config=config
        self.wkdir = config.wkdir if config.wkdir[-1] == '/' else config.wkdir+'/'
        self.file_suffix = getattr(config, "stream_file_suffix", None)
        self.mode = config.mode
        self.step = 0

        self.state = {}

        self.create_out_dir(
            working_directory=config.wkdir,
            dataset=config.dataset,
            project_name=config.project_name,
            run_id=config.run_id,
            seed=config.seed,
            file_suffix=self.file_suffix
        )

    def create_out_dir(
        self, working_directory, dataset, project_name, run_id, seed, file_suffix=None
        ):
        
        self.save_dir = '{}/run_data/{}/{}/{}/seed_{}'.format(
            working_directory,
            dataset,
            project_name,
            run_id,
            seed
        )

        makedirs(self.save_dir, exist_ok= True)
        self.file_name = f'sd_{seed}_{file_suffix}' if file_suffix else f'sd_{seed}_data'
        self.save_path = f"{self.save_dir}/{self.file_name}.jsonl"

    def start_run(self):
        '''
        Overwrites any existing output file for a given config
        Because a single run can initialize the streamer many times (e.g. for training
        eval, revaluation, etc.) by default if the output file exists it's just added to.
        This command creates a new file, or erases the existing one
        '''

        if path.exists(self.save_path):
            remove(self.save_path)

        self.record(data=self.config.__dict__, dtype='config')

    def add(self, data: dict):
        '''
        Allows accumulation of results before writing. data is added to self.state
        but not written to outfile
        '''
        self.state.update(data)

    def record_state(self, dtype:str ='step'):
        '''
        dumps self.state to outfile and clears the state
        '''
        self.record(data=self.state, dtype=dtype)
        self.state = {}

    def record(self, data: dict, dtype: str):
        '''
        Writes data to outfile, adds some run params beforehand (to help with analysis)
        additional params to track can be added in the run config by
        adding a list of config param names as "stream_params"
        '''
        data['type'] = dtype
        data['step'] = self.step
        data['mode'] = self.mode
        data['seed'] = self.config.seed

        if getattr(self.config, "stream_params", None):
            for param in self.config.stream_params:
                data[param] = self.config.__dict__[param]

        with open(self.save_path, "a") as outfile:
            outfile.write(json.dumps(data)+'\n')