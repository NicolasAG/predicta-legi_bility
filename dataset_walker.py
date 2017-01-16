import os, json, re
import numpy as np

DATA_PATH = "../data/dstc2-3/scripts"

class dataset_walker(object):
    def __init__(self, dataset, cost_function=None, goal=None):
        """
        Constructor.
        :param dataset: the name of the dataset to load. Can be string or array of strings or JSON.
        :param cost_function: takes a Call object and returns a positive cost for that Call.
        :param goal: takes a Call object and returns a Goal (TBD) for that Call.
        """
        self.cost_function = cost_function
        self.goal = goal

        if "[" in dataset :
            self.datasets = json.loads(dataset)
        elif type(dataset) == type([]) :
            self.datasets= dataset
        else:
            self.datasets = [dataset]
            self.dataset = dataset
        # self.install_root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
        self.install_root = DATA_PATH
        self.dataset_session_lists = [os.path.join(self.install_root, 'config', dataset+'.flist') for dataset in self.datasets]

        # self.labels = True
        install_parent = os.path.dirname(self.install_root)
        self.dataroot = os.path.join(install_parent,'data')

        # load dataset (list of calls)
        self.session_list = []
        for dataset_session_list in self.dataset_session_lists :
            f = open(dataset_session_list)
            for line in f:
                line = line.strip()
                #line = re.sub('/',r'\\',line)
                #line = re.sub(r'\\+$','',line)
                if (line in self.session_list):
                    raise RuntimeError,'Call appears twice: %s' % (line)
                self.session_list.append(line)
            f.close()   
        
    def __iter__(self):
        for session_id in self.session_list:
            session_id_list = session_id.split('/')
            session_dirname = os.path.join(self.dataroot,*session_id_list)
            applog_filename = os.path.join(session_dirname,'log.json')
            # if (self.labels):
            labels_filename = os.path.join(session_dirname,'label.json')
            if (not os.path.exists(labels_filename)):
                raise RuntimeError,'Cant score : cant open labels file %s' % (labels_filename)
            # else:
            #     labels_filename = None
            call = Call(applog_filename, labels_filename, self.cost_function, self.goal)
            call.dirname = session_dirname
            yield call
    def __len__(self, ):
        return len(self.session_list)
    

class Call(object):
    def __init__(self, applog_filename, labels_filename, cost_function=None, goal=None):
        self.applog_filename = applog_filename
        self.labels_filename = labels_filename
        # Log file
        f = open(applog_filename)
        self.log = json.load(f)
        f.close()
        # Labels file
        f = open(labels_filename)
        self.labels = json.load(f)
        f.close()

        # Goal / Target / EndState
        if goal:
            self.goal = goal(self)
        else:
            self.goal = 1 if self.labels['task-information']['feedback']['success'] else 0  # default goal is binary: success or not.

        # Cost & Predictability
        if cost_function:
            self.cost = cost_function(self)
        else:
            self.cost = len(self) * 2  # default cost function is the number of utterances in the dialogue.
        self.predictability = np.exp(-self.cost)

        # Legibility: cannot compute it without the kowledge of all other calls, see `main.py` where we compute it
        self.legibility = 0

    def __iter__(self):
        if (self.labels_filename != None):
            for (log,labels) in zip(self.log['turns'],self.labels['turns']):
                yield (log,labels)
        else: 
            for log in self.log['turns']:
                yield (log,None)
                
    def __len__(self, ):
        return len(self.log['turns'])

