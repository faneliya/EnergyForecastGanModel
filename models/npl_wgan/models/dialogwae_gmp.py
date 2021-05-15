import os
import sys
parentPath = os.path.abspath("../..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from modules import MixVariation
from models.npl_wgan.models.dialogwae import DialogWAE


class DialogWAE_GMP(DialogWAE):
    def __init__(self, config, vocab_size, PAD_token=0):
        super(DialogWAE_GMP, self).__init__(config, vocab_size, PAD_token)
        self.n_components = config['n_prior_components']
        self.gumbel_temp = config['gumbel_temp']
        
        self.prior_net = MixVariation(config['n_hidden'], config['z_size'], self.n_components, self.gumbel_temp) # p(e|c)
           
   
    

    


