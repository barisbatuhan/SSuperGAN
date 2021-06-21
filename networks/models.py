from networks.ssuper_model import SSuperModel

class DCGAN(SSuperModel):
    def __init__(self, **kwargs):
        super().__init__(use_seq_enc=False, enc_choice=None, gen_choice="dcgan",
                         local_disc_choice="dcgan", global_disc_choice=None, **kwargs)

class IntroVAE(SSuperModel):
    def __init__(self, **kwargs):
        super().__init__(use_seq_enc=False, enc_choice="vae", gen_choice="vae",
                         local_disc_choice=None, global_disc_choice=None, **kwargs)

class SSuperVAE(SSuperModel):
    def __init__(self, **kwargs):
        super().__init__(use_seq_enc=True, enc_choice=None, gen_choice="vae",
                         local_disc_choice=None, global_disc_choice=None, **kwargs)
        
class VAEGAN(SSuperModel):
    def __init__(self, **kwargs): 
        super().__init__(use_seq_enc=False, enc_choice="vae", gen_choice="dcgan",
                         local_disc_choice="dcgan", global_disc_choice=None, **kwargs)

class SSuperDCGAN(SSuperModel):
    def __init__(self, **kwargs):
        super().__init__(use_seq_enc=True, enc_choice=None, gen_choice="dcgan",
                         local_disc_choice="dcgan", global_disc_choice=None, **kwargs)
            
class SSuperGlobalDCGAN(SSuperModel):
    def __init__(self, **kwargs):
        super().__init__(use_seq_enc=True, enc_choice=None, gen_choice="dcgan",
                         local_disc_choice="dcgan", global_disc_choice="dcgan", **kwargs)
        
class FaceClozeModel(SSuperModel):
    def __init__(self, **kwargs):
        super().__init__(use_seq_enc=True, enc_choice="vae", gen_choice=None,
                         local_disc_choice=None, global_disc_choice=None, **kwargs)

        